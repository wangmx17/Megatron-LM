# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from functools import lru_cache
import time
import torch

from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core import mpu
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import reorder_cp_chunk_seq_order


@lru_cache(maxsize=16)
def calc_chunks_with_stride(cu_seqlen, moba_chunk_size, kernel_stride):
    """
    计算需要 MOBA 注意力的 chunks，支持 stride。
    返回:
        - filtered_indices: 用于直接索引 k 的索引。
        - cu_seqlens_compressed: 压缩后的累积序列长度。
    """
    # 1. 计算每个序列的长度
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # 2. 计算每个序列的 chunk 起始位置 (考虑 stride)
    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (max_seq_len - moba_chunk_size) // kernel_stride + 1  # 修正公式
    chunk_start_offsets = torch.arange(0, max_num_chunks_per_seq * kernel_stride, kernel_stride, device=cu_seqlen.device) # [0, 16, 32, ..., max_num_chunks_per_seq * kernel_stride - 16]
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]  # [batch_size, max_num_chunks_per_seq] # seq_start + chunk_offset_compared_to_seq_start

    # 3. 过滤掉超出序列长度的 chunk 和非完整大小的 chunk
    chunk_end_in_seq = chunk_start_in_seq + moba_chunk_size
    valid_chunk_mask = (chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None]))  # 完整 chunk

    # 4. 根据 valid_chunk_mask 过滤有效的 chunk 起始位置
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]  # [num_valid_chunks]
    del chunk_start_in_seq
    # 5. 生成 filtered_indices
    chunk_indices = torch.arange(
        0, moba_chunk_size, device=cu_seqlen.device
    )[None, :]  # [1, moba_chunk_size]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices  # [num_valid_chunks, moba_chunk_size]
    filtered_indices = filtered_indices.view(-1)  # 展平为一维索引

    # 6. 计算压缩后的累积序列长度
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)  # 每个 batch 的有效 chunk 数量
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0) # index?
    del num_filtered_chunks_per_batch, chunk_start_offsets, seq_starts, chunk_end_in_seq, valid_chunk_mask, chunk_indices, valid_chunk_starts
    # torch.cuda.empty_cache()
    return filtered_indices, cu_seqlens_compressed

def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens=None,
    cp_cu_seqlens_q: torch.Tensor = None,
    cp_max_seqlen_q: int = None,
) -> torch.Tensor:
    """Calculate topk indices using infllmv2_stage1 and max_pooling."""
    from infllm_v2 import infllmv2_attn_stage1, max_pooling_1d_varlen

    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1
        
        # Always prefilling stage
        # Compute cache_lens directly without CPU-GPU sync
        if cp_cu_seqlens_q is not None:
            cache_lens = (cp_cu_seqlens_q[1:] - cp_cu_seqlens_q[:-1]) - (cu_seqlens_q[1:] - cu_seqlens_q[:-1])
        else:
            cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=q.device)

        # Calculate attention score
        score = infllmv2_attn_stage1(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=True
        )
        
        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_seqlen_k,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride
        )  # shape: [num_heads, total_q_len, num_blocks]

        # get topk
        # if mpu.get_context_parallel_world_size() == 1:
        topk = min(topk, block_score.shape[-1])
        # print_rank_0(f"[NSA] block_score: {block_score.shape}")
        # print_rank_0(f"[NSA] topk: {topk}")
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx = topk_idx.to(torch.int32)
        
        # Clean up intermediate tensors
        del score, block_score, cache_lens
        # torch.cuda.empty_cache()
        
    return topk_idx


class CompressK(torch.nn.Module):
    def __init__(self, head_num_k, head_dim, kernel_size, kernel_stride=16):
        """
        压缩K模块，使用mean pooling方式
        Args:
            head_num_k: K头的数量
            head_dim: 每个头的维度
            kernel_size: 每个chunk的大小
            kernel_stride: 分块时的步长
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride

    def calc_cu_seqlens_compressed(self, cu_seqlens):
        return calc_chunks_with_stride(cu_seqlens, self.kernel_size, self.kernel_stride)[1]

    def forward(self, k: torch.Tensor, cu_seqlens):
        """
        前向传播，压缩K
        Args:
            k: 输入的K张量
            cu_seqlens: 累积序列长度
        Returns:
            compress_k: 压缩后的K
            cu_seqlens_compressed: 压缩后的累积序列长度
        """
        # 计算chunk相关信息，支持stride
        # return: 
        #   filtered_k_indices = 直接把每个block的key的indice都拼在一起, [seq_len]
        #   cu_seqlens_compressed = 考虑每个block的key已经合并了，新的cu_seqlen会长什么样    
        filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )
        # 提取过滤后的k
        # 上面filtered_k_indices是index, 这里把对应的元素抽取出来, [seq_len, head_num, head_dim]
        filtered_k = k.index_select(0, filtered_k_indices.view(-1))

        # 分块
        # 因为前面的filtered_k是展平的，挪一下形状做mean pooling
        filtered_k = filtered_k.view(filtered_k.shape[0] // self.kernel_size, self.kernel_size, self.head_num_k, self.head_dim)  #[l, block_size, h, d]
        
        # Mean pooling
        compress_k = filtered_k.mean(dim=1)  # [l, h, d]

        del filtered_k, filtered_k_indices
        # torch.cuda.empty_cache()

        # 最后compress_k是压缩后的k, cu_seqlens_compressed是压缩后的累积序列长度
        return compress_k, cu_seqlens_compressed


def cat_output_list(output_list, packed_seq_params, world_size):
    # 就是get_query_list的逆过程
    head_dest_indices, tail_dest_indices = packed_seq_params.cp_query_indices_w_span_attn
    _shape = list(output_list[0].shape)
    _shape[0] = _shape[0] * 2
    output = torch.empty(_shape, dtype=output_list[0].dtype, device=output_list[0].device)
    output.index_copy_(0, head_dest_indices, output_list[0])
    output.index_copy_(0, tail_dest_indices, output_list[1])
    return output

class NSADotProductAttention(DotProductAttention):
    """Native Sparse Attention implementation."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        window_size: int = 0,
        kernel_size: int = 32,
        kernel_stride: int = 16,
        block_size: int = 64,
        topk = 32,
        init_blocks: int = 1,
        local_blocks: int = 0,
        nsa_stage1_nope: bool = False,
        use_span_based_attn: bool = False,
        **kwargs
    ):
        # Initialize parent without NSA params
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            **kwargs
        )
        
        # Set NSA specific parameters
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.block_size = block_size
        self.init_blocks = init_blocks
        self.local_blocks = self.window_size // self.block_size
        self.topk = topk
        self.nsa_stage1_nope = nsa_stage1_nope
        self.use_span_based_attn = use_span_based_attn
        
        self.compress_k = CompressK(
            self.num_query_groups_per_partition, 
            self.hidden_size_per_attention_head,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride
        )
        
        self.dropout_p = config.attention_dropout if ('attention_dropout' not in kwargs or kwargs['attention_dropout'] is None) else kwargs['attention_dropout']
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 从transformers 的qwen2中拷贝过来的初始化
        std = 0.02
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attn_mask_type=AttnMaskType.padding,
        query_lengths=None,
        key_lengths=None,
        packed_seq_params=None,
        attention_bias=None, # currently for compatibility
        query_nope=None,
        key_nope=None,
        gate_score=None,
    ):
        # 正确的维度排列
        rank_to_log = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        assert packed_seq_params is not None, "需要cu_seqlens!"

        # Context Parallel, Gather and Reorder
        context_parallel_size = mpu.get_context_parallel_world_size()
        ori_packed_seq_params = packed_seq_params

        if context_parallel_size > 1:
            # gather keys, values, key_nopes
            key_layer_whole = gather_from_sequence_parallel_region(key_layer.contiguous(), group=mpu.get_context_parallel_group())
            value_layer_whole = gather_from_sequence_parallel_region(value_layer.contiguous(), group=mpu.get_context_parallel_group())
            
            if key_nope is not None and query_nope is not None:
                key_nope_whole = gather_from_sequence_parallel_region(key_nope.contiguous(), group=mpu.get_context_parallel_group())

            # reorder keys, values, key_nopes
            # preprare: key_layer_list, value_layer_list, query_layer_list, query_nope_list, key_nope_list, packed_seq_params_list
            if not self.use_span_based_attn:
                key_layer = reorder_cp_chunk_seq_order(key_layer_whole, seq_dim=0, num_chunks=context_parallel_size)
                value_layer = reorder_cp_chunk_seq_order(value_layer_whole, seq_dim=0, num_chunks=context_parallel_size)

                packed_seq_params_head, packed_seq_params_tail = packed_seq_params.cp_packed_params_wo_span_attn
                packed_seq_params_list = [packed_seq_params_head, packed_seq_params_tail]
                
                block_size = packed_seq_params_head.max_seqlen_q
                query_layer_list = [query_layer[:block_size], query_layer[block_size:]]
                
                key_layer_list = [key_layer[:packed_seq_params_head.cu_seqlens_kv[-1]], key_layer[:packed_seq_params_tail.cu_seqlens_kv[-1]]]
                value_layer_list = [value_layer[:packed_seq_params_head.cu_seqlens_kv[-1]], value_layer[:packed_seq_params_tail.cu_seqlens_kv[-1]]]
                
                # nope
                query_nope_list = [query_nope[:block_size], query_nope[block_size:]] if query_nope is not None else [None, None]
                if key_nope is not None and query_nope is not None:
                    _key_nope = reorder_cp_chunk_seq_order(key_nope_whole, seq_dim=0, num_chunks=context_parallel_size)
                    key_nope_list = [_key_nope[:packed_seq_params_head.cu_seqlens_kv[-1]], _key_nope[:packed_seq_params_tail.cu_seqlens_kv[-1]]]
                else:
                    key_nope_list = [None, None]
                
            else:
                packed_seq_params_head, packed_seq_params_tail, kv_head_indices, kv_tail_indices = packed_seq_params.cp_packed_params_w_span_attn
                packed_seq_params_list = [packed_seq_params_head, packed_seq_params_tail]

                # key & value
                _kv_both_indices = torch.cat((kv_head_indices, kv_tail_indices), dim=0)
                _key_both = key_layer_whole.index_select(0, _kv_both_indices)
                _value_both = value_layer_whole.index_select(0, _kv_both_indices)

                _kv_head_len = kv_head_indices.numel()
                assert _kv_head_len == packed_seq_params_head.cu_seqlens_kv[-1], "kv_head_len != packed_seq_params_head.cu_seqlens_kv[-1]"
                assert (_kv_both_indices.numel() - _kv_head_len) == packed_seq_params_tail.cu_seqlens_kv[-1], "kv_head_len != packed_seq_params_tail.cu_seqlens_kv[-1]"
                key_layer_list = [_key_both.narrow(0, 0, _kv_head_len), _key_both.narrow(0, _kv_head_len, _key_both.size(0) - _kv_head_len)]
                value_layer_list = [_value_both.narrow(0, 0, _kv_head_len), _value_both.narrow(0, _kv_head_len, _value_both.size(0) - _kv_head_len)]
                
                # query
                query_head_indices, query_tail_indices = packed_seq_params.cp_query_indices_w_span_attn
                _query_both_indices = torch.cat((query_head_indices, query_tail_indices), dim=0)
                _query_both = query_layer.index_select(0, _query_both_indices)
                
                _query_head_len = query_head_indices.numel()
                query_layer_list = [_query_both.narrow(0, 0, _query_head_len), _query_both.narrow(0, _query_head_len, _query_both.size(0) - _query_head_len)]

                # nope
                if key_nope is not None and query_nope is not None:
                    _key_nope_both = key_nope_whole.index_select(0, _kv_both_indices)
                    key_nope_list = [_key_nope_both.narrow(0, 0, _kv_head_len), _key_nope_both.narrow(0, _kv_head_len, _key_nope_both.size(0) - _kv_head_len)]
                    _query_nope_both = query_nope.index_select(0, _query_both_indices)
                    query_nope_list = [_query_nope_both.narrow(0, 0, _query_head_len), _query_nope_both.narrow(0, _query_head_len, _query_nope_both.size(0) - _query_head_len)]
                else:
                    key_nope_list = [None, None]
                    query_nope_list = [None, None]
            
        else:
            packed_seq_params_list = [packed_seq_params]
            query_layer_list = [query_layer]
            key_layer_list = [key_layer]
            value_layer_list = [value_layer]
            query_nope_list = [query_nope] if query_nope is not None else [None]
            key_nope_list = [key_nope] if key_nope is not None else [None]

        
        outputs = []
        for i, (query_layer, query_nope, key_layer, key_nope, value_layer, packed_seq_params) in enumerate(zip(query_layer_list, query_nope_list, key_layer_list, key_nope_list, value_layer_list, packed_seq_params_list)):
            if packed_seq_params.max_seqlen_q == None:
                seqlens_q = packed_seq_params.cu_seqlens_q[1:] - packed_seq_params.cu_seqlens_q[:-1]
                max_seqlen_q = seqlens_q.max().item()
            else:
                max_seqlen_q = packed_seq_params.max_seqlen_q
                
            if packed_seq_params.max_seqlen_kv == None:
                seqlens_kv = packed_seq_params.cu_seqlens_kv[1:] - packed_seq_params.cu_seqlens_kv[:-1]
                max_seqlen_kv = seqlens_kv.max().item()
            else:
                max_seqlen_kv = packed_seq_params.max_seqlen_kv
            
            # --- stage 1 ----
            key_stage1 = key_layer
            query_stage1 = query_layer
            if self.nsa_stage1_nope:
                # use nope key and query for nsa stage1 nope
                assert query_nope is not None and key_nope is not None, "query_nope and key_nope are required for NSA stage1 NOPE"
                key_stage1 = key_nope
                query_stage1 = query_nope

            compressed_k, compressed_cu_seqlens = self.compress_k(key_stage1, packed_seq_params.cu_seqlens_kv)
            compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        
            if mpu.get_context_parallel_world_size() == 1:
                cp_cu_seqlens_q = None
                cp_max_seqlen_q = None
            else:
                # !!! only for non-span based attention !!!
                # as cu_seqlen_kv masks the real cu_seqlen_q span
                # only for q-idx
                cp_cu_seqlens_q = packed_seq_params.cu_seqlens_kv
                cp_max_seqlen_q = max_seqlen_kv

            topk_idx = compressed_attention(
                query_stage1,
                compressed_k,
                compressed_k, # placeholder
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.topk,
                packed_seq_params.cu_seqlens_q,
                compressed_cu_seqlens,
                max_seqlen_q,
                compressed_seqlens.max().item(),
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cp_cu_seqlens_q=cp_cu_seqlens_q,
                cp_max_seqlen_q=cp_max_seqlen_q,
            )

            # --- stage 2 ----
            from infllm_v2 import infllmv2_attn_varlen_func

            topk_attn_output = infllmv2_attn_varlen_func(
                query_layer,
                key_layer,
                value_layer,
                packed_seq_params.cu_seqlens_q,
                packed_seq_params.cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                dropout_p=self.dropout_p,
                deterministic=False,
                softmax_scale=None,
                causal=True,
                return_attn_probs=False,
                topk_idx=topk_idx
            )

            # Clean up intermediate tensors
            del compressed_k, compressed_cu_seqlens, compressed_seqlens, topk_idx
            # torch.cuda.empty_cache()

            outputs.append(topk_attn_output)

        if len(outputs) == 1:
            return outputs[0]

        # 带span-based-attn的时候，output需要重新组合一下
        if self.use_span_based_attn:
            concat_topk_attn_output = cat_output_list(outputs, ori_packed_seq_params, context_parallel_size)
        else:
            concat_topk_attn_output = torch.cat(outputs, dim=0) # TODO
        
        return concat_topk_attn_output


class NSAHybridDotProductAttention(torch.nn.Module):
    """Hybrid attention:
    - Use `TEDotProductAttention` for short sequences
    - Use `NSADotProductAttention` for long sequences

    The routing is decided by a length threshold computed from NSA params
    or provided explicitly via `hybrid_threshold`.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        *,
        # NSA params (defaults match NSADotProductAttention)
        window_size: int = 0,
        kernel_size: int = 32,
        kernel_stride: int = 16,
        block_size: int = 64,
        topk: int = 32,
        init_blocks: int = 1,
        local_blocks: int = 0,
        nsa_stage1_nope: bool = False,
        use_span_based_attn: bool = False,
        # Hybrid routing
        hybrid_threshold: int = None,
        **kwargs,
    ):
        super().__init__()

        # Build short-path TE attention
        try:
            from megatron.core.extensions.transformer_engine import (
                TEDotProductAttention as _TEAttn,
            )
        except Exception as exc:  # pragma: no cover - environment specific
            raise ImportError(
                "Transformer-Engine is required for NSAHybridDotProductAttention short path"
            ) from exc

        self.config = config
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # TE attention module for short sequences
        self.short_attn = _TEAttn(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=kwargs.get("attention_dropout", None),
            softmax_scale=kwargs.get("softmax_scale", None),
            k_channels=kwargs.get("k_channels", None),
            v_channels=kwargs.get("v_channels", None),
            cp_comm_type=kwargs.get("cp_comm_type", "p2p"),
        )

        # NSA attention module for long sequences
        self.nsa_stage1_nope = nsa_stage1_nope # for detecting nsa core in attention forward
        self.long_attn = NSADotProductAttention(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            window_size=window_size,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            nsa_stage1_nope=nsa_stage1_nope,
            use_span_based_attn=use_span_based_attn,
        )

        self.length_threshold = hybrid_threshold 

        # Backward-compat checkpoint loading: allow old TE checkpoints to load
        # where TE attention lived directly under `core_attention` without `short_attn`.
        def _remap_te_keys(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            # Try to map exact match first
            dst_key = f"{prefix}short_attn._extra_state"
            src_key = f"{prefix}_extra_state"
            
            if dst_key not in state_dict:
                if src_key in state_dict:
                    state_dict[dst_key] = state_dict[src_key]
                else:
                    # Handle distributed checkpoint shards (e.g. ..._extra_state/shard_0_1)
                    # Scan for keys that start with the source prefix
                    keys_to_remap = [k for k in state_dict.keys() if k.startswith(src_key)]
                    
                    if keys_to_remap:
                        for k in keys_to_remap:
                            # Construct destination key preserving the shard suffix
                            # e.g. "..._extra_state/shard_0_1" -> "...short_attn._extra_state/shard_0_1"
                            suffix = k[len(src_key):]
                            new_dst_key = f"{dst_key}{suffix}"
                            state_dict[new_dst_key] = state_dict[k]
                    else:
                        # Only create placeholder if no related keys found at all
                        state_dict[dst_key] = torch.empty(0, dtype=torch.uint8)
                    # _extra_state is used for fp8, we just ignore it anyway :P

        self._register_load_state_dict_pre_hook(_remap_te_keys, with_module=True)

    def _get_max_seqlen(self, query_layer, packed_seq_params):
        if packed_seq_params is None:
            batch_size, max_seqlen = query_layer.shape[1], query_layer.shape[0]
            return max_seqlen
        _max_seqlen = packed_seq_params.cu_seqlens_q[1:] - packed_seq_params.cu_seqlens_q[:-1]
        _max_seqlen = _max_seqlen.max().item()
        return _max_seqlen

    def forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        query_lengths=None,
        key_lengths=None,
        packed_seq_params=None,
        attention_bias=None,
        query_nope=None,
        key_nope=None,
        **kwargs,
    ):
        max_len = self._get_max_seqlen(query_layer, packed_seq_params)

        # Route: TE for short sequences or when NSA cannot run (no packed seq params)
        assert packed_seq_params is not None, "packed_seq_params is required for NSAHybridDotProductAttention"
        use_short_te = max_len <= self.length_threshold

        if use_short_te:
            # Align K/V channel sizes for TE if needed (mirrors SelfAttention logic)
            if key_layer.shape[-1] != value_layer.shape[-1]:
                value_layer = torch.nn.functional.pad(
                    value_layer,
                    [0, key_layer.shape[-1] - value_layer.shape[-1]],
                    value=0,
                )
            return self.short_attn(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        # Long path: NSA requires packed_seq_params
        return self.long_attn(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attn_mask_type=attn_mask_type,
            query_lengths=query_lengths,
            key_lengths=key_lengths,
            packed_seq_params=packed_seq_params,
            attention_bias=attention_bias,
            query_nope=query_nope,
            key_nope=key_nope,
        )