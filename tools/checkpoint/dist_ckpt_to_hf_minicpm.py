# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

from functools import partial
from typing import List, Optional, Tuple

import torch

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.training import get_args, get_timers, get_tokenizer, inprocess_restart, pretrain, print_rank_0
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider
from megatron.core import parallel_state
from megatron.core.datasets.modelbest_sdk_dataset_builder import ModelBestSDKDatasetBuilder
from megatron.core.packed_seq_params import PackedSeqParams

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

from megatron.training.initialize import initialize_megatron
from megatron.training.training import setup_model_and_optimizer
import os

stimer = StragglerDetector()


if __name__ == "__main__":

    initialize_megatron(extra_args_provider=None)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(partial(model_provider, gpt_builder), 
                        model_type=ModelType.encoder_or_decoder)
    args = get_args()
    model_real = model[0]

    # extract megatron state dict
    state_dict_megatron = model_real.state_dict()
    new_sd = dict()
    for k in state_dict_megatron:
        if "_extra" in k:
            continue
        new_sd[k] = state_dict_megatron[k]
        print(k, state_dict_megatron[k].shape)

    # torch.save({"model": new_sd}, args.save)
    # param name mapping
    state_dict_hf = dict()

    state_dict_hf["model.embed_tokens.weight"] = state_dict_megatron["embedding.word_embeddings.weight"]
    state_dict_hf["model.norm.weight"] = state_dict_megatron["decoder.final_layernorm.weight"]
    if args.untie_embeddings_and_output_weights:
        state_dict_hf["lm_head.weight"] = state_dict_megatron["output_layer.weight"]

    assert args.num_attention_heads % args.num_query_groups == 0
    assert args.hidden_size % args.num_attention_heads == 0

    num_query_heads_per_group = args.num_attention_heads // args.num_query_groups

    for layer_idx in range(args.num_layers):
        state_dict_hf[f"model.layers.{layer_idx}.input_layernorm.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight"]
        qkv_proj = state_dict_megatron[f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"]

        qkv_proj_split = torch.split(qkv_proj, split_size_or_sections=args.kv_channels, dim=0)

        q_proj_list, k_proj_list, v_proj_list, gate_proj_list = [], [], [], []
        for i in range(args.num_query_groups):
            if args.elementwise_attn_output_gate:
                q_proj_list.extend(qkv_proj_split[(num_query_heads_per_group * 2 + 2) * i: (num_query_heads_per_group * 2 + 2) * i + num_query_heads_per_group])
                gate_proj_list.extend(qkv_proj_split[(num_query_heads_per_group * 2 + 2) * i + num_query_heads_per_group: (num_query_heads_per_group * 2 + 2) * i + num_query_heads_per_group * 2])
                k_proj_list.append(qkv_proj_split[(num_query_heads_per_group * 2 + 2) * i + num_query_heads_per_group * 2])
                v_proj_list.append(qkv_proj_split[(num_query_heads_per_group * 2 + 2) * i + num_query_heads_per_group * 2 + 1])
            else:
                q_proj_list.extend(qkv_proj_split[(num_query_heads_per_group + 2) * i: (num_query_heads_per_group + 2) * i + num_query_heads_per_group])
                k_proj_list.append(qkv_proj_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group])
                v_proj_list.append(qkv_proj_split[(num_query_heads_per_group + 2) * i + num_query_heads_per_group + 1])
            
        q_proj = torch.cat(q_proj_list, dim=0)
        k_proj = torch.cat(k_proj_list, dim=0)
        v_proj = torch.cat(v_proj_list, dim=0)
        if len(gate_proj_list) > 0:
            merge_q_proj_list = []
            for i in range(len(q_proj_list)):
                merge_q_proj_list.append(q_proj_list[i])
                merge_q_proj_list.append(gate_proj_list[i])
            q_proj = torch.cat(merge_q_proj_list, dim=0)

        state_dict_hf[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v_proj
        state_dict_hf[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight"]

        if args.num_experts and args.moe_layer_freq[layer_idx] == 1:
            # router
            state_dict_hf[f"model.layers.{layer_idx}.mlp.gate.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.router.weight"]
            if args.moe_router_enable_expert_bias:
                state_dict_hf[f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.router.expert_bias"]
            # shared experts
            state_dict_hf[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight"]
            if args.moe_shared_expert_intermediate_size is not None:
                linear1_fc_weight = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc1.weight"]
                gate_proj, up_proj = torch.split(linear1_fc_weight, split_size_or_sections=(linear1_fc_weight.shape[0] // 2), dim=0)
                state_dict_hf[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"] = gate_proj
                state_dict_hf[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"] = up_proj
                state_dict_hf[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc2.weight"]

            if args.expert_model_parallel_size != 1:
                ep_idx = parallel_state.get_expert_model_parallel_rank()
                assert args.num_experts % args.expert_model_parallel_size == 0
                num_local_experts = args.num_experts // args.expert_model_parallel_size
            else:
                ep_idx = 0
                num_local_experts = args.num_experts

            for idx_expert in range(num_local_experts):
                linear1_fc_weight = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight{idx_expert}"]
                gate_proj, up_proj = torch.split(linear1_fc_weight, split_size_or_sections=(linear1_fc_weight.shape[0] // 2), dim=0)
                state_dict_hf[f"model.layers.{layer_idx}.mlp.experts.{num_local_experts * ep_idx + idx_expert}.gate_proj.weight"] = gate_proj
                state_dict_hf[f"model.layers.{layer_idx}.mlp.experts.{num_local_experts * ep_idx + idx_expert}.up_proj.weight"] = up_proj
                state_dict_hf[f"model.layers.{layer_idx}.mlp.experts.{num_local_experts * ep_idx + idx_expert}.down_proj.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2.weight{idx_expert}"]
        else:
            state_dict_hf[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight"]
            linear1_fc_weight = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight"]
            gate_proj, up_proj = torch.split(linear1_fc_weight, split_size_or_sections=(linear1_fc_weight.shape[0] // 2), dim=0)
            state_dict_hf[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = gate_proj
            state_dict_hf[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = up_proj
            state_dict_hf[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = state_dict_megatron[f"decoder.layers.{layer_idx}.mlp.linear_fc2.weight"]

    # save and generate hf repository
    os.makedirs(args.save, exist_ok=True)
    for k in state_dict_hf:
        state_dict_hf[k] = state_dict_hf[k].clone().contiguous()
    if args.expert_model_parallel_size == 1:
        torch.save(state_dict_hf, os.path.join(args.save, "pytorch_model.bin"))
    else:
        torch.save(state_dict_hf, os.path.join(args.save, f"pytorch_model.ep_{ep_idx}.bin"))
