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
    add_global_think_answer,
    add_per_task_think_answer,
    build_think_mask,
    compute_think_answer_split,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
    should_log_think_loss,
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

stimer = StragglerDetector()


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch."""
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None, None, None, None, None, None

    args = get_args()
    batch = get_batch_on_this_tp_rank(data_iterator)

    batch['think_mask'] = (
        build_think_mask(batch.get('labels')) if should_log_think_loss(args) else None
    )

    batch = get_batch_on_this_cp_rank(batch)

    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
        batch.get("dataset_id"),
        getattr(args, "current_num_real_seqs", None),
        getattr(args, "current_total_content_len", None),
        batch.get("think_mask"),
        batch["packed_seq_params"],
    )

# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor,
    task_num: int,
    dataset_id: torch.Tensor,
    position_ids: torch.Tensor,
    num_real_seqs: Optional[torch.Tensor],
    total_content_len: Optional[torch.Tensor],
    packing_eff_info: Optional[torch.Tensor],
    think_mask: Optional[torch.Tensor],
    output_tensor: torch.Tensor,
    model: Optional[GPTModel] = None,
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.float()
        losses_no_flatten = torch.sum(losses * loss_mask, dim=-1).clone().detach()
        tokens_no_flatten = torch.sum(loss_mask, dim=-1).clone().detach()

        think_split = compute_think_answer_split(args, losses, loss_mask, think_mask)

        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        # report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )
    if task_num != 0 and (args.curr_iteration + 1) % args.log_task_loss_interval == 0:
        task_tokens = torch.zeros((task_num, *dataset_id.shape), device=dataset_id.device)
        task_losses = torch.zeros((task_num, *dataset_id.shape), device=dataset_id.device)
        task_tokens.scatter_(0, dataset_id.unsqueeze(0), tokens_no_flatten.unsqueeze(0))
        task_losses.scatter_(0, dataset_id.unsqueeze(0), losses_no_flatten.unsqueeze(0))
        task_tokens = task_tokens.sum(dim=-1)
        task_losses = task_losses.sum(dim=-1)
        
        task_mask = torch.zeros((task_num, *dataset_id.shape), device=dataset_id.device)
        task_mask.scatter_(0, dataset_id.unsqueeze(0), 1)

        if (
            num_real_seqs is not None
            and total_content_len is not None
            and num_real_seqs.sum() > 0
        ):
            seq_counts_per_sample = num_real_seqs.float()
            token_counts_per_sample = total_content_len.float()
        else:
            seq_counts_per_sample = (position_ids == 0).float().sum(dim=-1).clone().detach()
            token_counts_per_sample = (position_ids >= 0).float().sum(dim=-1).clone().detach()

        task_seq_counts = torch.zeros((task_num, *dataset_id.shape), device=dataset_id.device)
        task_seq_counts.scatter_(0, dataset_id.unsqueeze(0), seq_counts_per_sample.unsqueeze(0))
        task_seq_counts = task_seq_counts.sum(dim=-1)
        task_token_counts = torch.zeros((task_num, *dataset_id.shape), device=dataset_id.device)
        task_token_counts.scatter_(0, dataset_id.unsqueeze(0), token_counts_per_sample.unsqueeze(0))
        task_token_counts = task_token_counts.sum(dim=-1)

        report = {
            'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)]),
            'task_tokens_list': task_tokens,
            'task_losses_list': task_losses,
            'task_seq_count_list': task_seq_counts,
            'task_token_count_list': task_token_counts,
        }
        if packing_eff_info is not None:
            report['packing-efficiency'] = packing_eff_info
        add_per_task_think_answer(report, think_split, task_num, dataset_id)
    else:
        report = {
            'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)]),
        }
        if packing_eff_info is not None:
            report['packing-efficiency'] = packing_eff_info

    add_global_think_answer(report, think_split)

    return loss, num_tokens, report
    # return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            dataset_id,
            num_real_seqs,
            total_content_len,
            think_mask,
            packed_seq_params,
        ) = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    with stimer:
        # tokens = torch.tensor([[1,3,5,7]], dtype=torch.int32, device=tokens.device)
        # position_ids = torch.tensor([[0,1,2,3]], dtype=torch.int32, device=tokens.device)
        # labels = torch.tensor([[3,5,1,8]], dtype=torch.int32, device=tokens.device)
        # loss_mask = torch.tensor([[1,1,1,1]], dtype=torch.int32, device=tokens.device)
        # mtp_tokens = torch.tensor([[3,5,7,5]], dtype=torch.int32, device=tokens.device)
        # mtp_labels = torch.tensor([[11,13,15,17]], dtype=torch.int32, device=tokens.device)
        # mtp_loss_mask = torch.tensor([[1,1,1,1]], dtype=torch.int32, device=tokens.device)
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                if args.log_task_loss_interval != -1 and args.data_path is not None:
                    task_num = len(args.data_path[1::2])
                else:
                    task_num = 0
                pe_info = None
                if getattr(args, 'log_packing_efficiency', False):
                    non_pad = (tokens != 0).sum().float().detach()
                    total = torch.tensor(tokens.numel(), dtype=torch.float, device=tokens.device)
                    pe_info = torch.stack([non_pad, total])
                return schedule_plan, partial(
                    loss_func,
                    loss_mask,
                    task_num,
                    dataset_id,
                    position_ids,
                    num_real_seqs,
                    total_content_len,
                    pe_info,
                    think_mask,
                    model=model,
                )
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, packed_seq_params=packed_seq_params
                )
        if args.log_task_loss_interval != -1 and args.data_path is not None:
            task_num = len(args.data_path[1::2])
        else:
            task_num = 0

    packing_eff_info = None
    if getattr(args, 'log_packing_efficiency', False):
        non_pad = (tokens != 0).sum().float().detach()
        total = torch.tensor(tokens.numel(), dtype=torch.float, device=tokens.device)
        packing_eff_info = torch.stack([non_pad, total])

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(
        loss_func,
        loss_mask,
        task_num,
        dataset_id,
        position_ids,
        num_real_seqs,
        total_content_len,
        packing_eff_info,
        think_mask,
        model=model,
    )


def is_dataset_built_on_rank(vp_stage=None):
    return is_first_or_last_pipeline_stage(vp_stage) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    if args.use_modelbest_sdk:
        from modelbest_sdk.dataset.thrift_wrapper.dataset_context import DatasetContext

        return DatasetContext(
            rank=parallel_state.get_data_parallel_rank(),
            world_size=parallel_state.get_data_parallel_world_size(),
            tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            tp_size=parallel_state.get_tensor_model_parallel_world_size(),
            pp_rank=parallel_state.get_pipeline_model_parallel_rank(),
            pp_size=parallel_state.get_pipeline_model_parallel_world_size(),
            num_workers=args.num_workers,
            dataset_config_path='',
            dataset_checkpoint_path=args.save,
            seed=args.seed + parallel_state.get_data_parallel_world_size() + parallel_state.get_data_parallel_rank()
        )
    else:
        tokenizer = get_tokenizer()

        # Sometimes --data-path is too long, instead we parse it from a file.
        blend: Optional[Tuple[List[str], Optional[List[float]]]]
        blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
        blend, blend_per_split = get_blend_and_blend_per_split(args)

        return GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=blend,
            blend_per_split=blend_per_split,
            split=args.split,
            num_dataset_builder_threads=args.num_dataset_builder_threads,
            path_to_cache=args.data_cache_path,
            mmap_bin_files=args.mmap_bin_files,
            tokenizer=tokenizer,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
            s3_cache_path=args.s3_cache_path,
        )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    if args.use_modelbest_sdk:
        train_ds, valid_ds, test_ds = ModelBestSDKDatasetBuilder(config).build(args)
    else:
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
        ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'HuggingFaceTokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
