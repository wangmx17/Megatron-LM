import torch
from megatron.core import parallel_state


class StatesLoggingHelper:
    """Helper class for logging statistics of hidden_states."""

    tracker = {}
    curr_iteration = 0
    should_log_hidden_rms = False
    should_log_attention_max_logit = False

    @staticmethod
    def save_max_attention_logit_to_tracker(
        max_attention_logit: torch.Tensor,
        layer_number: int,
    ):
        if layer_number not in StatesLoggingHelper.tracker:
            StatesLoggingHelper.tracker[layer_number] = {}
        StatesLoggingHelper.tracker[layer_number]["max_attention_logit"] = max_attention_logit.detach()

    @staticmethod
    def save_attn_hidden_rms_to_tracker(
        attn_hidden_states: torch.Tensor,
        layer_number: int,
    ):
        """Save the attention rms for logging.
        Args:
            attn_hidden_states (torch.Tensor): The attention hidden states tensor.
            layer_number (int): Layer index of the attention hidden states.
        """
        hidden = attn_hidden_states.detach()
        hidden_rms = (hidden.norm(dim=-1) / (hidden.shape[-1] ** 0.5)).mean()
        if layer_number not in StatesLoggingHelper.tracker:
            StatesLoggingHelper.tracker[layer_number] = {}
        StatesLoggingHelper.tracker[layer_number]["attn_rms"] = hidden_rms
        StatesLoggingHelper.tracker[layer_number]["attn_max"] = hidden.max()

    @staticmethod
    def save_mlp_hidden_rms_to_tracker(
        mlp_hidden_states: torch.Tensor,
        layer_number: int,
    ):
        """Save the mlp rms for logging.
        Args:
            mlp_hidden_states (torch.Tensor): The mlp hidden states tensor.
            layer_number (int): Layer index of the mlp hidden states.
        """
        hidden = mlp_hidden_states.detach()
        hidden_rms = (hidden.norm(dim=-1) / (hidden.shape[-1] ** 0.5)).mean()
        if layer_number not in StatesLoggingHelper.tracker:
            StatesLoggingHelper.tracker[layer_number] = {}
        StatesLoggingHelper.tracker[layer_number]["mlp_rms"] = hidden_rms
        StatesLoggingHelper.tracker[layer_number]["mlp_max"] = hidden.max()

    def reduce_states_in_tracker():
        """Reduce the states in the tracker across ranks."""
        tracker = StatesLoggingHelper.tracker
        for layer_idx in tracker:
            if "attn_rms" in tracker[layer_idx]:
                attn_rms = tracker[layer_idx]["attn_rms"]
                attn_max = tracker[layer_idx]["attn_max"]
                mlp_rms = tracker[layer_idx]["mlp_rms"]
                mlp_max = tracker[layer_idx]["mlp_max"]
                torch.distributed.all_reduce(attn_rms, op=torch.distributed.ReduceOp.AVG, group=parallel_state.get_data_parallel_group())
                torch.distributed.all_reduce(mlp_rms, op=torch.distributed.ReduceOp.AVG, group=parallel_state.get_data_parallel_group())
                torch.distributed.all_reduce(attn_max, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_data_parallel_group())
                torch.distributed.all_reduce(mlp_max, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_data_parallel_group())
            if "max_attention_logit" in tracker[layer_idx]:
                max_attn_logit = tracker[layer_idx]["max_attention_logit"]
                torch.distributed.all_reduce(max_attn_logit, op=torch.distributed.ReduceOp.AVG, group=parallel_state.get_data_parallel_group())

    def clean_statistics_in_tracker():
        """Clear the statistics in the tracker."""
        StatesLoggingHelper.tracker = {}
        StatesLoggingHelper.should_log_hidden_rms = False
        StatesLoggingHelper.should_log_attention_max_logit = False

    def track_states_metrics(iteration: int, writer, wandb_writer=None, log_hidden_rms_interval=None, log_max_attention_logit_interval=None):
        """Track the states metrics for logging."""
        StatesLoggingHelper.curr_iteration = iteration

        if (log_hidden_rms_interval is not None and iteration % log_hidden_rms_interval == 0) or \
        (log_max_attention_logit_interval is not None and iteration % log_max_attention_logit_interval == 0):
            StatesLoggingHelper.reduce_states_in_tracker()
            tracker = StatesLoggingHelper.tracker
            for layer_idx in tracker:
                if writer is not None:
                    if "attn_rms" in tracker[layer_idx]:
                        writer.add_scalar(f"attention_output_rms/layer_{layer_idx}", tracker[layer_idx]["attn_rms"], iteration)
                        writer.add_scalar(f"attention_output_max/layer_{layer_idx}", tracker[layer_idx]["attn_max"], iteration)
                        writer.add_scalar(f"mlp_output_rms/layer_{layer_idx}", tracker[layer_idx]["mlp_rms"], iteration)
                        writer.add_scalar(f"mlp_output_max/layer_{layer_idx}", tracker[layer_idx]["mlp_max"], iteration)
                    if "max_attention_logit" in tracker[layer_idx]:
                        writer.add_scalar(f"max attention logit/layer_{layer_idx}", tracker[layer_idx]["max_attention_logit"], iteration)
                if wandb_writer is not None:
                    if "attn_rms" in tracker[layer_idx]:
                        wandb_writer.log({f"attention_output_rms/layer_{layer_idx}": tracker[layer_idx]["attn_rms"]}, iteration)
                        wandb_writer.log({f"attention_output_max/layer_{layer_idx}": tracker[layer_idx]["attn_max"]}, iteration)
                        wandb_writer.log({f"mlp_output_rms/layer_{layer_idx}": tracker[layer_idx]["mlp_rms"]}, iteration)
                        wandb_writer.log({f"mlp_output_max/layer_{layer_idx}": tracker[layer_idx]["mlp_max"]}, iteration)
                    if "max_attention_logit" in tracker[layer_idx]:
                        wandb_writer.log({f"max attention logit/layer_{layer_idx}": tracker[layer_idx]["max_attention_logit"]}, iteration)

            StatesLoggingHelper.clean_statistics_in_tracker()

        if log_hidden_rms_interval is not None and iteration % log_hidden_rms_interval == log_hidden_rms_interval - 1:
            StatesLoggingHelper.should_log_hidden_rms = True

        if log_max_attention_logit_interval is not None and iteration % log_max_attention_logit_interval == log_max_attention_logit_interval - 1:
            StatesLoggingHelper.should_log_attention_max_logit = True