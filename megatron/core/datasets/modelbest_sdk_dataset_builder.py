# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
from typing import Any, Callable, Iterable, List, Optional, Type, Union

from megatron.core import mpu
from modelbest_sdk.dataset.batch_packer.batch_packer_factory import MEGATRON_BATCH_PACKER
# from megatron.core.datasets.mtp_dataset import MTPBatchPacker
from modelbest_sdk.dataset.modelbest_dataloader import ModelbestDataloader
from modelbest_sdk.dataset.sampler.sampler_factory import WEIGHTED_MEGATRON_SAMPLER
from modelbest_sdk.dataset.segment.segment_factory import FIXED_LENGTH_SEGMENT, CONDITIONAL_FIXED_LENGTH_SEGMENT, NO_SEGMENT
from modelbest_sdk.dataset.thrift_wrapper.dataset_context import DatasetContext
from modelbest_sdk.dataset.thrift_wrapper.dataset_checkpoint import DatasetInfoList, DatasetInfo


class ModelBestSDKDatasetBuilder(object):
    """Builder class for the ModelBestSDKDatasetBuilder

    Args:
        config (DatasetContext): The config object which informs dataset creation
    """

    def __init__(
        self,
        config: DatasetContext,
    ):
        self.config = config

    def build(self, args) -> List[Optional[ModelbestDataloader]]:
        if args.data_path is not None:
            dataloader = self._build(args.data_path, args)
            return dataloader, None, None
        else:
            if args.train_data_path is not None:
                train_dataloader = self._build(args.train_data_path, args)
            else:
                train_dataloader = None

            if args.valid_data_path is not None:
                valid_dataloader = self._build(args.valid_data_path, args)
            else:
                valid_dataloader = None

            if args.test_data_path is not None:
                test_dataloader = self._build(args.test_data_path, args)
            else:
                test_dataloader = None

            return train_dataloader, valid_dataloader, test_dataloader

    def _build(self, data_path, args) -> ModelbestDataloader:
        dataset_info_list = self.parse_data_path(data_path)
        batch_packer_type = MEGATRON_BATCH_PACKER
        # if args.mtp_num_layers is not None:
        #     batch_packer_type = "mtp_batch_packer"
        # else:
        #     batch_packer_type = MEGATRON_BATCH_PACKER
        segment_type = CONDITIONAL_FIXED_LENGTH_SEGMENT

        dataloader = ModelbestDataloader(
            self.config, 
            dataset_info_list, 
            batch_size=args.micro_batch_size, 
            max_len=args.seq_length, 
            chunk_size=args.data_chunk_size, 
            num_workers=args.num_workers,
            prefetch_chunk_cnt=args.prefetch_chunk_cnt,
            prefetch_factor=args.prefetch_factor,
            cuda_prefetch=False, 
            segment_type=segment_type, 
            sampler_type=WEIGHTED_MEGATRON_SAMPLER, 
            batch_packer_type=batch_packer_type, 
            dp_group=mpu.get_data_parallel_group(), 
            clear_sampler_state=args.clear_sampler_state,
            track_drop_stats=getattr(args, 'track_drop_stats', False)
        )
        if not args.no_load_data_state:
            if args.ckpt_step:
                directory = os.path.join(args.load, 'iter_{:07d}'.format(int(args.ckpt_step)))
                dataloader.resume(directory)
            else:
                print(">>>> loading dataset ckpt ......")
                directory = os.path.join(args.load, 'release')
                dataloader.resume(directory)
        return dataloader

    def normalize(self, weights):
        total = sum(weights)
        return [w / total for w in weights]

    def parse_data_path(self, data_path):
        weights, paths = zip(*[(float(data_path[i]), data_path[i + 1].strip()) for i in range(0, len(data_path), 2)])
        weights = self.normalize(weights)
        dataset_info_list = [DatasetInfo(path=path, weight=weight) for path, weight in zip(paths, weights)]
        return DatasetInfoList(dataset_info_list)