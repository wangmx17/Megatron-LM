import os
import sys
from modelbest_sdk.dataset.thrift_wrapper.dataset_checkpoint import DatasetCheckpointList
from collections import defaultdict

ckpt_dir = sys.argv[1]
global_checkpoint = []
for ckpt_file in os.listdir(ckpt_dir):
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    ckpt = DatasetCheckpointList.load_from_file(ckpt_path)
    global_checkpoint.append(ckpt)

print(">>> merging checkpoint ...")
merged_ckpt = None
for ckpt in global_checkpoint:
    ckpt: DatasetCheckpointList
    if merged_ckpt is None:
        merged_ckpt = ckpt
    else:
        merged_ckpt.merge(ckpt)

print(">>> merging checkpoint done ...")
progress_dict = {}

for checkpoint in merged_ckpt.checkpoint_list:
    dataset_info = checkpoint.dataset_info
    used = checkpoint.used
    consumed_samples_per_epoch = defaultdict(int)
    for chunk, index_set in used.active.items():
        consumed_samples_per_epoch[chunk.epoch] += len(index_set)
    for epoch, chunk_set in used.done.items():
        for chunk in chunk_set:
            consumed_samples_per_epoch[chunk.epoch] += (chunk.stop - chunk.start)
    print(dataset_info.path, consumed_samples_per_epoch)