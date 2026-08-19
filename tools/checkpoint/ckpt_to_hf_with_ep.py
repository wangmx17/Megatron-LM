import argparse
import os
import torch

# from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--num_layer", type=int, default=52)
parser.add_argument("--moe_layer_start_idx", type=int, default=1)
parser.add_argument("--ep_num", type=int, default=160)
parser.add_argument("--ep_size", type=int, default=8)
parser.add_argument("--in_dir", type=str, default="")
parser.add_argument("--save_path", type=str, default="")
args = parser.parse_args()


state_dict = torch.load(args.in_dir + "/pytorch_model.ep_0.bin")
for k in state_dict:
    print(k, state_dict[k].shape)

num_local_experts = args.ep_num // args.ep_size
for idx in range(1, args.ep_size):
    state_dict_ep = torch.load(args.in_dir + f"/pytorch_model.ep_{idx}.bin")
    for layer_idx in range(args.moe_layer_start_idx, args.num_layer):
        for expert_idx in range(num_local_experts * idx, num_local_experts * (idx + 1)):
            print(f">>> ep_idx: {idx}, expert_idx: {expert_idx}")
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"] = state_dict_ep[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"].clone()
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"] = state_dict_ep[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"].clone()
            state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"] = state_dict_ep[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"].clone()

torch.save(state_dict, args.save_path + "/pytorch_model.bin")