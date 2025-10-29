# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict, List
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import gymnasium as gym

def extract_many_from_batch(batch, pattern: str):
    filtered_dict = {}
    regex = re.compile(pattern)
    for key, value in batch.items():
        if regex.search(key):
            filtered_dict[key] = value
    if len(filtered_dict) == 0:
        raise ValueError(
            f"Couldn't find the regex key '{pattern}' in the batch. "
            f"'Available keys are: {list(batch.keys())}"
        )
    return filtered_dict

def flatten_time_dim_into_channel_dim(
    tensor: torch.Tensor, has_view_axis: bool = True):
    if has_view_axis:
        bs, v, t, ch = tensor.shape[:4]
        return tensor.view(bs, v, t * ch, *tensor.shape[4:])
    bs, t, ch = tensor.shape[:3]
    return tensor.view(bs, t * ch, *tensor.shape[3:])

def extract_from_spec(spec: gym.spaces.Dict, key, missing_ok: bool = False):
    if key not in list(spec.keys()):
        if missing_ok:
            return None
        raise ValueError(
            f"Couldn't find '{key}' in the space. "
            f"Available keys are: {list(spec.keys())}"
        )
    return spec[key]
        

def stack_tensor_dictionary(
    tensor_dict: Dict[str, torch.Tensor],
    dim: int,
    key_order: List[str]= [
            "left_shoulder_rgb",
            "right_shoulder_rgb",
            "wrist_rgb",
            "front_rgb",
        ]
) -> torch.Tensor:
    """
    Stack tensors from `tensor_dict` along `dim`, pulling them out
    in exactly the order given by `key_order`.
    """
    missing = set(key_order) - set(tensor_dict.keys())
    if missing:
        raise KeyError(f"stack_tensor_dictionary: missing keys {missing!r}")
    ordered_tensors = [tensor_dict[k] for k in key_order]
    return torch.stack(ordered_tensors, dim=dim)
# def stack_tensor_dictionary(tensor_dict: Dict[str, torch.Tensor], dim: int):
#     return torch.stack(list(tensor_dict.values()), dim)


class ImgChLayerNorm(nn.Module):
    def __init__(self, num_channels, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def layernorm_for_cnn(num_channels):
    return nn.GroupNorm(1, num_channels)


def identity_cls(num_channels):
    return nn.Identity()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0
