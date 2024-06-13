# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os, sys

local_rank = os.environ.get("LOCAL_RANK", None)
def rank0_print(*args):
    if local_rank == "0" or local_rank == 0 or local_rank is None:
        print(*args)

from collections import OrderedDict
import math
import requests
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Callable, Optional, Sequence, Tuple, List, Union
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: (H, W)
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    dtype = abs_pos.dtype
    device = abs_pos.device

    pos_2d = F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).to(dtype=dtype)
    pos_1d = torch.cat( # padding
        [
            pos_2d[0].view(tgt_size[0] * tgt_size[1], -1),
            torch.zeros((576 - tgt_size[0] * tgt_size[1], abs_pos.shape[1]), dtype=dtype, device=device),
        ],
        dim=0,
    )
    return pos_1d


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def trunc_normal(tensor, std):
    trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)

class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
        self,
        src_grid_size,
        tgt_seq_len,
        embed_dim,
        num_heads,
        kv_dim = None,
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.num_queries = tgt_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # rank0_print(f"Resampler init: grid_size {src_grid_size} | embed_dim {embed_dim} | num_heads {num_heads} | kv_dim {kv_dim}")
        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, src_grid_size)).float()
        )#.requires_grad_(False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.query = nn.Parameter(torch.zeros(1, self.num_queries, embed_dim))
        trunc_normal(self.query, std=1 / math.sqrt(embed_dim))
        self.sep_embeddings = nn.Parameter(torch.zeros((3, 1, embed_dim)))
        trunc_normal(self.sep_embeddings, std=1 / math.sqrt(embed_dim))

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.3, batch_first=True)

        self.ln_kv = norm_layer(embed_dim)
        self.ln1 = norm_layer(embed_dim)
        self.ln2 = norm_layer(embed_dim)

        self.updim_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.act = nn.GELU()
        self.proj = nn.Linear(embed_dim * 2, embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal(m.weight, std=1. / math.sqrt(m.weight.shape[1]))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, tgt_sizes, attention_mask=None):
        # ----------------------------------------------------
        # 原代码遗漏或写错的逻辑：
        # 1. 没有用`attnetion_mask`来掩盖padding部分
        # 2. 用错了对x增加的`pos_embed`, 原代码错误的创建了一个
        # 8x8的初始二维`self.pos_embed`，之后在此基础上插值，
        # 这是错的，初始的`self.pos_embed`应该是24x24的
        # 
        # 此外，我修改了一点架构设计，比如把子图分割的spetial tokens
        # 放在这里定义，方便一阶段的训练和模型保存。
        # [Edited by zhenwei - 2024-06-12 18:59]
        # ----------------------------------------------------
        pos_embeds = []
        for tgt_size in tgt_sizes:
            # rank0_print(f"[{local_rank}][{self.__class__.__name__}] tgt_size {tgt_size}")
            pos_embed = get_abs_pos(self.pos_embed, tgt_size)
            pos_embeds.append(pos_embed)
        pos_embeds = torch.stack(pos_embeds, dim=0)

        x = self.kv_proj(x)
        x = self.ln_kv(x)
        
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        # rank0_print(f"[{local_rank}][{self.__class__.__name__}] q.shape {q.shape} | x.shape {x.shape} | pos_embed.shape {pos_embeds.shape} | attention_mask.shape {attention_mask.shape}")
        # `MultiheadAttention`中的mask是True表示遮盖
        out = self.attn(q, x + F.tanh(self.alpha) * pos_embeds, x, key_padding_mask=~attention_mask)[0]
        x = self.ln1(out)

        y = self.updim_proj(x)
        y = self.act(y)
        x = self.ln2(self.proj(y) + x)
        return x
