"""LoRA on the attention in-projection (q and v) of an nn.MultiheadAttention, plus a WiSE-FT scale.

The low-rank delta is added to the packed `in_proj_weight` at call time, so the pretrained module
is untouched. `scale` multiplies the delta: 0 = zero-shot weights, 1 = fully fine-tuned, which
gives WiSE-FT weight-space interpolation without storing two models.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAAttention(nn.Module):
    def __init__(self, mha: nn.MultiheadAttention, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.mha = mha
        d = mha.embed_dim
        self.rank, self.alpha = rank, alpha
        self.A_q = nn.Parameter(torch.randn(rank, d) * 0.01)
        self.B_q = nn.Parameter(torch.zeros(d, rank))
        self.A_v = nn.Parameter(torch.randn(rank, d) * 0.01)
        self.B_v = nn.Parameter(torch.zeros(d, rank))
        self.scale = 1.0  # WiSE-FT mixing coefficient

    def delta(self) -> torch.Tensor:
        d = self.mha.embed_dim
        s = self.scale * self.alpha / self.rank
        z = torch.zeros(d, d, device=self.A_q.device, dtype=self.A_q.dtype)
        return torch.cat([s * self.B_q @ self.A_q, z, s * self.B_v @ self.A_v], dim=0)

    def forward(self, query, key, value, need_weights=False, attn_mask=None, **_):
        m = self.mha
        w = m.in_proj_weight + self.delta().to(m.in_proj_weight.dtype)
        x = query
        if m.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))
        out, weights = F.multi_head_attention_forward(
            query, key, value, m.embed_dim, m.num_heads, w, m.in_proj_bias, None, None, False, 0.0,
            m.out_proj.weight, m.out_proj.bias, training=self.training, need_weights=need_weights,
            attn_mask=attn_mask)
        if m.batch_first:
            out = out.transpose(0, 1)
        return out, weights


def apply_lora(visual: nn.Module, blocks=range(8, 12), rank: int = 8, alpha: float = 16.0):
    """Freeze the whole visual tower, then wrap attention in `blocks` (0-indexed 8..11 = blocks 9-12)."""
    for p in visual.parameters():
        p.requires_grad = False
    mods = []
    for i in blocks:
        blk = visual.transformer.resblocks[i]
        blk.attn = LoRAAttention(blk.attn, rank, alpha)
        mods.append(blk.attn)
    return mods


def set_wise_scale(visual: nn.Module, scale: float) -> None:
    for m in visual.modules():
        if isinstance(m, LoRAAttention):
            m.scale = scale
