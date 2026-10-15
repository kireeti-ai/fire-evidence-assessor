"""CLIP ViT-B/16 + LoRA fire model and the feature extractor behind the shared embedding.

One forward pass yields: block-8 CLS token (frozen features for the authenticity head), the final
embedding z (fire head, OOD, reuse), and two fire logits (flames, smoke).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .lora import apply_lora, set_wise_scale
from .scorer import ScorerOutput

FIRE_PROMPTS = {
    "flames": "a photo of flames and fire burning",
    "smoke": "a photo of smoke rising from a wildfire",
}
CLIP_NAME = "ViT-B-16-quickgelu"  # OpenAI weights need QuickGELU; plain "ViT-B-16" silently degrades them
AUTH_TAP_BLOCK = 8  # CLS token taken after block 8 (1-indexed); blocks 1-8 are never touched by LoRA


class FireModel(nn.Module):
    def __init__(self, clip_model: nn.Module, lora_rank: int = 8, lora_blocks=range(8, 12),
                 text_init: torch.Tensor | None = None):
        super().__init__()
        self.visual = clip_model.visual
        self.lora = apply_lora(self.visual, lora_blocks, lora_rank)
        d = self.visual.proj.shape[1]
        self.head = nn.Linear(d, 2)
        if text_init is not None:  # zero-shot initialisation from (normalised) prompt embeddings
            with torch.no_grad():
                self.head.weight.copy_(F.normalize(text_init, dim=-1) * 10.0)
                self.head.bias.zero_()
        self.register_buffer("head0_w", self.head.weight.detach().clone())  # for WiSE-FT interpolation
        self.register_buffer("head0_b", self.head.bias.detach().clone())
        self.wise = 1.0

    def set_wise(self, alpha: float) -> None:
        self.wise = alpha
        set_wise_scale(self.visual, alpha)

    def trainable_groups(self, lr_lora=1e-4, lr_head=1e-3, wd=0.01):
        lora = [p for m in self.lora for p in m.parameters() if p.requires_grad]
        return [{"params": lora, "lr": lr_lora, "weight_decay": wd},
                {"params": list(self.head.parameters()), "lr": lr_head, "weight_decay": wd}]

    def features(self, x: torch.Tensor):
        v = self.visual
        x = v.conv1(x).flatten(2).transpose(1, 2)
        cls = v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        block_feat = None
        for i, blk in enumerate(v.transformer.resblocks, start=1):
            x = blk(x)
            if i == AUTH_TAP_BLOCK:
                block_feat = x[:, 0]
        z = v.ln_post(x[:, 0]) @ v.proj
        return block_feat, z

    def forward(self, x: torch.Tensor):
        block_feat, z = self.features(x)
        z = F.normalize(z, dim=-1)
        w = (1 - self.wise) * self.head0_w + self.wise * self.head.weight
        b = (1 - self.wise) * self.head0_b + self.wise * self.head.bias
        return F.linear(z, w, b), z, block_feat


def build_model(name=CLIP_NAME, pretrained="openai", lora_rank=8):
    """Load OpenCLIP. Check the licence of the chosen checkpoint before use (design section 7)."""
    import open_clip

    clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    tok = open_clip.get_tokenizer(name)
    with torch.no_grad():
        t = clip.encode_text(tok([FIRE_PROMPTS["flames"], FIRE_PROMPTS["smoke"]]))
    return FireModel(clip, lora_rank, text_init=t), preprocess


class TorchScorer:
    """Inference-time Scorer backed by a trained FireModel checkpoint (CPU by default)."""

    def __init__(self, model: FireModel, preprocess, device="cpu"):
        self.model, self.pre, self.device = model.eval().to(device), preprocess, device

    @classmethod
    def from_artifact(cls, weights_path: str, name=CLIP_NAME, device="cpu", wise_alpha: float = 1.0):
        model, pre = build_model(name, pretrained=None)
        ck = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        model.set_wise(wise_alpha)
        return cls(model, pre, device)

    @torch.no_grad()
    def score(self, image: Image.Image) -> ScorerOutput:
        x = self.pre(image).unsqueeze(0).to(self.device)
        logits, z, bf = self.model(x)
        return ScorerOutput(float(logits[0, 0]), float(logits[0, 1]), z[0].cpu().numpy(),
                            bf[0].cpu().numpy())
