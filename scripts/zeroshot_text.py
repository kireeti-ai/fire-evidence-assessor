"""Save CLIP text embeddings for zero-shot fire/smoke prompt ensembles (GPU/CPU; run once)."""
import numpy as np, torch, open_clip
from fire_assessor.model import CLIP_NAME
POS = {"flames": ["a photo of flames and fire burning", "a photo of a wildfire with visible flames", "a photo of a fire burning in a forest", "a photo of burning vegetation with flames"],
       "smoke": ["a photo of smoke rising from a wildfire", "a photo of thick smoke in the sky from a fire", "a photo of a smoke plume over a forest", "a photo of smoke from a distant fire"]}
NEG = {"flames": ["a photo of a landscape with no fire", "a photo of a sunset sky", "a photo of a forest with no flames", "a photo of a city street"],
       "smoke": ["a photo of a clear sky with no smoke", "a photo of a landscape with no smoke", "a photo of clouds in the sky", "a photo of a forest on a clear day"]}
clip, _, _ = open_clip.create_model_and_transforms(CLIP_NAME, pretrained="openai"); tok = open_clip.get_tokenizer(CLIP_NAME)
out = {}
with torch.no_grad():
    for k in ("flames", "smoke"):
        for nm, d in (("pos", POS), ("neg", NEG)):
            e = clip.encode_text(tok(d[k])); e = e / e.norm(dim=-1, keepdim=True); m = e.mean(0); out[f"{k}_{nm}"] = (m / m.norm()).numpy()
np.savez("text_emb.npz", **out); print({k: v.shape for k, v in out.items()})
