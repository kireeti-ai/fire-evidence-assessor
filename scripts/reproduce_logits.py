"""Load the checkpoint (torch.load weights_only=True: tensors only) into the HEAD FireModel built with pretrained=None,
strict=True, run CPU inference on the first N test-partition images, compare with logits stored by the GPU export."""
import csv, sys, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
import open_clip
from PIL import Image
from fire_assessor.model import CLIP_NAME, build_model
S, N = sys.argv[1], int(sys.argv[2]); torch.set_num_threads(4)
model, pre = build_model(CLIP_NAME, pretrained=None)          # random init; weights come only from the checkpoint
ck = torch.load(sys.argv[5], map_location="cpu", weights_only=True)
res = model.load_state_dict(ck["model"], strict=True); print("strict load_state_dict:", res)
model.eval()
cfg = open_clip.get_pretrained_cfg(CLIP_NAME, "openai")
tf = pre.transforms; print("eval transform:", [type(t).__name__ for t in tf])
norm = [t for t in tf if type(t).__name__ == "Normalize"][0]
print("Normalize mean/std match openai cfg:", np.allclose(norm.mean, cfg["mean"]) and np.allclose(norm.std, cfg["std"]), "| resize:", [ (type(t).__name__, getattr(t,'size',None), getattr(t,'interpolation',None)) for t in tf[:2]])
rows = [r for r in csv.DictReader(open(f"{S}/manifests/dfire_manifest.csv")) if r["partition_xsrc"] == sys.argv[3]][:N]
stored = np.load(f"{S}/runs_lora_s0/{sys.argv[3]}.npz")
x = torch.stack([pre(Image.open(sys.argv[4] + "/" + r["path"]).convert("RGB")) for r in rows])
print("input tensor:", tuple(x.shape), x.dtype, "range", float(x.min()), float(x.max()))
with torch.no_grad(): lg, z, bf = model(x)
d = np.abs(lg.numpy() - stored["logits"][:N]); dz = np.abs(z.numpy() - stored["emb"][:N]); db = np.abs(bf.numpy() - stored["block"][:N])
print(f"CPU vs stored GPU export, first {N} test images: max|dlogit|={d.max():.2e}  max|demb|={dz.max():.2e}  max|dblock|={db.max():.2e}")
print("output shapes: logits", tuple(lg.shape), " emb", tuple(z.shape), "(L2 norm %.4f)" % float(z.norm(dim=-1).mean()), " block8 CLS", tuple(bf.shape))
print("sample logits (flames, smoke) vs labels:"); [print("  ", np.round(lg[i].numpy(), 3), "labels", [int(rows[i]["flames"]), int(rows[i]["smoke"])]) for i in range(min(6, N))]
