"""Compare frozen vision tensors in the checkpoint with the reference safetensors (HTTP range reads; sample, not full download)."""
import json, struct, subprocess, sys, zipfile, numpy as np
ckpt, inv_p, rev = sys.argv[1], sys.argv[2], sys.argv[3]
URL = f"https://huggingface.co/timm/vit_base_patch16_clip_224.openai/resolve/{rev}/open_clip_model.safetensors"
def rng(a, b): return subprocess.run(["curl", "-sSL", "--fail", "-r", f"{a}-{b}", URL], capture_output=True, check=True).stdout
n = struct.unpack("<Q", rng(0, 7))[0]; hdr = json.loads(rng(8, 8 + n - 1)); base = 8 + n
meta = hdr.pop("__metadata__", None); print("reference header: tensors", len(hdr), " visual:", sum(k.startswith("visual.") for k in hdr), " dtypes:", sorted({v["dtype"] for v in hdr.values()}), " metadata:", meta)
inv = json.load(open(inv_p)); z = zipfile.ZipFile(ckpt); root = z.namelist()[0].split("/")[0]
mine = {r["name"]: r for r in inv["tensors"]}
def ck(name): r = mine[name]; return np.frombuffer(z.read(f"{root}/data/{r['key']}"), np.float32).reshape(r["shape"])
def ref(name):
    h = hdr[name]; a, b = h["data_offsets"]; raw = rng(base + a, base + b - 1)
    dt = {"F32": np.float32, "F16": np.float16, "BF16": None}[h["dtype"]]; return np.frombuffer(raw, dt).reshape(h["shape"]).astype(np.float32)
pairs = [(k, k) for k in ("visual.class_embedding", "visual.positional_embedding", "visual.proj", "visual.conv1.weight", "visual.ln_pre.weight", "visual.ln_pre.bias", "visual.ln_post.weight", "visual.ln_post.bias")]
for b in (0, 5, 8, 11):   # 0,5 frozen+unwrapped ; 8,11 LoRA-wrapped (base weights live under attn.mha.*)
    for suf in ("attn.in_proj_weight", "attn.in_proj_bias", "attn.out_proj.weight", "attn.out_proj.bias", "ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias"):
        r = f"visual.transformer.resblocks.{b}.{suf}"; m = r.replace("attn.", "attn.mha.") if (b >= 8 and suf.startswith("attn.")) else r
        pairs.append((m, r))
eq = tot = 0; bad = []; nb = 0
for m, r in pairs:
    if m not in mine or r not in hdr: bad.append((m, "MISSING")); continue
    a, c = ck(m), ref(r); nb += a.nbytes; tot += 1
    same = a.shape == c.shape and np.array_equal(a, c)
    eq += same
    if not same: bad.append((m, f"max|diff|={np.abs(a-c).max():.3g}" if a.shape == c.shape else f"shape {a.shape} vs {c.shape}"))
print(f"compared {tot} tensors ({nb/1e6:.0f} MB): bit-identical {eq}/{tot}"); print("mismatches:", bad)
