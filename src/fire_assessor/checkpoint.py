"""Torch-free checkpoint inspection and architecture-compatibility checking.

Reads a PyTorch zip checkpoint WITHOUT executing pickle code: the pickle is parsed with a restricted unpickler that only allows
tensor-rebuild stubs. The file is opened read-only and is never modified.
"""
from __future__ import annotations

import collections
import pickle
import zipfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class CheckpointFormatError(Exception):
    """Not a plain tensor checkpoint (blocked pickle global, bad layout, unsupported storage)."""


_STORAGE_DTYPES = {
    "FloatStorage": "float32", "HalfStorage": "float16", "BFloat16Storage": "bfloat16", "DoubleStorage": "float64",
    "LongStorage": "int64", "IntStorage": "int32", "ShortStorage": "int16", "CharStorage": "int8", "ByteStorage": "uint8", "BoolStorage": "bool",
}
_NP = {"float32": np.float32, "float16": np.float16, "float64": np.float64, "int64": np.int64, "int32": np.int32, "int16": np.int16,
       "int8": np.int8, "uint8": np.uint8, "bool": np.bool_}


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple
    dtype: str
    storage_key: str

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


@dataclass
class CheckpointInventory:
    tensors: dict = field(default_factory=dict)  # name -> TensorInfo
    top_level_keys: tuple = ()
    pickle_globals: tuple = ()
    root: str = ""

    @property
    def n_params(self) -> int:
        return sum(t.numel for t in self.tensors.values())


def _rebuild(storage, offset, size, stride, *rest):
    return {"storage": storage, "offset": offset, "size": tuple(size), "stride": tuple(stride)}


def _restricted_unpickler(fp, seen: set):
    class RU(pickle.Unpickler):
        def find_class(self, module, name):
            seen.add(f"{module}.{name}")
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return _rebuild
            if module == "torch" and name in _STORAGE_DTYPES:
                return name
            if (module, name) == ("collections", "OrderedDict"):
                return collections.OrderedDict
            raise CheckpointFormatError(f"blocked pickle global {module}.{name}")

        def persistent_load(self, pid):
            if not (isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage"):
                raise CheckpointFormatError("unsupported persistent id")
            return {"key": str(pid[2]), "dtype": pid[1]}

    return RU(fp)


def inspect_checkpoint(path, *, state_key: Optional[str] = "model") -> CheckpointInventory:
    """Inventory a checkpoint of the form {"model": state_dict} (or a bare state_dict when state_key is None)."""
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        raise CheckpointFormatError(f"not a zip checkpoint: {e}") from e
    pk = [n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
    if len(pk) != 1:
        raise CheckpointFormatError(f"expected exactly one data.pkl, found {len(pk)}")
    root = pk[0].rsplit("/", 1)[0] if "/" in pk[0] else ""
    seen: set = set()
    try:
        top = _restricted_unpickler(z.open(pk[0]), seen).load()
    except CheckpointFormatError:
        raise
    except Exception as e:
        raise CheckpointFormatError(f"unreadable pickle: {type(e).__name__}") from e
    if not isinstance(top, dict):
        raise CheckpointFormatError("top-level object is not a dict")
    sd = top
    if state_key is not None:
        if state_key not in top or not isinstance(top[state_key], dict):
            raise CheckpointFormatError(f"missing state dict key {state_key!r}; top-level keys {list(top)}")
        sd = top[state_key]
    inv = CheckpointInventory(top_level_keys=tuple(top), pickle_globals=tuple(sorted(seen)), root=root)
    for name, v in sd.items():
        if not (isinstance(v, dict) and "storage" in v):
            raise CheckpointFormatError(f"entry {name!r} is not a plain tensor")
        dt = _STORAGE_DTYPES.get(v["storage"]["dtype"])
        if dt is None:
            raise CheckpointFormatError(f"unsupported storage type for {name!r}")
        inv.tensors[name] = TensorInfo(tuple(v["size"]), dt, v["storage"]["key"])
    return inv


def read_tensor(path, inv: CheckpointInventory, name: str) -> np.ndarray:
    """Read one tensor's values (contiguous tensors only; the rebuild stub's stride is not applied)."""
    t = inv.tensors[name]
    with zipfile.ZipFile(path) as z:
        raw = z.read(f"{inv.root}/data/{t.storage_key}" if inv.root else f"data/{t.storage_key}")
    return np.frombuffer(raw, dtype=_NP[t.dtype]).reshape(t.shape)


# --- architecture specification -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ArchitectureSpec:
    """Vision tower + LoRA + fire head, as built by `fire_assessor.model.FireModel`. Defaults = CLIP ViT-B/16, rank-8 LoRA on
    zero-indexed blocks 8-11 (q and v), two-logit head."""

    width: int = 768
    layers: int = 12
    patch: int = 16
    image_size: int = 224
    embed_dim: int = 512
    mlp_ratio: int = 4
    lora_blocks: tuple = (8, 9, 10, 11)
    lora_rank: int = 8
    n_outputs: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "ArchitectureSpec":
        lora = d.get("lora", {})
        return cls(width=int(d["vision_width"]), layers=int(d["layers"]), patch=int(d["patch_size"]), image_size=int(d["image_size"]),
                   embed_dim=int(d["embed_dim"]), lora_blocks=tuple(int(b) for b in lora.get("blocks_zero_indexed", ())),
                   lora_rank=int(lora.get("rank", 0)), n_outputs=int(d["head"]["outputs"]))


def expected_tensors(spec: ArchitectureSpec) -> dict:
    W, E, P = spec.width, spec.embed_dim, spec.patch
    grid = spec.image_size // P
    e = {"visual.class_embedding": (W,), "visual.positional_embedding": (grid * grid + 1, W), "visual.proj": (W, E),
         "visual.conv1.weight": (W, 3, P, P), "visual.ln_pre.weight": (W,), "visual.ln_pre.bias": (W,),
         "visual.ln_post.weight": (W,), "visual.ln_post.bias": (W,)}
    M = spec.mlp_ratio * W
    for i in range(spec.layers):
        p = f"visual.transformer.resblocks.{i}."
        for ln in ("ln_1", "ln_2"):
            e[p + ln + ".weight"] = (W,)
            e[p + ln + ".bias"] = (W,)
        a = p + ("attn.mha." if i in spec.lora_blocks else "attn.")
        e[a + "in_proj_weight"] = (3 * W, W)
        e[a + "in_proj_bias"] = (3 * W,)
        e[a + "out_proj.weight"] = (W, W)
        e[a + "out_proj.bias"] = (W,)
        if i in spec.lora_blocks:
            r = spec.lora_rank
            e[p + "attn.A_q"], e[p + "attn.B_q"], e[p + "attn.A_v"], e[p + "attn.B_v"] = (r, W), (W, r), (r, W), (W, r)
        e[p + "mlp.c_fc.weight"], e[p + "mlp.c_fc.bias"] = (M, W), (M,)
        e[p + "mlp.c_proj.weight"], e[p + "mlp.c_proj.bias"] = (W, M), (W,)
    e["head.weight"], e["head.bias"] = (spec.n_outputs, E), (spec.n_outputs,)
    e["head0_w"], e["head0_b"] = (spec.n_outputs, E), (spec.n_outputs,)
    return e


def check_compatibility(inv: CheckpointInventory, spec: ArchitectureSpec) -> list:
    """Return a list of human-readable mismatches (empty list = compatible). Exact name/shape/dtype set equality."""
    exp, got, problems = expected_tensors(spec), inv.tensors, []
    for n in sorted(set(exp) - set(got)):
        problems.append(f"missing tensor {n} {exp[n]}")
    for n in sorted(set(got) - set(exp)):
        problems.append(f"unexpected tensor {n} {got[n].shape}")
    for n in sorted(set(exp) & set(got)):
        if tuple(got[n].shape) != tuple(exp[n]):
            problems.append(f"shape mismatch {n}: checkpoint {tuple(got[n].shape)} != expected {tuple(exp[n])}")
        if got[n].dtype != "float32":
            problems.append(f"dtype mismatch {n}: {got[n].dtype} != float32")
    return problems
