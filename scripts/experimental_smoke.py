"""Real-checkpoint smoke test for the DEVELOPMENT-ONLY experimental path. EXPERIMENTAL / NOT FOR OPERATIONAL USE.

usage: python scripts/experimental_smoke.py <weights.pt> [--provenance docs/provenance/lora_s0_provenance.json] [--runs 5] [--json out.json]
Uses only synthetic images generated here (no user or dataset images). Prints raw, uncalibrated logits; they are not fire verdicts.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
PINNED_SHA256 = "7bd5498f04838fe19240da2b7339d794cc5fc3948074a0d826c33303d5c42dfa"  # audited checkpoint (docs/provenance/lora_s0_provenance.json)


def synthetic_jpegs() -> dict:
    rng = np.random.default_rng(0)
    out = {}
    flame = Image.new("RGB", (640, 480), (20, 10, 10))
    d = ImageDraw.Draw(flame)
    for i in range(40):
        x = 320 + int(rng.normal(0, 60)); h = int(rng.integers(60, 260))
        d.ellipse([x - 25, 480 - h, x + 25, 480], fill=(255, int(rng.integers(80, 200)), 10))
    out["synthetic_orange_flame_shapes"] = flame.filter(ImageFilter.GaussianBlur(3))
    haze = np.linspace(90, 190, 480)[:, None, None] * np.ones((480, 640, 3)) + rng.normal(0, 6, (480, 640, 3))
    out["synthetic_grey_haze_gradient"] = Image.fromarray(np.clip(haze, 0, 255).astype("uint8"))
    out["synthetic_uniform_noise"] = Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype="uint8"))
    blobs = {}
    for k, im in out.items():
        b = io.BytesIO(); im.save(b, "JPEG", quality=90); blobs[k] = b.getvalue()
    return blobs


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("--provenance", default=str(REPO / "docs/provenance/lora_s0_provenance.json"))
    ap.add_argument("--runs", type=int, default=5); ap.add_argument("--json")
    a = ap.parse_args()
    import torch, open_clip, timm  # noqa: E401
    from fire_assessor import experimental as ex

    torch.set_num_threads(max(1, torch.get_num_threads()))
    t = time.perf_counter(); ex.verify_experimental_checkpoint(a.weights, a.provenance, PINNED_SHA256); t_verify = time.perf_counter() - t
    rss0 = rss_mb(); t = time.perf_counter()
    asr = ex.load_experimental(a.weights, a.provenance, expected_sha256=PINNED_SHA256); t_load = time.perf_counter() - t
    results, lat = {}, []
    for name, blob in synthetic_jpegs().items():
        r = asr.assess_bytes(blob, name); results[name] = r
        for _ in range(a.runs):
            t = time.perf_counter(); asr.assess_bytes(blob, name); lat.append((time.perf_counter() - t) * 1000)
    again = asr.assess_bytes(synthetic_jpegs()["synthetic_orange_flame_shapes"], "x")["raw_logits"]
    report = {
        "notice": ex.EXPERIMENTAL_NOTICE, "checkpoint_sha256": PINNED_SHA256, "device": "cpu", "cuda_available": torch.cuda.is_available(),
        "verify_sha256_and_architecture_s": round(t_verify, 2), "model_load_s": round(t_load, 2),
        "inference_ms": {"n": len(lat), "median": round(float(np.median(lat)), 1), "p95": round(float(np.percentile(lat, 95)), 1), "min": round(min(lat), 1)},
        "peak_rss_mb": round(rss_mb(), 0), "rss_before_load_mb": round(rss0, 0), "torch_threads": torch.get_num_threads(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "open_clip_torch": open_clip.__version__, "timm": timm.__version__,
                     "numpy": np.__version__, "pillow": Image.__version__, "platform": platform.platform()},
        "deterministic_repeat_identical": again == results["synthetic_orange_flame_shapes"]["raw_logits"],
        "results": {k: {"assessment_status": v["assessment_status"], "raw_logits": v["raw_logits"], "artifact_status": v["artifact_status"]} for k, v in results.items()},
    }
    try:
        import transformers  # noqa: F401
        report["versions"]["transformers"] = transformers.__version__
    except Exception:
        report["versions"]["transformers"] = "not installed (not used by this model)"
    print(json.dumps(report, indent=2))
    if a.json: Path(a.json).write_text(json.dumps(report, indent=2))
    ok = all(v["assessment_status"] == "EXPERIMENTAL" and all(np.isfinite(list(v["raw_logits"].values()))) for v in results.values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
