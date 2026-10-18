"""Render a model card (markdown) from an artifact + an optional results JSON. Numbers are copied, never invented."""
from __future__ import annotations

import json
from pathlib import Path

from .artifact import ModelArtifact

LIMITS = """## Limitations (read first)
- Advisory triage only. It does not confirm a wildfire, a location or a time. Every report still goes to a human.
- Training and evaluation data are public benchmarks (D-Fire, HPWREN FIgLib) and generated images, not citizen phone photos.
  Results on those do not establish deployment performance; only a TerraWatch-like test set and shadow mode can.
- The false-negative bound holds only if deployment data is exchangeable with the calibration data.
- Authenticity flags are weak signals. `NO_CONCERNS_FOUND` never means "verified real". Detectors degrade on unseen generators
  and after compression.
"""


def render(artifact_dir: str, results_json: str | None = None) -> str:
    a = ModelArtifact.load(artifact_dir, require_operational=False)
    th = "NOT PROVIDED" if a.thresholds is None else f"flames {a.thresholds.flames:.4f}, smoke {a.thresholds.smoke:.4f}"
    lines = [f"# Model card: {a.model_version}", "", f"**Artifact status: {a.status}**  (blockers: {len(a.blockers())})", "", LIMITS,
             "## Operating point", f"- Thresholds: {th}",
             f"- Temperatures: {json.dumps(a.temperatures)}", f"- Uncertainty margin: {a.uncertainty_margin}",
             f"- OOD detector: {json.dumps(a.ood_info)}", f"- Authenticity head present: {a.auth_head is not None}", "",
             "## Provenance", "```json", json.dumps(a.metadata, indent=2, default=str), "```"]
    if results_json and Path(results_json).exists():
        lines += ["", "## Measured results (copied from the results file)", "```json", Path(results_json).read_text(), "```"]
    return "\n".join(lines)
