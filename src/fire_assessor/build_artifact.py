"""Build the versioned artifact from exported partition features.

Fits, on the calibration partition only: per-head temperature, risk-controlled thresholds
(FNR <= alpha with probability >= 1 - delta), the OOD cutoff and the authenticity head.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .artifact import ModelArtifact
from .authenticity import AuthenticityHead
from .calibration import calibration_report, fit_temperature, sigmoid
from .config import MODEL_VERSION
from .decision import Thresholds
from .ood import KNNOOD, fit_ood
from .risk_control import select_threshold


def build(train, cal, alpha, delta, margin, weights=None, out="artifacts/fire-assessor-v1.0.0",
          ood_out=None, auth=None, extra_meta=None):
    T, th, report = {}, {}, {}
    for k, name in enumerate(("flames", "smoke")):
        T[name] = fit_temperature(cal["logits"][:, k], cal["y"][:, k])
        report[name] = calibration_report(cal["logits"][:, k], cal["y"][:, k], T[name])
        rc = select_threshold(sigmoid(cal["logits"][:, k] / T[name]), cal["y"][:, k], alpha, delta)
        th[name] = rc.threshold
        report[name]["risk_control"] = rc.__dict__
    ood = KNNOOD()
    info = fit_ood(ood, train["emb"], cal["emb"], None if ood_out is None else ood_out["emb"])
    head = None
    if auth is not None:
        head = AuthenticityHead.fit(auth["X"], auth["Y"])
    art = ModelArtifact(MODEL_VERSION, Thresholds(th["flames"], th["smoke"]), T, margin, ood, info, head,
                        {"alpha": alpha, "delta": delta, "calibration": report, **(extra_meta or {})})
    art.save(out, weights_src=weights)
    return art, report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--cal", required=True)
    ap.add_argument("--ood-out", help="npz of unrelated images (emb) to validate the OOD detector")
    ap.add_argument("--weights"); ap.add_argument("--out", default="artifacts/fire-assessor-v1.0.0")
    ap.add_argument("--alpha", type=float, required=True, help="target FNR (stakeholder decision)")
    ap.add_argument("--delta", type=float, default=0.1); ap.add_argument("--margin", type=float, default=0.05)
    a = ap.parse_args(argv)
    ld = lambda p: dict(np.load(p, allow_pickle=False))
    _, rep = build(ld(a.train), ld(a.cal), a.alpha, a.delta, a.margin, a.weights, a.out,
                   ld(a.ood_out) if a.ood_out else None)
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
