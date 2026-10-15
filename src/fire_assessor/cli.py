"""`fire-assess IMAGE --artifact DIR` : run one image through the pipeline and print the JSON."""
from __future__ import annotations

import argparse
import sys

from .artifact import ModelArtifact
from .pipeline import Assessor


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fire-assess")
    ap.add_argument("image"); ap.add_argument("--artifact", required=True)
    ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float)
    a = ap.parse_args(argv)
    from .model import TorchScorer  # needs the [model] extra

    art = ModelArtifact.load(a.artifact)
    if art.weights_path is None:
        sys.exit("artifact has no weights")
    scorer = TorchScorer.from_artifact(art.weights_path)
    loc = (a.lat, a.lon) if a.lat is not None and a.lon is not None else None
    with open(a.image, "rb") as f:
        res = Assessor(scorer, art).assess_bytes(f.read(), a.image, reported_location=loc)
    print(res.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
