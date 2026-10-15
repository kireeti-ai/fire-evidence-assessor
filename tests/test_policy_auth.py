from datetime import datetime, timedelta, timezone

import numpy as np

from fire_assessor.authenticity import (AuthenticityHead, ReuseIndex, assess_authenticity,
                                        haversine_km, provenance_signals)
from fire_assessor.config import AuthenticityConfig, PolicyConfig
from fire_assessor.decision import Thresholds, decide
from fire_assessor.exif import ExifInfo
from fire_assessor.phash import hamming, phash
from fire_assessor.splits import assign_partitions, cluster_ids, verify_no_leakage
from conftest import textured

TH = Thresholds(0.5, 0.5)


def test_policy_table():
    d = lambda pf, ps, q=(), o=False, cfg=PolicyConfig(): decide(pf, ps, TH, list(q), o, cfg)
    assert d(0.9, 0.1).derived_label == "ACTIVE_FIRE" and d(0.9, 0.1).review_priority == "HIGH"
    assert d(0.1, 0.9).derived_label == "SMOKE_ONLY"
    assert d(0.1, 0.9, cfg=PolicyConfig(smoke_high_priority=False)).review_priority == "STANDARD"
    r = d(0.1, 0.1)
    assert (r.derived_label, r.review_priority) == ("NO_VISIBLE_FIRE", "STANDARD")
    assert d(0.1, 0.1, q=["TOO_DARK"]).derived_label == "UNCERTAIN"
    assert d(0.1, 0.1, o=True).derived_label == "UNCERTAIN"
    assert d(0.52, 0.1).derived_label == "UNCERTAIN"  # inside margin of threshold
    assert d(0.52, 0.1).review_priority == "HIGH"
    assert d(0.9, 0.1, q=["EXTREME_BLUR"]).derived_label == "UNCERTAIN"  # quality gate wins


def test_margin_boundary():
    assert decide(0.55, 0.0, TH, [], False).derived_label == "ACTIVE_FIRE"  # exactly at margin edge
    assert decide(0.5499, 0.0, TH, [], False).derived_label == "UNCERTAIN"


def test_phash_robust_to_jpeg_and_resize():
    im = textured(5, (256, 256))
    import io
    b = io.BytesIO(); im.save(b, "JPEG", quality=40); b.seek(0)
    from PIL import Image
    assert hamming(phash(im), phash(Image.open(b))) <= 6
    assert hamming(phash(im), phash(im.resize((128, 128)))) <= 6
    assert hamming(phash(im), phash(textured(99, (256, 256)))) > 10


def test_provenance_and_consistency():
    strong, weak = provenance_signals(b"xx <x:DigitalSourceType>trainedAlgorithmicMedia</x:DigitalSourceType>")
    assert "IPTC_DIGITAL_SOURCE_ALGORITHMIC" in strong
    assert provenance_signals(b"plain bytes") == ([], [])
    assert abs(haversine_km((0, 0), (0, 1)) - 111.19) < 0.5


def _auth(**kw):
    base = dict(raw=b"", exif=ExifInfo(), ph=0, image_id="a", embedding=None, block_feat=None,
                head=None, index=None, upload_time=None, reported_location=None)
    base.update(kw)
    return assess_authenticity(**base)


def test_authenticity_statuses():
    now = datetime.now(timezone.utc)
    assert _auth()[0] == "INCONCLUSIVE"  # no head, no signals: absence means nothing
    st, sig = _auth(exif=ExifInfo(captured_at=now + timedelta(days=2)), upload_time=now)
    assert st == "INCONSISTENT" and "EXIF_CAPTURE_AFTER_UPLOAD" in sig
    st, _ = _auth(exif=ExifInfo(gps=(0, 0)), reported_location=(10, 10))
    assert st == "INCONSISTENT"
    idx = ReuseIndex(); idx.add("old", 12345)
    assert _auth(ph=12345, index=idx)[0] == "REUSED"
    assert _auth(ph=12345, index=idx, image_id="old")[0] == "INCONCLUSIVE"  # no self-match
    assert _auth(raw=b"Midjourney")[0] == "POSSIBLE_SYNTHETIC"
    # editing-software tag alone is a weak signal, not a flag
    st, sig = _auth(raw=b"Adobe Photoshop")
    assert st == "INCONCLUSIVE" and "EDITING_SOFTWARE_TAG" in sig


def test_learned_head_and_no_concerns():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (200, 6)), rng.normal(3, 1, (200, 6))]
    Y = np.zeros((400, 2)); Y[200:, 0] = 1
    h = AuthenticityHead.fit(X, Y)
    assert _auth(head=h, block_feat=np.full(6, 3.0))[0] == "POSSIBLE_SYNTHETIC"
    assert _auth(head=h, block_feat=np.full(6, -1.0))[0] == "NO_CONCERNS_FOUND"
    h2 = AuthenticityHead.from_state(h.state())
    assert np.allclose(h2.probs(X[:3]), h.probs(X[:3]))


def test_reuse_index_roundtrip(tmp_path):
    i = ReuseIndex(); i.add("x", 2**63 + 5, np.ones(4))
    i.save(tmp_path / "i.json")
    j = ReuseIndex.load(tmp_path / "i.json")
    assert j.hashes == i.hashes and j.query("y", 2**63 + 5, np.ones(4), AuthenticityConfig()) == ["x"]


def test_split_by_cluster_no_leakage():
    base = [phash(textured(s, (128, 128))) for s in range(30)]
    hashes = base + [h ^ 0b11 for h in base]  # near-duplicate of each
    groups = [f"vid{i % 7}" for i in range(60)]
    cl = cluster_ids(hashes, groups)
    parts = assign_partitions(cl, seed=1)
    verify_no_leakage(cl, parts)
    assert set(parts) <= {"train", "val", "cal", "test"}
    bad = parts.copy(); bad[np.flatnonzero(cl == cl[0])[:1]] = "test" if parts[0] != "test" else "train"
    try:
        verify_no_leakage(cl, bad); assert False
    except AssertionError as e:
        assert "spans" in str(e)
