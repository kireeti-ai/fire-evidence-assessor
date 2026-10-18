import os
import pathlib

import pytest

from fire_assessor.checkpoint import (ArchitectureSpec, CheckpointFormatError, check_compatibility, expected_tensors, inspect_checkpoint, read_tensor)
from fire_assessor.config import OUTPUT_NAMES
from conftest import TINY_ARCH, write_tiny_checkpoint

torch = pytest.importorskip("torch")
SPEC = ArchitectureSpec.from_dict(TINY_ARCH)


def test_expected_tensor_count_matches_real_vit_b16_layout():
    # 8 stem/ln tensors + 12 blocks x 12 + 4 LoRA matrices x 4 blocks + 4 head tensors = 172 (the audited checkpoint)
    assert len(expected_tensors(ArchitectureSpec())) == 172


def test_tiny_checkpoint_is_compatible(tmp_path):
    inv = inspect_checkpoint(write_tiny_checkpoint(tmp_path / "w.pt"))
    assert inv.top_level_keys == ("model",) and check_compatibility(inv, SPEC) == []
    assert set(inv.pickle_globals) <= {"collections.OrderedDict", "torch.FloatStorage", "torch._utils._rebuild_tensor_v2"}


@pytest.mark.parametrize("mutate,needle", [
    (lambda sd: sd.pop("head.weight"), "missing tensor head.weight"),
    (lambda sd: sd.__setitem__("transformer.text.weight", torch.zeros(3)), "unexpected tensor transformer.text.weight"),
    (lambda sd: sd.__setitem__("head.weight", torch.zeros(3, 6)), "shape mismatch head.weight"),
    (lambda sd: sd.__setitem__("visual.proj", torch.zeros(8, 6).half()), "dtype mismatch visual.proj"),
    (lambda sd: sd.pop("visual.transformer.resblocks.1.attn.A_q"), "missing tensor visual.transformer.resblocks.1.attn.A_q"),
])
def test_architecture_mismatch_is_reported(tmp_path, mutate, needle):
    inv = inspect_checkpoint(write_tiny_checkpoint(tmp_path / "w.pt", mutate=mutate))
    assert any(needle in p for p in check_compatibility(inv, SPEC))


def test_wrong_lora_rank_or_block_placement_is_detected(tmp_path):
    inv = inspect_checkpoint(write_tiny_checkpoint(tmp_path / "w.pt"))
    assert check_compatibility(inv, ArchitectureSpec.from_dict({**TINY_ARCH, "lora": {**TINY_ARCH["lora"], "rank": 4}}))
    assert check_compatibility(inv, ArchitectureSpec.from_dict({**TINY_ARCH, "lora": {**TINY_ARCH["lora"], "blocks_zero_indexed": [0]}}))
    assert check_compatibility(inv, ArchitectureSpec.from_dict({**TINY_ARCH, "vision_width": 16}))


def test_malicious_pickle_is_blocked_and_never_executed(tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    torch.save({"model": {"x": Evil()}}, tmp_path / "evil.pt")
    with pytest.raises(CheckpointFormatError, match="blocked pickle global"):
        inspect_checkpoint(tmp_path / "evil.pt")
    assert not marker.exists()


def test_non_checkpoint_inputs_rejected(tmp_path):
    (tmp_path / "junk.pt").write_bytes(b"not a zip")
    with pytest.raises(CheckpointFormatError):
        inspect_checkpoint(tmp_path / "junk.pt")
    torch.save({"state": {}}, tmp_path / "nokey.pt")
    with pytest.raises(CheckpointFormatError, match="missing state dict key"):
        inspect_checkpoint(tmp_path / "nokey.pt")


def test_read_tensor_returns_values(tmp_path):
    def mut(sd): sd["head.bias"] = torch.tensor([1.5, -2.0])
    p = write_tiny_checkpoint(tmp_path / "w.pt", mutate=mut)
    assert read_tensor(p, inspect_checkpoint(p), "head.bias").tolist() == [1.5, -2.0]


def test_output_label_mapping_logit0_is_flames_logit1_is_smoke():
    from PIL import Image
    from fire_assessor.model import TorchScorer

    class Stub(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0, -3.0]]), torch.ones(1, 512) / 512 ** 0.5, torch.zeros(1, 768)

    sc = TorchScorer(Stub(), lambda im: torch.zeros(3, 4, 4))
    out = sc.score(Image.new("RGB", (8, 8)))
    assert (out.flames_logit, out.smoke_logit) == (2.0, -3.0) and OUTPUT_NAMES == ("flames_visible", "smoke_visible")


REAL = os.environ.get("FIRE_ASSESSOR_CHECKPOINT")


@pytest.mark.skipif(not REAL or not pathlib.Path(REAL or "").exists(), reason="set FIRE_ASSESSOR_CHECKPOINT to the audited lora_s0 weights.pt to run")
def test_audited_checkpoint_matches_recorded_provenance():
    import json
    from fire_assessor.hashing import sha256_file
    prov = json.loads((pathlib.Path(__file__).parents[1] / "docs/provenance/lora_s0_provenance.json").read_text())
    assert sha256_file(REAL) == prov["checkpoint"]["sha256"]
    inv = inspect_checkpoint(REAL)
    assert len(inv.tensors) == 172 and inv.n_params == 86292996 and check_compatibility(inv, ArchitectureSpec()) == []
