import pytest

torch = pytest.importorskip("torch")
open_clip = pytest.importorskip("open_clip")

from fire_assessor.model import CLIP_NAME, FireModel


@pytest.fixture
def clip():
    return open_clip.create_model(CLIP_NAME, pretrained=None).eval()


def test_lora_starts_as_identity_and_wise_zero_is_zeroshot(clip):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = clip.encode_image(x)
        m = FireModel(clip).eval()
        _, z, bf = m(x)
        assert torch.allclose(torch.nn.functional.normalize(ref, dim=-1), z, atol=1e-4)
        assert bf.shape == (2, 768)
        for mod in m.lora:  # perturb adapters, then WiSE alpha=0 must restore the original
            torch.nn.init.normal_(mod.B_q, std=0.1); torch.nn.init.normal_(mod.B_v, std=0.1)
        _, z1, _ = m(x)
        assert not torch.allclose(z1, z, atol=1e-3)
        m.set_wise(0.0)
        _, z0, _ = m(x)
        assert torch.allclose(z0, z, atol=1e-4)


def test_only_lora_and_head_trainable_and_block8_untouched(clip):
    m = FireModel(clip)
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert names and all(("A_" in n or "B_" in n or n.startswith("head.")) for n in names)
    assert not any(f"resblocks.{i}." in n for n in names for i in range(8))


def test_backbone_uses_quickgelu_for_openai_weights(clip):
    from open_clip.transformer import QuickGELU
    assert any(isinstance(m, QuickGELU) for m in clip.visual.modules())
