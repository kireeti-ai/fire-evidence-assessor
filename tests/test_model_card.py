from fire_assessor.model_card import render


def test_card_contains_limits_and_thresholds(artifact, tmp_path):
    artifact.metadata = {"alpha": 0.05}
    artifact.save(tmp_path)
    t = render(str(tmp_path))
    assert "does not confirm a wildfire" in t and "0.5000" in t and '"alpha": 0.05' in t
