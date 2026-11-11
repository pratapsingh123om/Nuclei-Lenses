import numpy as np

from app.metastasis import (
    MetastasisService,
    MetastasisState,
    calibrate_probabilities,
    render_probability_outputs,
)


def test_temperature_one_preserves_probabilities():
    raw = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    calibrated = calibrate_probabilities(raw, 1.0)
    assert np.allclose(calibrated, raw, atol=1e-6)


def test_probability_rendering_returns_all_views():
    rgb = np.full((24, 32, 3), 128, dtype=np.uint8)
    probability = np.linspace(0, 1, 24 * 32, dtype=np.float32).reshape(24, 32)
    visuals = render_probability_outputs(rgb, probability, threshold=0.6)
    assert set(visuals) == {
        "original",
        "metastasis_overlay",
        "metastasis_heatmap",
        "evidence_map",
    }
    assert all(value.startswith("data:image/png;base64,") for value in visuals.values())


def test_analysis_returns_plain_language_diagnostic_support(tmp_path):
    class PositiveModel:
        @staticmethod
        def predict(batch, verbose=0):
            return np.full((len(batch), 1), 0.92, dtype=np.float32)

    service = MetastasisService(tmp_path)
    service.model = PositiveModel()
    service.state = MetastasisState(
        ready=True,
        status="ready",
        config={
            "input": {"shape": [96, 96, 3]},
            "output": {
                "temperature": 1.0,
                "thresholds": {"balanced_youden": 0.62, "sensitivity_focused": 0.30},
            },
        },
    )

    result = service.analyze(np.full((160, 160, 3), 128, dtype=np.uint8), "balanced")

    assert result["diagnostic_support"]["finding_code"] == "detected"
    assert result["diagnostic_support"]["finding_title"] == "Metastatic tissue pattern detected"
    assert "pathologist" in result["diagnostic_support"]["next_step"]

