import numpy as np
from PIL import Image

from app.imaging import clean_binary_mask, decode_rgb, split_instances, split_instances_component_aware
from app.model import InferenceService, _starts


def test_sliding_window_covers_final_edge():
    assert _starts(1000, 256, 64)[0] == 0
    assert _starts(1000, 256, 64)[-1] == 744
    assert _starts(128, 256, 64) == [0]


def test_missing_artifacts_report_setup_required(tmp_path):
    state = InferenceService(tmp_path).load()
    assert state.ready is False
    assert state.status == "setup_required"
    assert "best_unet.weights.h5" in state.message


def test_decode_rgb_and_clean_mask(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (16, 12), (120, 50, 90)).save(path)
    rgb = decode_rgb(path.read_bytes(), 1_000)
    probability = np.zeros((12, 16), dtype=np.float32)
    probability[2:10, 3:13] = 0.9
    mask = clean_binary_mask(probability, threshold=0.5, min_size=4)
    assert rgb.shape == (12, 16, 3)
    assert mask.sum() == 80


def test_watershed_returns_integer_instances():
    yy, xx = np.ogrid[:64, :64]
    mask = ((yy - 32) ** 2 + (xx - 24) ** 2 < 12**2) | ((yy - 32) ** 2 + (xx - 40) ** 2 < 12**2)
    instances = split_instances(mask, min_distance=8, min_size=8)
    assert np.issubdtype(instances.dtype, np.integer)
    assert instances.max() >= 2


V2_CONFIG = {
    "median_nucleus_area": 350.0,
    "min_component_size": 20,
    "closing_radius": 0,
    "distance_sigma": 0.75,
    "h_value": 1.0,
    "min_distance": 8,
    "split_area_factor": 1.4,
    "count_cap_factor": 1.2,
}


def test_component_aware_pipeline_keeps_isolated_nucleus_single():
    yy, xx = np.ogrid[:96, :96]
    probability = np.zeros((96, 96), dtype=np.float32)
    probability[(yy - 48) ** 2 + (xx - 48) ** 2 < 11**2] = 0.95
    instances = split_instances_component_aware(probability, 0.5, V2_CONFIG)
    assert int(instances.max()) == 1


def test_component_aware_pipeline_splits_touching_nuclei():
    yy, xx = np.ogrid[:96, :96]
    probability = np.zeros((96, 96), dtype=np.float32)
    left = (yy - 48) ** 2 + (xx - 39) ** 2 < 12**2
    right = (yy - 48) ** 2 + (xx - 57) ** 2 < 12**2
    probability[left | right] = 0.95
    instances = split_instances_component_aware(probability, 0.5, V2_CONFIG)
    assert int(instances.max()) == 2

