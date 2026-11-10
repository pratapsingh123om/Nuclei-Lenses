from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.draw import disk


SCRIPT = Path(__file__).parents[1] / "scripts" / "tune_instance_v2.py"
SPEC = importlib.util.spec_from_file_location("tune_instance_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_component_aware_watershed_keeps_isolated_nucleus_single():
    probability = np.zeros((96, 96), dtype=np.float32)
    rows, columns = disk((48, 48), 11, shape=probability.shape)
    probability[rows, columns] = 0.95

    prediction = MODULE.component_aware_watershed(
        probability,
        0.5,
        median_nucleus_area=380.0,
        min_component_size=20,
        closing_radius=0,
        distance_sigma=0.75,
        h_maxima=1.0,
        min_distance=8,
        split_area_factor=1.4,
    )

    assert int(prediction.max()) == 1


def test_component_aware_watershed_splits_two_touching_nuclei():
    probability = np.zeros((96, 96), dtype=np.float32)
    for center in ((48, 39), (48, 57)):
        rows, columns = disk(center, 12, shape=probability.shape)
        probability[rows, columns] = 0.95

    prediction = MODULE.component_aware_watershed(
        probability,
        0.5,
        median_nucleus_area=350.0,
        min_component_size=20,
        closing_radius=0,
        distance_sigma=0.75,
        h_maxima=1.0,
        min_distance=8,
        split_area_factor=1.4,
    )

    assert int(prediction.max()) == 2


def test_trial_selection_prefers_count_within_aji_tolerance():
    trials = pd.DataFrame(
        [
            {"mean_validation_aji": 0.70, "validation_count_mae": 50.0, "validation_count_bias": 50.0},
            {"mean_validation_aji": 0.69, "validation_count_mae": 20.0, "validation_count_bias": 10.0},
            {"mean_validation_aji": 0.66, "validation_count_mae": 5.0, "validation_count_bias": 1.0},
        ]
    )

    selected = MODULE.select_trial(trials, aji_tolerance=0.015)

    assert float(selected.validation_count_mae) == 20.0
