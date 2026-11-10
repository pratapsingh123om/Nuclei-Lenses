#!/usr/bin/env python3
"""Tune a conservative nuclei instance pipeline without retraining the U-Net.

This script is designed to run in the same Google Colab runtime as the main
MoNuSeg notebook, but it is also restart-safe: it rebuilds the trained U-Net,
loads the saved checkpoint, and caches reconstructed full-image probabilities.

Protocol safeguards
-------------------
* The semantic threshold remains the value frozen by the original notebook.
* Instance settings are selected on the six validation images only.
* The fourteen benchmark images are evaluated only after v2 is frozen.
* All outputs are written under ``artifacts/instance_v2``. V1 files are never
  overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import feature, measure, morphology, segmentation


DEFAULT_PROJECT_ROOT = "/content/drive/MyDrive/Nuclei-lens/monuseg_project"


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_binary(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path)) > 0


def load_instances(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.int32)


def save_uint16(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint16), mode="I;16").save(path)


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)
    denominator = int(y_true.sum() + y_pred.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(y_true, y_pred).sum() / denominator)


def iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)
    union = int(np.logical_or(y_true, y_pred).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(y_true, y_pred).sum() / union)


def aggregated_jaccard_index(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute one-to-one Aggregated Jaccard Index (AJI)."""

    ground_truth, _, _ = segmentation.relabel_sequential(ground_truth.astype(np.int32))
    prediction, _, _ = segmentation.relabel_sequential(prediction.astype(np.int32))
    gt_count, pred_count = int(ground_truth.max()), int(prediction.max())
    if gt_count == 0 and pred_count == 0:
        return 1.0

    intersections = np.bincount(
        (ground_truth.ravel() * (pred_count + 1) + prediction.ravel()).astype(np.int64),
        minlength=(gt_count + 1) * (pred_count + 1),
    ).reshape(gt_count + 1, pred_count + 1)[1:, 1:]
    gt_areas = np.bincount(ground_truth.ravel(), minlength=gt_count + 1)[1:]
    pred_areas = np.bincount(prediction.ravel(), minlength=pred_count + 1)[1:]

    if gt_count and pred_count:
        unions = gt_areas[:, None] + pred_areas[None, :] - intersections
        ious = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=np.float64),
            where=unions > 0,
        )
        pairs = np.argwhere(intersections > 0)
        pairs = sorted(pairs, key=lambda pair: ious[tuple(pair)], reverse=True)
    else:
        unions = np.zeros((gt_count, pred_count), dtype=np.int64)
        pairs = []

    matched_gt: set[int] = set()
    matched_prediction: set[int] = set()
    intersection_sum = 0
    union_sum = 0
    for gt_index, prediction_index in pairs:
        gt_index, prediction_index = int(gt_index), int(prediction_index)
        if gt_index in matched_gt or prediction_index in matched_prediction:
            continue
        matched_gt.add(gt_index)
        matched_prediction.add(prediction_index)
        intersection_sum += int(intersections[gt_index, prediction_index])
        union_sum += int(unions[gt_index, prediction_index])

    union_sum += int(sum(gt_areas[i] for i in range(gt_count) if i not in matched_gt))
    union_sum += int(sum(pred_areas[i] for i in range(pred_count) if i not in matched_prediction))
    return float(intersection_sum / union_sum) if union_sum else 1.0


def semantic_mask(probability: np.ndarray, threshold: float, min_size: int) -> np.ndarray:
    """Reproduce the v1 semantic cleanup for unchanged Dice/IoU reporting."""

    mask = morphology.remove_small_objects(probability >= threshold, min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=min_size)
    return morphology.binary_opening(mask, morphology.disk(1))


def _merge_small_watershed_regions(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Merge tiny watershed fragments into the neighbor with most contact."""

    labels = labels.astype(np.int32, copy=True)
    counts = np.bincount(labels.ravel())
    small_ids = np.flatnonzero((counts < min_area) & (np.arange(len(counts)) > 0))
    for instance_id in small_ids[np.argsort(counts[small_ids])]:
        region = labels == int(instance_id)
        if not region.any():
            continue
        border = morphology.binary_dilation(region, morphology.disk(1)) & ~region
        neighbors = labels[border]
        neighbors = neighbors[(neighbors > 0) & (neighbors != instance_id)]
        if len(neighbors):
            neighbor_counts = np.bincount(neighbors)
            labels[region] = int(np.argmax(neighbor_counts))
        else:
            labels[region] = 0
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels.astype(np.int32)


def _nms_peaks(
    coordinates: Iterable[tuple[int, int]],
    distance: np.ndarray,
    min_distance: int,
    limit: int,
) -> list[tuple[int, int]]:
    ordered = sorted(coordinates, key=lambda point: float(distance[point]), reverse=True)
    selected: list[tuple[int, int]] = []
    minimum_squared = float(min_distance * min_distance)
    for point in ordered:
        if all((point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 >= minimum_squared for other in selected):
            selected.append(point)
            if len(selected) >= limit:
                break
    return selected


def component_aware_watershed(
    probability: np.ndarray,
    threshold: float,
    median_nucleus_area: float,
    *,
    min_component_size: int,
    closing_radius: int,
    distance_sigma: float,
    h_maxima: float,
    min_distance: int,
    split_area_factor: float,
    count_cap_factor: float = 1.20,
) -> np.ndarray:
    """Separate touching nuclei while resisting single-nucleus oversplitting.

    Ordinary watershed creates a seed at every distance-map maximum. Here,
    normal-sized connected components receive exactly one seed. Only components
    larger than a training-derived nucleus-area threshold may receive multiple
    seeds. Weak peaks are removed by h-maxima, nearby peaks are suppressed, and
    tiny watershed fragments are merged instead of discarded.
    """

    mask = probability >= threshold
    mask = morphology.remove_small_objects(mask, min_size=min_component_size)
    mask = morphology.remove_small_holes(mask, area_threshold=min_component_size)
    if closing_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(closing_radius))
    mask = morphology.remove_small_objects(mask, min_size=min_component_size)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    components = measure.label(mask, connectivity=1)
    raw_distance = ndi.distance_transform_edt(mask)
    distance = (
        ndi.gaussian_filter(raw_distance, sigma=distance_sigma)
        if distance_sigma > 0
        else raw_distance
    )
    prominent = morphology.h_maxima(distance, h=h_maxima) & mask
    peak_regions = measure.label(prominent, connectivity=2)

    candidates_by_component: dict[int, list[tuple[int, int]]] = {}
    peak_ids = np.arange(1, int(peak_regions.max()) + 1)
    peak_positions = ndi.maximum_position(distance, labels=peak_regions, index=peak_ids)
    for point_value in peak_positions:
        point = (int(point_value[0]), int(point_value[1]))
        component_id = int(components[point])
        if component_id > 0:
            candidates_by_component.setdefault(component_id, []).append(point)

    markers = np.zeros(mask.shape, dtype=np.int32)
    marker_id = 0
    split_area = float(split_area_factor * median_nucleus_area)
    component_areas = np.bincount(components.ravel())
    for component_id in range(1, len(component_areas)):
        local_mask = components == component_id
        local_values = np.where(local_mask, distance, -np.inf)
        fallback_flat = int(np.argmax(local_values))
        fallback = tuple(int(value) for value in np.unravel_index(fallback_flat, mask.shape))

        component_area = int(component_areas[component_id])
        if component_area <= split_area:
            selected = [fallback]
        else:
            candidates = candidates_by_component.get(component_id, [])
            if fallback not in candidates:
                candidates.append(fallback)
            estimated_count = max(2, int(math.ceil(component_area / median_nucleus_area)))
            cap = max(2, int(math.ceil(estimated_count * count_cap_factor)))
            selected = _nms_peaks(candidates, distance, min_distance, cap)
            if not selected:
                selected = [fallback]

        for point in selected:
            marker_id += 1
            markers[point] = marker_id

    instances = segmentation.watershed(-distance, markers, mask=mask)
    return _merge_small_watershed_regions(instances, min_component_size)


def v1_watershed(mask: np.ndarray, min_distance: int, min_size: int) -> np.ndarray:
    cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
    if not cleaned.any():
        return np.zeros(cleaned.shape, dtype=np.int32)
    distance = ndi.distance_transform_edt(cleaned)
    coordinates = feature.peak_local_max(
        distance,
        labels=cleaned,
        min_distance=min_distance,
        exclude_border=False,
    )
    markers = np.zeros(cleaned.shape, dtype=np.int32)
    if len(coordinates):
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    else:
        markers = measure.label(cleaned, connectivity=1)
    instances = segmentation.watershed(-distance, markers, mask=cleaned)
    instances = morphology.remove_small_objects(instances, min_size=min_size)
    instances, _, _ = segmentation.relabel_sequential(instances)
    return instances.astype(np.int32)


def window_starts(length: int, patch_size: int, overlap: int) -> list[int]:
    if length <= patch_size:
        return [0]
    stride = max(1, patch_size - overlap)
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def build_unet(patch_size: int, base_filters: int):
    import tensorflow as tf

    def convolution_block(x, filters_count: int):
        x = tf.keras.layers.Conv2D(filters_count, 3, padding="same", activation="relu")(x)
        return tf.keras.layers.Conv2D(filters_count, 3, padding="same", activation="relu")(x)

    inputs = tf.keras.Input((patch_size, patch_size, 3), name="image")
    c1 = convolution_block(inputs, base_filters)
    c2 = convolution_block(tf.keras.layers.MaxPooling2D()(c1), base_filters * 2)
    c3 = convolution_block(tf.keras.layers.MaxPooling2D()(c2), base_filters * 4)
    c4 = convolution_block(tf.keras.layers.MaxPooling2D()(c3), base_filters * 8)
    bridge = convolution_block(tf.keras.layers.MaxPooling2D()(c4), base_filters * 16)
    x = bridge
    for skip, filters_count in (
        (c4, base_filters * 8),
        (c3, base_filters * 4),
        (c2, base_filters * 2),
        (c1, base_filters),
    ):
        x = tf.keras.layers.Conv2DTranspose(filters_count, 2, strides=2, padding="same")(x)
        x = convolution_block(tf.keras.layers.Concatenate()([x, skip]), filters_count)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="nuclei_probability")(x)
    return tf.keras.Model(inputs, outputs, name="nuclei_lens_unet")


def overlapping_predict(
    model,
    rgb: np.ndarray,
    *,
    patch_size: int,
    overlap: int,
    batch_size: int,
) -> np.ndarray:
    height, width = rgb.shape[:2]
    padded_height, padded_width = max(height, patch_size), max(width, patch_size)
    padded = np.pad(
        rgb,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="reflect" if height > 1 and width > 1 else "edge",
    )
    coordinates = [
        (y, x)
        for y in window_starts(padded_height, patch_size, overlap)
        for x in window_starts(padded_width, patch_size, overlap)
    ]
    probability_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    visit_count = np.zeros((padded_height, padded_width), dtype=np.float32)
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        batch = np.stack(
            [padded[y : y + patch_size, x : x + patch_size] for y, x in batch_coordinates]
        ).astype(np.float32) / 255.0
        predictions = model.predict(batch, verbose=0)[..., 0]
        for (y, x), prediction in zip(batch_coordinates, predictions):
            probability_sum[y : y + patch_size, x : x + patch_size] += prediction
            visit_count[y : y + patch_size, x : x + patch_size] += 1
    return (probability_sum / np.maximum(visit_count, 1))[:height, :width]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probability_cache(
    rows: pd.DataFrame,
    checkpoint: Path,
    output_root: Path,
    frozen: dict[str, Any],
    *,
    force: bool,
) -> dict[str, np.ndarray]:
    import tensorflow as tf

    checkpoint_hash = file_sha256(checkpoint)
    cache_root = output_root / "probabilities" / checkpoint_hash[:12]
    cache_root.mkdir(parents=True, exist_ok=True)
    required = {str(row.image_id): cache_root / f"{row.image_id}.npy" for _, row in rows.iterrows()}
    missing = list(required) if force else [image_id for image_id, path in required.items() if not path.exists()]

    if missing:
        print(f"Reconstructing {len(missing)} full-image probability map(s)...")
        tf.keras.backend.clear_session()
        model = build_unet(int(frozen["patch_size"]), int(frozen["base_filters"]))
        model.load_weights(checkpoint)
        selected_rows = rows.loc[rows.image_id.astype(str).isin(missing)]
        for number, (_, row) in enumerate(selected_rows.iterrows(), start=1):
            probability = overlapping_predict(
                model,
                load_rgb(row.image_path),
                patch_size=int(frozen["patch_size"]),
                overlap=int(frozen["overlap"]),
                batch_size=int(frozen.get("inference_batch_size", 8)),
            )
            np.save(required[str(row.image_id)], probability.astype(np.float32))
            print(f"  probability {number}/{len(selected_rows)}: {row.image_id}")
        del model
        tf.keras.backend.clear_session()
    else:
        print(f"Using {len(required)} cached full-image probability map(s).")

    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "patch_size": int(frozen["patch_size"]),
        "overlap": int(frozen["overlap"]),
        "images": len(required),
    }
    (cache_root / "cache_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {image_id: np.load(path, mmap_mode="r") for image_id, path in required.items()}


def training_median_nucleus_area(training_rows: pd.DataFrame) -> float:
    areas: list[np.ndarray] = []
    for _, row in training_rows.iterrows():
        labels = load_instances(row.instance_map_path)
        counts = np.bincount(labels.ravel())[1:]
        areas.append(counts[counts > 0])
    return float(np.median(np.concatenate(areas)))


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    frame = pd.DataFrame(rows)
    errors = frame.predicted_count - frame.ground_truth_count
    return {
        "images": int(len(frame)),
        "mean_aji": float(frame.aji.mean()),
        "count_mae": float(errors.abs().mean()),
        "count_median_absolute_error": float(errors.abs().median()),
        "count_relative_mae_percent": float((errors.abs() / frame.ground_truth_count).mean() * 100.0),
        "count_bias": float(errors.mean()),
        "mean_predicted_count": float(frame.predicted_count.mean()),
        "mean_ground_truth_count": float(frame.ground_truth_count.mean()),
    }


def evaluate_parameters(
    rows: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    threshold: float,
    median_area: float,
    parameters: dict[str, Any],
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        image_id = str(row.image_id)
        prediction = component_aware_watershed(
            np.asarray(probabilities[image_id]),
            threshold,
            median_area,
            **parameters,
        )
        truth = ground_truth[image_id]
        gt_count, pred_count = int(truth.max()), int(prediction.max())
        results.append(
            {
                "image_id": image_id,
                "aji": aggregated_jaccard_index(truth, prediction),
                "ground_truth_count": gt_count,
                "predicted_count": pred_count,
                "count_error": pred_count - gt_count,
                "count_absolute_error": abs(pred_count - gt_count),
            }
        )
    return metric_summary(results), results


def evaluate_v1(
    rows: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    frozen: dict[str, Any],
) -> dict[str, float | int]:
    results: list[dict[str, Any]] = []
    threshold = float(frozen["mask_threshold"])
    min_size = int(frozen["min_object_size"])
    for _, row in rows.iterrows():
        image_id = str(row.image_id)
        mask = semantic_mask(np.asarray(probabilities[image_id]), threshold, min_size)
        prediction = v1_watershed(mask, int(frozen["watershed_min_distance"]), min_size)
        truth = ground_truth[image_id]
        gt_count, pred_count = int(truth.max()), int(prediction.max())
        results.append(
            {
                "image_id": image_id,
                "aji": aggregated_jaccard_index(truth, prediction),
                "ground_truth_count": gt_count,
                "predicted_count": pred_count,
            }
        )
    return metric_summary(results)


def search_grid(profile: str) -> list[dict[str, Any]]:
    if profile == "quick":
        values = {
            "min_component_size": (24, 48),
            "closing_radius": (0, 1),
            "distance_sigma": (0.75,),
            "h_maxima": (1.0, 2.5),
            "min_distance": (8, 11, 14),
            "split_area_factor": (1.4, 1.8),
        }
    else:
        values = {
            "min_component_size": (20, 40, 64),
            "closing_radius": (0, 1),
            "distance_sigma": (0.75, 1.25),
            "h_maxima": (1.0, 2.5),
            "min_distance": (8, 11, 14),
            "split_area_factor": (1.4, 1.8),
        }
    keys = list(values)
    return [dict(zip(keys, combination)) for combination in itertools.product(*(values[key] for key in keys))]


def select_trial(trials: pd.DataFrame, aji_tolerance: float) -> pd.Series:
    best_aji = float(trials.mean_validation_aji.max())
    eligible = trials.loc[trials.mean_validation_aji >= best_aji - aji_tolerance].copy()
    eligible["absolute_bias"] = eligible.validation_count_bias.abs()
    return eligible.sort_values(
        ["validation_count_mae", "absolute_bias", "mean_validation_aji"],
        ascending=[True, True, False],
    ).iloc[0]


def overlay_instances(rgb: np.ndarray, instances: np.ndarray) -> np.ndarray:
    output = rgb.copy()
    output[segmentation.find_boundaries(instances, mode="outer")] = np.array([255, 228, 112])
    return output


def save_diagnostics(
    output_root: Path,
    trials: pd.DataFrame,
    selected: pd.Series,
    test_frame: pd.DataFrame,
    v1_test_frame: pd.DataFrame | None,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(trials.validation_count_mae, trials.mean_validation_aji, alpha=0.55, s=24)
    axes[0].scatter(
        [selected.validation_count_mae],
        [selected.mean_validation_aji],
        marker="*",
        s=220,
        color="#ef4444",
        label="selected v2",
    )
    axes[0].set_xlabel("Validation count MAE (lower is better)")
    axes[0].set_ylabel("Validation AJI (higher is better)")
    axes[0].set_title("Validation-only selection")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    maximum = float(max(test_frame.ground_truth_count.max(), test_frame.predicted_count.max()))
    axes[1].plot([0, maximum], [0, maximum], "--", color="#64748b", label="perfect count")
    if v1_test_frame is not None:
        axes[1].scatter(
            v1_test_frame.ground_truth_count,
            v1_test_frame.predicted_count,
            alpha=0.7,
            label="v1",
        )
    axes[1].scatter(
        test_frame.ground_truth_count,
        test_frame.predicted_count,
        alpha=0.8,
        label="v2",
    )
    axes[1].set_xlabel("Ground-truth count")
    axes[1].set_ylabel("Predicted count")
    axes[1].set_title("Held-out counts")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_root / "instance_v2_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--search-profile", choices=("quick", "balanced"), default="balanced")
    parser.add_argument(
        "--aji-tolerance",
        type=float,
        default=0.015,
        help="Select lowest count MAE within this distance of the best validation AJI.",
    )
    parser.add_argument("--force-predictions", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Tune and freeze v2 without opening the benchmark test images.",
    )
    # Colab/IPython injects a -f kernel argument. Ignore unknown notebook flags.
    arguments, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignoring notebook arguments:", " ".join(unknown))
    return arguments


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.project_root)
    artifacts = root / "artifacts"
    checkpoint = artifacts / "best_unet.weights.h5"
    frozen_path = artifacts / "frozen_config.json"
    manifest_path = artifacts / "split_manifest.csv"
    output_root = artifacts / "instance_v2"
    output_root.mkdir(parents=True, exist_ok=True)

    required = (checkpoint, frozen_path, manifest_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required training artifacts:\n- " + "\n- ".join(missing) +
            "\nMount Google Drive and confirm --project-root. No retraining is required."
        )

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(manifest_path)
    expected = {"train": 24, "validation": 6, "test": 14}
    actual = manifest.groupby("split").size().to_dict()
    if any(int(actual.get(split, 0)) != count for split, count in expected.items()):
        raise AssertionError(f"Expected split counts {expected}; found {actual}")

    training_rows = manifest.query("split == 'train'").copy()
    validation_rows = manifest.query("split == 'validation'").copy()
    test_rows = manifest.query("split == 'test'").copy()
    median_area = training_median_nucleus_area(training_rows)
    print(f"Training-derived median nucleus area: {median_area:.1f} pixels")
    print(f"Frozen semantic threshold: {float(frozen['mask_threshold']):.3f}")

    needed_rows = validation_rows if args.validation_only else pd.concat([validation_rows, test_rows])
    probabilities = probability_cache(
        needed_rows,
        checkpoint,
        output_root,
        frozen,
        force=args.force_predictions,
    )
    validation_probabilities = {
        str(row.image_id): probabilities[str(row.image_id)] for _, row in validation_rows.iterrows()
    }
    validation_truth = {
        str(row.image_id): load_instances(row.instance_map_path) for _, row in validation_rows.iterrows()
    }

    print("Evaluating the original v1 instance settings on validation...")
    v1_validation = evaluate_v1(validation_rows, validation_probabilities, validation_truth, frozen)
    print(json.dumps(v1_validation, indent=2))

    grid = search_grid(args.search_profile)
    print(f"Searching {len(grid)} component-aware configurations on validation only...")
    trial_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    threshold = float(frozen["mask_threshold"])
    for index, parameters in enumerate(grid, start=1):
        metrics, _ = evaluate_parameters(
            validation_rows,
            validation_probabilities,
            validation_truth,
            threshold,
            median_area,
            parameters,
        )
        trial_rows.append(
            {
                **parameters,
                "mean_validation_aji": metrics["mean_aji"],
                "validation_count_mae": metrics["count_mae"],
                "validation_relative_mae_percent": metrics["count_relative_mae_percent"],
                "validation_count_bias": metrics["count_bias"],
            }
        )
        if index == 1 or index % 12 == 0 or index == len(grid):
            elapsed = time.perf_counter() - started
            print(f"  trial {index}/{len(grid)} ({elapsed:.1f}s)")

    trials = pd.DataFrame(trial_rows)
    selected = select_trial(trials, args.aji_tolerance)
    parameter_names = (
        "min_component_size",
        "closing_radius",
        "distance_sigma",
        "h_maxima",
        "min_distance",
        "split_area_factor",
    )
    selected_parameters = {
        name: int(selected[name]) if name in {"min_component_size", "closing_radius", "min_distance"}
        else float(selected[name])
        for name in parameter_names
    }
    selected_metrics, _ = evaluate_parameters(
        validation_rows,
        validation_probabilities,
        validation_truth,
        threshold,
        median_area,
        selected_parameters,
    )
    trials.to_csv(output_root / "validation_search.csv", index=False)

    v2_config = {
        "version": "instance_v2_component_aware",
        "selection_split": "validation",
        "selection_rule": (
            "minimum count MAE among configurations within "
            f"{args.aji_tolerance:.3f} AJI of best validation AJI"
        ),
        "semantic_mask_threshold": threshold,
        "semantic_min_object_size": int(frozen["min_object_size"]),
        "median_training_nucleus_area": median_area,
        "parameters": selected_parameters,
        "v1_validation": v1_validation,
        "v2_validation": selected_metrics,
        "test_images_used_for_selection": 0,
    }
    config_path = output_root / "frozen_instance_config_v2.json"
    config_path.write_text(json.dumps(v2_config, indent=2), encoding="utf-8")
    print("\nFrozen v2 configuration:")
    print(json.dumps(v2_config, indent=2))

    if args.validation_only:
        print("\nValidation-only run complete. Benchmark images were not evaluated.")
        print("Frozen configuration:", config_path)
        return

    print("\nConfiguration frozen. Evaluating the 14 benchmark images once...")
    test_probabilities = {
        str(row.image_id): probabilities[str(row.image_id)] for _, row in test_rows.iterrows()
    }
    test_truth = {str(row.image_id): load_instances(row.instance_map_path) for _, row in test_rows.iterrows()}
    test_metrics, test_results = evaluate_parameters(
        test_rows,
        test_probabilities,
        test_truth,
        threshold,
        median_area,
        selected_parameters,
    )

    test_frame = pd.DataFrame(test_results)
    semantic_dice: list[float] = []
    semantic_iou: list[float] = []
    instance_dir = output_root / "predictions" / "instances"
    overlay_dir = output_root / "predictions" / "overlays"
    instance_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for _, row in test_rows.iterrows():
        image_id = str(row.image_id)
        probability = np.asarray(test_probabilities[image_id])
        semantic = semantic_mask(
            probability,
            threshold,
            int(frozen["min_object_size"]),
        )
        semantic_dice.append(dice_score(load_binary(row.binary_mask_path), semantic))
        semantic_iou.append(iou_score(load_binary(row.binary_mask_path), semantic))
        instances = component_aware_watershed(
            probability,
            threshold,
            median_area,
            **selected_parameters,
        )
        save_uint16(instance_dir / f"{image_id}.tif", instances)
        Image.fromarray(overlay_instances(load_rgb(row.image_path), instances)).save(
            overlay_dir / f"{image_id}.png"
        )

    test_metrics["mean_semantic_dice"] = float(np.mean(semantic_dice))
    test_metrics["mean_semantic_iou"] = float(np.mean(semantic_iou))
    test_frame["semantic_dice"] = semantic_dice
    test_frame["semantic_iou"] = semantic_iou
    test_frame.to_csv(output_root / "test_per_image_v2.csv", index=False)

    benchmark_path = artifacts / "resume_benchmark_report.json"
    if not benchmark_path.exists():
        benchmark_path = root / "results" / "benchmark_summary.json"
    v1_benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else None
    comparison = {
        "protocol": "canonical_30_grouped_24_6_test_14",
        "v1": v1_benchmark,
        "v2": test_metrics,
        "v2_minus_v1": {
            "aji": (
                float(test_metrics["mean_aji"] - v1_benchmark["mean_aji"])
                if v1_benchmark and "mean_aji" in v1_benchmark
                else None
            ),
            "count_mae": (
                float(test_metrics["count_mae"] - v1_benchmark["count_mae"])
                if v1_benchmark and "count_mae" in v1_benchmark
                else None
            ),
            "count_bias": (
                float(test_metrics["count_bias"] - v1_benchmark["count_bias"])
                if v1_benchmark and "count_bias" in v1_benchmark
                else None
            ),
        },
        "frozen_config": str(config_path),
    }
    (output_root / "test_summary_v2.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    v1_test_frame = None
    v1_per_image_path = artifacts / "per_image_metrics.csv"
    if v1_per_image_path.exists():
        v1_test_frame = pd.read_csv(v1_per_image_path)
    save_diagnostics(output_root, trials, selected, test_frame, v1_test_frame)

    print("\nInstance v2 benchmark summary:")
    print(json.dumps(comparison, indent=2))
    print("\nOutputs:", output_root)
    print("V1 artifacts were not modified.")


if __name__ == "__main__":
    main()
