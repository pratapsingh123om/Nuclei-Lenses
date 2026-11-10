from __future__ import annotations

import base64
import math
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import color, feature, measure, morphology, segmentation


def decode_rgb(payload: bytes, max_pixels: int) -> np.ndarray:
    if not payload:
        raise ValueError("The uploaded file is empty.")

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError("The file could not be decoded as an image.") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    if rgb.shape[0] * rgb.shape[1] > max_pixels:
        raise ValueError(f"Image exceeds the {max_pixels:,}-pixel limit.")
    return rgb


def clean_binary_mask(probability: np.ndarray, threshold: float, min_size: int) -> np.ndarray:
    mask = probability >= threshold
    mask = morphology.remove_small_objects(mask, min_size=max(1, int(min_size)))
    mask = morphology.remove_small_holes(mask, area_threshold=max(1, int(min_size)))
    return mask.astype(bool)


def split_instances(mask: np.ndarray, min_distance: int, min_size: int) -> np.ndarray:
    cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=max(1, int(min_size)))
    if not cleaned.any():
        return np.zeros(cleaned.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(cleaned)
    coordinates = feature.peak_local_max(
        distance,
        labels=cleaned,
        min_distance=max(1, int(min_distance)),
        exclude_border=False,
    )
    markers = np.zeros(cleaned.shape, dtype=np.int32)
    if coordinates.size:
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1, dtype=np.int32)
    else:
        markers = measure.label(cleaned, connectivity=1)

    instances = segmentation.watershed(-distance, markers, mask=cleaned)
    instances = morphology.remove_small_objects(instances, min_size=max(1, int(min_size)))
    instances, _, _ = segmentation.relabel_sequential(instances)
    return instances.astype(np.int32)


def _merge_small_instances(labels: np.ndarray, min_area: int) -> np.ndarray:
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
            labels[region] = int(np.argmax(np.bincount(neighbors)))
        else:
            labels[region] = 0
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels.astype(np.int32)


def _suppress_nearby_peaks(
    coordinates: list[tuple[int, int]],
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


def split_instances_component_aware(
    probability: np.ndarray,
    threshold: float,
    config: dict[str, Any],
) -> np.ndarray:
    """Run the validation-frozen v2 instance pipeline on a probability map."""

    median_area = max(1.0, float(config["median_nucleus_area"]))
    min_component_size = max(1, int(config["min_component_size"]))
    closing_radius = max(0, int(config["closing_radius"]))
    distance_sigma = max(0.0, float(config["distance_sigma"]))
    h_value = max(1e-6, float(config["h_value"]))
    min_distance = max(1, int(config["min_distance"]))
    split_area_factor = max(1.0, float(config["split_area_factor"]))
    count_cap_factor = max(1.0, float(config.get("count_cap_factor", 1.2)))

    mask = probability >= threshold
    mask = morphology.remove_small_objects(mask, min_size=min_component_size)
    mask = morphology.remove_small_holes(mask, area_threshold=min_component_size)
    if closing_radius:
        mask = morphology.binary_closing(mask, morphology.disk(closing_radius))
    mask = morphology.remove_small_objects(mask, min_size=min_component_size)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    components = measure.label(mask, connectivity=1)
    raw_distance = ndi.distance_transform_edt(mask)
    distance = ndi.gaussian_filter(raw_distance, sigma=distance_sigma) if distance_sigma else raw_distance
    prominent = morphology.h_maxima(distance, h=h_value) & mask
    peak_regions = measure.label(prominent, connectivity=2)

    candidates_by_component: dict[int, list[tuple[int, int]]] = {}
    peak_count = int(peak_regions.max())
    if peak_count:
        positions = ndi.maximum_position(
            distance,
            labels=peak_regions,
            index=np.arange(1, peak_count + 1),
        )
        for position in positions:
            point = (int(position[0]), int(position[1]))
            component_id = int(components[point])
            if component_id:
                candidates_by_component.setdefault(component_id, []).append(point)

    markers = np.zeros(mask.shape, dtype=np.int32)
    marker_id = 0
    component_areas = np.bincount(components.ravel())
    split_area = split_area_factor * median_area
    for component_id in range(1, len(component_areas)):
        component_mask = components == component_id
        component_area = int(component_areas[component_id])
        fallback_flat = int(np.argmax(np.where(component_mask, distance, -np.inf)))
        fallback = tuple(int(value) for value in np.unravel_index(fallback_flat, mask.shape))

        if component_area <= split_area:
            selected = [fallback]
        else:
            candidates = list(candidates_by_component.get(component_id, []))
            if fallback not in candidates:
                candidates.append(fallback)
            estimated_count = max(2, int(math.ceil(component_area / median_area)))
            limit = max(2, int(math.ceil(estimated_count * count_cap_factor)))
            selected = _suppress_nearby_peaks(candidates, distance, min_distance, limit) or [fallback]

        for point in selected:
            marker_id += 1
            markers[point] = marker_id

    instances = segmentation.watershed(-distance, markers, mask=mask)
    return _merge_small_instances(instances, min_component_size)


def _png_data_url(array: np.ndarray, mode: str) -> str:
    buffer = BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_outputs(rgb: np.ndarray, mask: np.ndarray, instances: np.ndarray) -> dict[str, str]:
    mask_u8 = (mask.astype(np.uint8) * 255)

    overlay = rgb.astype(np.float32).copy()
    tint = np.array([28, 214, 176], dtype=np.float32)
    overlay[mask] = overlay[mask] * 0.48 + tint * 0.52
    boundaries = segmentation.find_boundaries(instances, mode="outer")
    overlay[boundaries] = np.array([255, 228, 112], dtype=np.float32)

    if instances.max() > 0:
        instance_rgb = (color.label2rgb(instances, bg_label=0, bg_color=(0.03, 0.07, 0.08)) * 255).astype(np.uint8)
    else:
        instance_rgb = np.zeros((*instances.shape, 3), dtype=np.uint8)

    return {
        "original": _png_data_url(rgb, "RGB"),
        "mask": _png_data_url(mask_u8, "L"),
        "overlay": _png_data_url(np.clip(overlay, 0, 255).astype(np.uint8), "RGB"),
        "instances": _png_data_url(instance_rgb, "RGB"),
    }

