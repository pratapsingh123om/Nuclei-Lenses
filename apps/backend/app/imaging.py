from __future__ import annotations

import base64
from io import BytesIO

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

