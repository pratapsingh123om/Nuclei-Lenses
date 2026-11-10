# %% [markdown]
# # Nuclei Lens — MoNuSeg U-Net and nuclei counting
#
# This Colab notebook builds a reproducible nuclei-segmentation pipeline for
# H&E pathology images. It uses the **original 30-image MoNuSeg training cohort**,
# creates a grouped 24/6 development split before patch sampling, and keeps all
# 14 benchmark images untouched until final evaluation.
#
# Outputs include a color/morphology baseline, a small TensorFlow U-Net,
# overlapping-patch inference, watershed instance separation, Dice, IoU, AJI,
# count MAE, CSV reports, overlays, a deployable checkpoint, and a single-image
# demo. The notebook is a research workflow, not a clinical device.

# %% [markdown]
# ## Session 1 — environment and the inspection gate
#
# Run the separate Phase One inspection cell first. This notebook refuses to
# prepare data unless its report says the archives passed and the canonical
# benchmark is ready.

# %%
import subprocess
import sys

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "scikit-image>=0.25,<1",
    "scikit-learn>=1.5,<2",
    "pandas>=2,<3",
])

# %%
from google.colab import drive

drive.mount("/content/drive")

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET
from collections import defaultdict
from typing import Any
import io
import json
import math
import random
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, exposure, feature, filters, measure, morphology, segmentation
from skimage.draw import polygon as draw_polygon
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf


@dataclass(frozen=True)
class Config:
    project_root: str = "/content/drive/MyDrive/Nuclei-lens/monuseg_project"
    train_archive: str = "/content/drive/MyDrive/Nuclei-lens/MoNuSeg 2018 Training Data.zip"
    test_archive: str = "/content/drive/MyDrive/Nuclei-lens/MoNuSegTestData.zip"
    inspection_report: str = "/content/drive/MyDrive/Nuclei-lens/phase1_inspection/phase1_report.json"
    patch_size: int = 256
    overlap: int = 64
    batch_size: int = 8
    base_filters: int = 32
    epochs: int = 60
    steps_per_epoch: int = 120
    learning_rate: float = 3e-4
    foreground_patch_probability: float = 0.75
    run_overfit_check: bool = True
    overfit_max_steps: int = 300
    seed: int = 42


CFG = Config()
ROOT = Path(CFG.project_root)
PREPARED = ROOT / "prepared"
ARTIFACTS = ROOT / "artifacts"
CHECKPOINT = ARTIFACTS / "best_unet.weights.h5"
FROZEN_CONFIG = ARTIFACTS / "frozen_config.json"
for directory in (PREPARED, ARTIFACTS):
    directory.mkdir(parents=True, exist_ok=True)

random.seed(CFG.seed)
np.random.seed(CFG.seed)
tf.keras.utils.set_random_seed(CFG.seed)

print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print(json.dumps(asdict(CFG), indent=2))

# %%
inspection_path = Path(CFG.inspection_report)
assert inspection_path.exists(), (
    f"Missing {inspection_path}. Run phase1_colab_cell.py first; no training data "
    "will be prepared before the archive inspection passes."
)
inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
integrity_pass = inspection.get("data_integrity_passed", inspection.get("integrity_pass", False))
policy = inspection.get("policy", {})
benchmark_ready = policy.get("benchmark_ready", inspection.get("benchmark_ready", False))
selected_protocol = policy.get("selected_protocol", inspection.get("selected_protocol"))
assert integrity_pass, "Phase One reports an archive-integrity failure. Fix that before training."
assert benchmark_ready, "Phase One reports that the canonical 30 + benchmark 14 protocol is not ready."
assert selected_protocol == "canonical_30", f"Expected canonical_30; report selected {selected_protocol!r}."
print("Inspection gate: PASS")
print("Protocol: canonical_30 (24 train / 6 validation / 14 final test)")

# %% [markdown]
# ## Prepare images, binary masks, and integer instance maps
#
# Boundary-crossing annotation coordinates are clipped to the image canvas.
# Every remaining polygon receives its own positive integer ID. This preserves
# per-nucleus labels for AJI and count evaluation.

# %%
CANONICAL_TRAIN_ORGANS = {
    "breast": [
        "TCGA-A7-A13E-01Z-00-DX1", "TCGA-A7-A13F-01Z-00-DX1",
        "TCGA-AR-A1AK-01Z-00-DX1", "TCGA-AR-A1AS-01Z-00-DX1",
        "TCGA-E2-A1B5-01Z-00-DX1", "TCGA-E2-A14V-01Z-00-DX1",
    ],
    "kidney": [
        "TCGA-B0-5711-01Z-00-DX1", "TCGA-HE-7128-01Z-00-DX1",
        "TCGA-HE-7129-01Z-00-DX1", "TCGA-HE-7130-01Z-00-DX1",
        "TCGA-B0-5710-01Z-00-DX1", "TCGA-B0-5698-01Z-00-DX1",
    ],
    "liver": [
        "TCGA-18-5592-01Z-00-DX1", "TCGA-38-6178-01Z-00-DX1",
        "TCGA-49-4488-01Z-00-DX1", "TCGA-50-5931-01Z-00-DX1",
        "TCGA-21-5784-01Z-00-DX1", "TCGA-21-5786-01Z-00-DX1",
    ],
    "prostate": [
        "TCGA-G9-6336-01Z-00-DX1", "TCGA-G9-6348-01Z-00-DX1",
        "TCGA-G9-6356-01Z-00-DX1", "TCGA-G9-6363-01Z-00-DX1",
        "TCGA-CH-5767-01Z-00-DX1", "TCGA-G9-6362-01Z-00-DX1",
    ],
    "bladder": ["TCGA-DK-A2I6-01A-01-TS1", "TCGA-G2-A2EK-01A-02-TSB"],
    "colon": ["TCGA-AY-A8YK-01A-01-TS1", "TCGA-NH-A8F7-01A-01-TS1"],
    "stomach": ["TCGA-KB-A93J-01A-01-TS1", "TCGA-RD-A8N9-01A-01-TS1"],
}
CANONICAL_IDS = {item for values in CANONICAL_TRAIN_ORGANS.values() for item in values}
ORGAN_BY_ID = {item: organ for organ, values in CANONICAL_TRAIN_ORGANS.items() for item in values}


def normalized_stem(member: str) -> str:
    value = PurePosixPath(member).stem
    for suffix in ("_annotations", "_annotation", "_annot"):
        if value.lower().endswith(suffix):
            return value[:-len(suffix)]
    return value


def patient_id(image_id: str) -> str:
    parts = image_id.split("-")
    return "-".join(parts[:3]).upper() if len(parts) >= 3 and parts[0].upper() == "TCGA" else image_id


def tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def archive_pairs(handle: zipfile.ZipFile) -> dict[str, tuple[str, str]]:
    image_map: dict[str, list[str]] = defaultdict(list)
    xml_map: dict[str, list[str]] = defaultdict(list)
    for info in handle.infolist():
        if info.is_dir():
            continue
        parts = PurePosixPath(info.filename).parts
        if "__MACOSX" in parts or any(part.startswith("._") for part in parts):
            continue
        suffix = PurePosixPath(info.filename).suffix.lower()
        if suffix in {".tif", ".tiff"}:
            image_map[normalized_stem(info.filename)].append(info.filename)
        elif suffix == ".xml":
            xml_map[normalized_stem(info.filename)].append(info.filename)
    pairs = {}
    for image_id in sorted(set(image_map) | set(xml_map)):
        if len(image_map[image_id]) == 1 and len(xml_map[image_id]) == 1:
            pairs[image_id] = (image_map[image_id][0], xml_map[image_id][0])
    return pairs


def polygons_from_xml(payload: bytes, width: int, height: int) -> tuple[list[np.ndarray], int]:
    root = ET.fromstring(payload)
    polygons: list[np.ndarray] = []
    skipped = 0
    for region in (node for node in root.iter() if tag_name(node.tag) == "Region"):
        points = []
        for node in region.iter():
            if tag_name(node.tag) != "Vertex":
                continue
            try:
                x, y = float(node.attrib["X"]), float(node.attrib["Y"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                points.append((np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)))
        polygon = np.asarray(points, dtype=np.float32)
        if len(polygon) < 3 or len(np.unique(np.rint(polygon).astype(int), axis=0)) < 3:
            skipped += 1
            continue
        area = 0.5 * abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
        if area <= 0.5:
            skipped += 1
            continue
        polygons.append(polygon)
    return polygons, skipped


def rasterize_instances(polygons: list[np.ndarray], height: int, width: int) -> np.ndarray:
    instances = np.zeros((height, width), dtype=np.uint16)
    instance_id = 0
    for points in polygons:
        rows, columns = draw_polygon(points[:, 1], points[:, 0], shape=instances.shape)
        if len(rows) == 0:
            continue
        instance_id += 1
        instances[rows, columns] = instance_id
    instances, _, _ = segmentation.relabel_sequential(instances)
    return instances.astype(np.uint16)


def save_uint16(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint16), mode="I;16").save(path)


def prepare_archive(archive_path: str, split: str, selected_ids: set[str] | None) -> list[dict[str, Any]]:
    destination = PREPARED / split
    image_dir, binary_dir, instance_dir = (
        destination / "images", destination / "binary_masks", destination / "instance_maps"
    )
    for directory in (image_dir, binary_dir, instance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    with zipfile.ZipFile(archive_path) as handle:
        pairs = archive_pairs(handle)
        ids = sorted(selected_ids if selected_ids is not None else pairs)
        missing = sorted(set(ids) - set(pairs))
        assert not missing, f"Missing selected {split} IDs: {missing}"
        for index, image_id in enumerate(ids, start=1):
            image_member, xml_member = pairs[image_id]
            with Image.open(io.BytesIO(handle.read(image_member))) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            polygons, skipped = polygons_from_xml(handle.read(xml_member), rgb.shape[1], rgb.shape[0])
            instances = rasterize_instances(polygons, rgb.shape[0], rgb.shape[1])
            assert instances.max() > 0, f"No usable nuclei for {image_id}"
            image_path = image_dir / f"{image_id}.png"
            binary_path = binary_dir / f"{image_id}.png"
            instance_path = instance_dir / f"{image_id}.tif"
            Image.fromarray(rgb).save(image_path)
            Image.fromarray((instances > 0).astype(np.uint8) * 255).save(binary_path)
            save_uint16(instance_path, instances)
            records.append({
                "image_id": image_id,
                "patient_id": patient_id(image_id),
                "source": split,
                "organ": ORGAN_BY_ID.get(image_id, "unknown"),
                "image_path": str(image_path),
                "binary_mask_path": str(binary_path),
                "instance_map_path": str(instance_path),
                "nuclei_count": int(instances.max()),
                "skipped_regions": skipped,
            })
            if index == 1 or index % 10 == 0 or index == len(ids):
                print(f"Prepared {split}: {index}/{len(ids)}")
    return records


train_records = prepare_archive(CFG.train_archive, "train", CANONICAL_IDS)
test_records = prepare_archive(CFG.test_archive, "test", None)
assert len(train_records) == 30 and len(test_records) == 14
data_index = pd.DataFrame(train_records + test_records)
data_index.to_csv(ARTIFACTS / "data_index.csv", index=False)
data_index.groupby("source").agg(images=("image_id", "count"), nuclei=("nuclei_count", "sum"))

# %% [markdown]
# ## Grouped 24/6 split before patch extraction
#
# The split is based on source image and patient ID. The assertions below make
# leakage impossible before any random patch is sampled.

# %%
train_frame = pd.DataFrame(train_records).sort_values("image_id").reset_index(drop=True)
groups = train_frame["patient_id"].to_numpy()
splitter = GroupShuffleSplit(n_splits=200, test_size=0.2, random_state=CFG.seed)
candidates = []
for train_indices, validation_indices in splitter.split(train_frame, groups=groups):
    if len(validation_indices) == 6:
        validation_organs = train_frame.iloc[validation_indices]["organ"].nunique()
        candidates.append((validation_organs, train_indices, validation_indices))
assert candidates, "Could not construct an exact 24/6 grouped split."
_, train_indices, validation_indices = max(candidates, key=lambda item: item[0])

train_frame["split"] = "train"
train_frame.loc[validation_indices, "split"] = "validation"
test_frame = pd.DataFrame(test_records)
test_frame["split"] = "test"
manifest = pd.concat([train_frame, test_frame], ignore_index=True)

development_groups = set(manifest.loc[manifest.split == "train", "patient_id"])
validation_groups = set(manifest.loc[manifest.split == "validation", "patient_id"])
assert len(manifest.query("split == 'train'")) == 24
assert len(manifest.query("split == 'validation'")) == 6
assert len(manifest.query("split == 'test'")) == 14
assert development_groups.isdisjoint(validation_groups)
assert manifest.groupby("image_id")["split"].nunique().max() == 1
manifest.to_csv(ARTIFACTS / "split_manifest.csv", index=False)
display(manifest.groupby(["split", "organ"]).size().unstack(fill_value=0))

# %%
def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_binary(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path)) > 0


def load_instances(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.int32)


def show_prepared_examples(rows: pd.DataFrame, count: int = 3) -> None:
    sample = rows.sample(min(count, len(rows)), random_state=CFG.seed)
    figure, axes = plt.subplots(len(sample), 3, figsize=(13, 4 * len(sample)))
    axes = np.atleast_2d(axes)
    for row_axes, (_, row) in zip(axes, sample.iterrows()):
        rgb, binary, instances = load_rgb(row.image_path), load_binary(row.binary_mask_path), load_instances(row.instance_map_path)
        row_axes[0].imshow(rgb); row_axes[0].set_title(row.image_id)
        row_axes[1].imshow(binary, cmap="gray"); row_axes[1].set_title("Binary mask")
        row_axes[2].imshow(color.label2rgb(instances, bg_label=0)); row_axes[2].set_title(f"Instances: {instances.max()}")
        for axis in row_axes: axis.axis("off")
    plt.tight_layout()


show_prepared_examples(manifest.query("split == 'train'"))

# %% [markdown]
# ## Color/morphology baseline
#
# The baseline isolates hematoxylin with HED color deconvolution, applies an
# Otsu threshold plus a validation-selected offset, and cleans the result with
# simple morphology. It is evaluated on the same full images as the U-Net.

# %%
def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)
    denominator = y_true.sum() + y_pred.sum()
    return float((2 * np.logical_and(y_true, y_pred).sum()) / denominator) if denominator else 1.0


def iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    union = np.logical_or(y_true, y_pred).sum()
    return float(np.logical_and(y_true, y_pred).sum() / union) if union else 1.0


def clean_mask(mask: np.ndarray, min_size: int = 12) -> np.ndarray:
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=min_size)
    return morphology.binary_opening(mask, morphology.disk(1))


def hematoxylin_score(rgb: np.ndarray) -> np.ndarray:
    channel = color.rgb2hed(rgb)[..., 0]
    low, high = np.percentile(channel, (1, 99))
    return exposure.rescale_intensity(channel, in_range=(low, max(high, low + 1e-6)), out_range=(0, 1))


def baseline_mask(rgb: np.ndarray, offset: float, min_size: int = 12) -> np.ndarray:
    score = hematoxylin_score(rgb)
    return clean_mask(score > (filters.threshold_otsu(score) + offset), min_size=min_size)


validation_rows = manifest.query("split == 'validation'")
baseline_trials = []
for offset in np.linspace(-0.12, 0.12, 13):
    scores = []
    for _, row in validation_rows.iterrows():
        scores.append(dice_score(load_binary(row.binary_mask_path), baseline_mask(load_rgb(row.image_path), float(offset))))
    baseline_trials.append({"offset": float(offset), "mean_validation_dice": float(np.mean(scores))})
baseline_trials = pd.DataFrame(baseline_trials)
BASELINE_OFFSET = float(baseline_trials.loc[baseline_trials.mean_validation_dice.idxmax(), "offset"])
print("Frozen baseline offset:", BASELINE_OFFSET)
display(baseline_trials)

# %% [markdown]
# ## Session 2 — patch pipeline and 2D U-Net
#
# First run the optional four-patch overfit check. A model that cannot memorize
# a tiny batch usually signals a mask, loss, shape, or optimization bug.

# %%
def arrays_for_split(split: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rows = manifest.query("split == @split")
    return (
        [load_rgb(path) for path in rows.image_path],
        [load_binary(path).astype(np.float32)[..., None] for path in rows.binary_mask_path],
    )


train_images, train_masks = arrays_for_split("train")
validation_images, validation_masks = arrays_for_split("validation")


def sample_patch(images, masks, augment: bool, rng: np.random.Generator):
    index = int(rng.integers(len(images)))
    image, mask = images[index], masks[index]
    size = CFG.patch_size
    if rng.random() < CFG.foreground_patch_probability and mask[..., 0].any():
        foreground_coordinates = np.argwhere(mask[..., 0] > 0)
        y_center, x_center = foreground_coordinates[int(rng.integers(len(foreground_coordinates)))]
        y0 = int(np.clip(y_center - rng.integers(0, size), 0, image.shape[0] - size))
        x0 = int(np.clip(x_center - rng.integers(0, size), 0, image.shape[1] - size))
    else:
        y0 = int(rng.integers(0, image.shape[0] - size + 1))
        x0 = int(rng.integers(0, image.shape[1] - size + 1))
    image_patch = image[y0:y0 + size, x0:x0 + size].astype(np.float32) / 255.0
    mask_patch = mask[y0:y0 + size, x0:x0 + size].astype(np.float32)
    if augment:
        rotations = int(rng.integers(4))
        image_patch, mask_patch = np.rot90(image_patch, rotations), np.rot90(mask_patch, rotations)
        if rng.random() < 0.5:
            image_patch, mask_patch = np.flip(image_patch, 0), np.flip(mask_patch, 0)
        if rng.random() < 0.5:
            image_patch, mask_patch = np.flip(image_patch, 1), np.flip(mask_patch, 1)
        gain = rng.uniform(0.92, 1.08, size=(1, 1, 3))
        bias = rng.uniform(-0.035, 0.035, size=(1, 1, 3))
        image_patch = np.clip(image_patch * gain + bias, 0, 1)
    return np.ascontiguousarray(image_patch), np.ascontiguousarray(mask_patch)


def patch_dataset(images, masks, augment: bool, seed: int) -> tf.data.Dataset:
    def generator():
        rng = np.random.default_rng(seed)
        while True:
            yield sample_patch(images, masks, augment, rng)
    signature = (
        tf.TensorSpec((CFG.patch_size, CFG.patch_size, 3), tf.float32),
        tf.TensorSpec((CFG.patch_size, CFG.patch_size, 1), tf.float32),
    )
    return (
        tf.data.Dataset.from_generator(generator, output_signature=signature)
        .batch(CFG.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


training_dataset = patch_dataset(train_images, train_masks, augment=True, seed=CFG.seed)
validation_dataset = patch_dataset(validation_images, validation_masks, augment=False, seed=CFG.seed + 1)

# %%
def convolution_block(x, filters_count: int):
    x = tf.keras.layers.Conv2D(filters_count, 3, padding="same", activation="relu")(x)
    return tf.keras.layers.Conv2D(filters_count, 3, padding="same", activation="relu")(x)


def build_unet() -> tf.keras.Model:
    inputs = tf.keras.Input((CFG.patch_size, CFG.patch_size, 3), name="image")
    c1 = convolution_block(inputs, CFG.base_filters)
    c2 = convolution_block(tf.keras.layers.MaxPooling2D()(c1), CFG.base_filters * 2)
    c3 = convolution_block(tf.keras.layers.MaxPooling2D()(c2), CFG.base_filters * 4)
    c4 = convolution_block(tf.keras.layers.MaxPooling2D()(c3), CFG.base_filters * 8)
    bridge = convolution_block(tf.keras.layers.MaxPooling2D()(c4), CFG.base_filters * 16)
    x = bridge
    for skip, filters_count in (
        (c4, CFG.base_filters * 8), (c3, CFG.base_filters * 4),
        (c2, CFG.base_filters * 2), (c1, CFG.base_filters),
    ):
        x = tf.keras.layers.Conv2DTranspose(filters_count, 2, strides=2, padding="same")(x)
        x = convolution_block(tf.keras.layers.Concatenate()([x, skip]), filters_count)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="nuclei_probability")(x)
    return tf.keras.Model(inputs, outputs, name="nuclei_lens_unet")


def soft_dice(y_true, y_pred, smooth=1.0):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    axes = (1, 2, 3)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
    return tf.reduce_mean((2.0 * intersection + smooth) / (denominator + smooth))


def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + (1.0 - soft_dice(y_true, y_pred))


def compiled_model() -> tf.keras.Model:
    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CFG.learning_rate),
        loss=bce_dice_loss,
        metrics=[soft_dice],
    )
    return model


model = compiled_model()
model.summary()

# %%
if CFG.run_overfit_check:
    # Use the same four non-augmented, nucleus-containing patches for every
    # update. Forty updates are not enough for this model; the purpose here is
    # to verify that the pipeline can memorize, not to estimate generalization.
    overfit_rng = np.random.default_rng(CFG.seed + 99)
    fixed_samples = []
    attempts = 0
    while len(fixed_samples) < 4 and attempts < 100:
        candidate = sample_patch(train_images, train_masks, augment=False, rng=overfit_rng)
        attempts += 1
        if float(candidate[1].mean()) >= 0.01:
            fixed_samples.append(candidate)
    assert len(fixed_samples) == 4, "Could not sample four foreground patches; inspect prepared masks."
    overfit_images = tf.convert_to_tensor(np.stack([item[0] for item in fixed_samples]), tf.float32)
    overfit_masks = tf.convert_to_tensor(np.stack([item[1] for item in fixed_samples]), tf.float32)
    overfit_model = compiled_model()
    overfit_model.optimizer.learning_rate.assign(1e-3)
    final_tiny_dice = 0.0
    completed_steps = 0
    for step in range(1, CFG.overfit_max_steps + 1):
        overfit_model.train_on_batch(overfit_images, overfit_masks)
        if step == 1 or step % 25 == 0:
            tiny_prediction = overfit_model(overfit_images, training=False)
            final_tiny_dice = float(soft_dice(overfit_masks, tiny_prediction).numpy())
            print(f"Overfit step {step:3d}: soft Dice {final_tiny_dice:.4f}")
            completed_steps = step
            if final_tiny_dice >= 0.90:
                break
    print(f"Four-patch overfit Dice: {final_tiny_dice:.4f} after {completed_steps} updates")
    assert final_tiny_dice >= 0.85, "Overfit check failed; inspect masks, loss, and model before full training."
    del overfit_model

# %% [markdown]
# ## Full-image reconstruction and checkpoint selection
#
# Validation is performed on reconstructed 1000×1000 images, not isolated
# patches. The callback saves the weights with the best mean per-image Dice.

# %%
def window_starts(length: int, patch_size: int, overlap: int) -> list[int]:
    if length <= patch_size:
        return [0]
    stride = max(1, patch_size - overlap)
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def overlapping_predict(
    active_model: tf.keras.Model,
    rgb: np.ndarray,
    patch_size: int = CFG.patch_size,
    overlap: int = CFG.overlap,
    batch_size: int = CFG.batch_size,
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
        batch_coordinates = coordinates[start:start + batch_size]
        batch = np.stack([
            padded[y:y + patch_size, x:x + patch_size] for y, x in batch_coordinates
        ]).astype(np.float32) / 255.0
        predictions = active_model.predict(batch, verbose=0)[..., 0]
        for (y, x), prediction in zip(batch_coordinates, predictions):
            probability_sum[y:y + patch_size, x:x + patch_size] += prediction
            visit_count[y:y + patch_size, x:x + patch_size] += 1
    return (probability_sum / np.maximum(visit_count, 1))[:height, :width]


class ReconstructedValidation(tf.keras.callbacks.Callback):
    def __init__(self, rows: pd.DataFrame, checkpoint: Path):
        super().__init__()
        self.rows = rows.copy()
        self.checkpoint = checkpoint
        self.best = -np.inf
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        scores = []
        for _, row in self.rows.iterrows():
            probability = overlapping_predict(self.model, load_rgb(row.image_path))
            prediction = clean_mask(probability >= 0.5)
            scores.append(dice_score(load_binary(row.binary_mask_path), prediction))
        mean_dice = float(np.mean(scores))
        logs = logs if logs is not None else {}
        logs["val_reconstructed_dice"] = mean_dice
        self.history.append({"epoch": epoch + 1, "val_reconstructed_dice": mean_dice})
        print(f" — val_reconstructed_dice: {mean_dice:.4f}")
        if mean_dice > self.best:
            self.best = mean_dice
            self.model.save_weights(self.checkpoint)
            pd.DataFrame(self.history).to_csv(ARTIFACTS / "reconstructed_validation_history.csv", index=False)


reconstructed_callback = ReconstructedValidation(validation_rows, CHECKPOINT)
history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    steps_per_epoch=CFG.steps_per_epoch,
    validation_steps=20,
    epochs=CFG.epochs,
    callbacks=[
        reconstructed_callback,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_reconstructed_dice", mode="max", factor=0.5, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_reconstructed_dice", mode="max", patience=12, restore_best_weights=False
        ),
        tf.keras.callbacks.CSVLogger(ARTIFACTS / "training_history.csv"),
    ],
)
assert CHECKPOINT.exists(), "No reconstructed-validation checkpoint was written."
model.load_weights(CHECKPOINT)
print("Reloaded best checkpoint:", CHECKPOINT)

# %%
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="patch validation")
axes[0].set_title("BCE + Dice loss"); axes[0].legend()
reconstructed_history = pd.DataFrame(reconstructed_callback.history)
axes[1].plot(reconstructed_history.epoch, reconstructed_history.val_reconstructed_dice)
axes[1].set_title("Reconstructed validation Dice")
for axis in axes: axis.set_xlabel("Epoch"); axis.grid(alpha=0.2)
plt.tight_layout()

# %% [markdown]
# ## Session 3 — freeze semantic and instance settings on validation
#
# The U-Net threshold is selected by mean validation Dice. Watershed parameters
# are selected by mean validation AJI. Both are then written to a frozen JSON
# file before the benchmark test images are evaluated.

# %%
def watershed_instances(mask: np.ndarray, min_distance: int, min_size: int) -> np.ndarray:
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


def aggregated_jaccard_index(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
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
        ious = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)
        candidate_pairs = np.argwhere(intersections > 0)
        candidate_pairs = sorted(candidate_pairs, key=lambda pair: ious[tuple(pair)], reverse=True)
    else:
        unions = np.zeros((gt_count, pred_count), dtype=np.int64)
        candidate_pairs = []
    matched_gt, matched_pred = set(), set()
    intersection_sum = union_sum = 0
    for gt_index, pred_index in candidate_pairs:
        if int(gt_index) in matched_gt or int(pred_index) in matched_pred:
            continue
        matched_gt.add(int(gt_index)); matched_pred.add(int(pred_index))
        intersection_sum += int(intersections[gt_index, pred_index])
        union_sum += int(unions[gt_index, pred_index])
    union_sum += int(sum(gt_areas[index] for index in range(gt_count) if index not in matched_gt))
    union_sum += int(sum(pred_areas[index] for index in range(pred_count) if index not in matched_pred))
    return float(intersection_sum / union_sum) if union_sum else 1.0


validation_probabilities = {}
for _, row in validation_rows.iterrows():
    validation_probabilities[row.image_id] = overlapping_predict(model, load_rgb(row.image_path))

threshold_trials = []
for threshold in np.linspace(0.25, 0.75, 21):
    scores = [
        dice_score(
            load_binary(row.binary_mask_path),
            clean_mask(validation_probabilities[row.image_id] >= threshold),
        )
        for _, row in validation_rows.iterrows()
    ]
    threshold_trials.append({"threshold": float(threshold), "mean_validation_dice": float(np.mean(scores))})
threshold_trials = pd.DataFrame(threshold_trials)
MASK_THRESHOLD = float(threshold_trials.loc[threshold_trials.mean_validation_dice.idxmax(), "threshold"])

watershed_trials = []
for min_distance in (3, 5, 7, 9):
    for min_size in (8, 12, 20):
        scores = []
        for _, row in validation_rows.iterrows():
            binary = clean_mask(validation_probabilities[row.image_id] >= MASK_THRESHOLD, min_size=min_size)
            prediction = watershed_instances(binary, min_distance=min_distance, min_size=min_size)
            scores.append(aggregated_jaccard_index(load_instances(row.instance_map_path), prediction))
        watershed_trials.append({
            "watershed_min_distance": min_distance,
            "min_object_size": min_size,
            "mean_validation_aji": float(np.mean(scores)),
        })
watershed_trials = pd.DataFrame(watershed_trials)
best_watershed = watershed_trials.loc[watershed_trials.mean_validation_aji.idxmax()]
WATERSHED_MIN_DISTANCE = int(best_watershed.watershed_min_distance)
MIN_OBJECT_SIZE = int(best_watershed.min_object_size)

frozen = {
    "protocol": "canonical_30",
    "patch_size": CFG.patch_size,
    "overlap": CFG.overlap,
    "base_filters": CFG.base_filters,
    "inference_batch_size": CFG.batch_size,
    "mask_threshold": MASK_THRESHOLD,
    "min_object_size": MIN_OBJECT_SIZE,
    "watershed_min_distance": WATERSHED_MIN_DISTANCE,
    "baseline_offset": BASELINE_OFFSET,
    "selection_split": "validation",
    "seed": CFG.seed,
}
FROZEN_CONFIG.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
threshold_trials.to_csv(ARTIFACTS / "threshold_trials.csv", index=False)
watershed_trials.to_csv(ARTIFACTS / "watershed_trials.csv", index=False)
print(json.dumps(frozen, indent=2))

# %% [markdown]
# ## Held-out benchmark evaluation
#
# This is the first section that reads test images. Settings above are frozen;
# no test result is used to change the model, threshold, or watershed.

# %%
def overlay_mask(rgb: np.ndarray, mask: np.ndarray, instances: np.ndarray | None = None) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    output[mask] = output[mask] * 0.48 + np.array([28, 214, 176]) * 0.52
    if instances is not None:
        output[segmentation.find_boundaries(instances, mode="outer")] = np.array([255, 228, 112])
    return np.clip(output, 0, 255).astype(np.uint8)


def save_prediction_set(method: str, image_id: str, binary: np.ndarray, instances: np.ndarray | None, overlay: np.ndarray):
    root = ARTIFACTS / "predictions" / method
    for name in ("binary", "instances", "overlays"):
        (root / name).mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary.astype(np.uint8) * 255).save(root / "binary" / f"{image_id}.png")
    if instances is not None:
        save_uint16(root / "instances" / f"{image_id}.tif", instances)
    Image.fromarray(overlay).save(root / "overlays" / f"{image_id}.png")


test_rows = manifest.query("split == 'test'").copy()
evaluation_rows = []
started = time.perf_counter()
for number, (_, row) in enumerate(test_rows.iterrows(), start=1):
    rgb = load_rgb(row.image_path)
    ground_truth_binary = load_binary(row.binary_mask_path)
    ground_truth_instances = load_instances(row.instance_map_path)

    baseline_binary = baseline_mask(rgb, BASELINE_OFFSET, min_size=MIN_OBJECT_SIZE)
    save_prediction_set("baseline", row.image_id, baseline_binary, None, overlay_mask(rgb, baseline_binary))

    probability = overlapping_predict(model, rgb)
    unet_binary = clean_mask(probability >= MASK_THRESHOLD, min_size=MIN_OBJECT_SIZE)
    predicted_instances = watershed_instances(
        unet_binary,
        min_distance=WATERSHED_MIN_DISTANCE,
        min_size=MIN_OBJECT_SIZE,
    )
    save_prediction_set(
        "unet",
        row.image_id,
        unet_binary,
        predicted_instances,
        overlay_mask(rgb, unet_binary, predicted_instances),
    )
    evaluation_rows.append({
        "image_id": row.image_id,
        "baseline_dice": dice_score(ground_truth_binary, baseline_binary),
        "baseline_iou": iou_score(ground_truth_binary, baseline_binary),
        "unet_dice": dice_score(ground_truth_binary, unet_binary),
        "unet_iou": iou_score(ground_truth_binary, unet_binary),
        "aji": aggregated_jaccard_index(ground_truth_instances, predicted_instances),
        "ground_truth_count": int(ground_truth_instances.max()),
        "predicted_count": int(predicted_instances.max()),
        "count_absolute_error": abs(int(ground_truth_instances.max()) - int(predicted_instances.max())),
    })
    print(f"Evaluated test image {number}/{len(test_rows)}")

per_image = pd.DataFrame(evaluation_rows)
per_image.to_csv(ARTIFACTS / "per_image_metrics.csv", index=False)
summary = pd.DataFrame([
    {
        "method": "color_morphology_baseline",
        "images": len(per_image),
        "mean_dice": per_image.baseline_dice.mean(),
        "mean_iou": per_image.baseline_iou.mean(),
        "mean_aji": np.nan,
        "count_mae": np.nan,
    },
    {
        "method": "unet_watershed",
        "images": len(per_image),
        "mean_dice": per_image.unet_dice.mean(),
        "mean_iou": per_image.unet_iou.mean(),
        "mean_aji": per_image.aji.mean(),
        "count_mae": per_image.count_absolute_error.mean(),
    },
])
summary.to_csv(ARTIFACTS / "summary_metrics.csv", index=False)
evaluation_seconds = time.perf_counter() - started
display(summary)
display(per_image.sort_values("unet_dice", ascending=False))
print(f"Held-out evaluation time: {evaluation_seconds:.1f} seconds")

# %%
ranked = per_image.sort_values("unet_dice").reset_index(drop=True)
case_ids = {
    "failure": ranked.iloc[0].image_id,
    "typical": ranked.iloc[len(ranked) // 2].image_id,
    "strong": ranked.iloc[-1].image_id,
}
figure, axes = plt.subplots(3, 3, figsize=(15, 14))
for row_index, (case_name, image_id) in enumerate(case_ids.items()):
    row = test_rows.loc[test_rows.image_id == image_id].iloc[0]
    rgb = load_rgb(row.image_path)
    ground_truth = load_binary(row.binary_mask_path)
    predicted_overlay = load_rgb(ARTIFACTS / "predictions" / "unet" / "overlays" / f"{image_id}.png")
    prediction = load_binary(ARTIFACTS / "predictions" / "unet" / "binary" / f"{image_id}.png")
    error = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    error[np.logical_and(prediction, ~ground_truth)] = (255, 85, 90)
    error[np.logical_and(~prediction, ground_truth)] = (65, 150, 255)
    axes[row_index, 0].imshow(rgb); axes[row_index, 0].set_title(f"{case_name.title()}: {image_id}")
    axes[row_index, 1].imshow(predicted_overlay); axes[row_index, 1].set_title("Prediction overlay")
    axes[row_index, 2].imshow(error); axes[row_index, 2].set_title("Error: red FP / blue FN")
    for axis in axes[row_index]: axis.axis("off")
plt.tight_layout()
figure.savefig(ARTIFACTS / "cases_strong_typical_failure.png", dpi=180, bbox_inches="tight")

# %% [markdown]
# ## Session 4 — single-image demo and deployment artifacts
#
# The demo accepts any supported H&E image path. The checkpoint and frozen JSON
# use the exact filenames expected by the FastAPI backend.

# %%
def run_demo(image_path: str | Path) -> dict[str, Any]:
    rgb = load_rgb(image_path)
    probability = overlapping_predict(model, rgb)
    binary = clean_mask(probability >= MASK_THRESHOLD, min_size=MIN_OBJECT_SIZE)
    instances = watershed_instances(binary, WATERSHED_MIN_DISTANCE, MIN_OBJECT_SIZE)
    overlay = overlay_mask(rgb, binary, instances)
    instance_rgb = color.label2rgb(instances, bg_label=0, bg_color=(0.03, 0.07, 0.08))
    figure, axes = plt.subplots(1, 4, figsize=(20, 5))
    for axis, image, title in zip(
        axes,
        (rgb, binary, overlay, instance_rgb),
        ("Original", "Predicted mask", "Overlay", f"Instances · count {instances.max()}"),
    ):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title); axis.axis("off")
    plt.tight_layout()
    return {
        "predicted_count": int(instances.max()),
        "nuclei_area_percent": float(binary.mean() * 100),
        "binary_mask": binary,
        "instance_map": instances,
        "overlay": overlay,
    }


DEMO_IMAGE = test_rows.iloc[0].image_path
demo_result = run_demo(DEMO_IMAGE)
print({key: value for key, value in demo_result.items() if not isinstance(value, np.ndarray)})

# %%
run_report = f"""# Nuclei Lens run report

- Protocol: canonical 30 training images, grouped 24/6 development split
- Final benchmark images: {len(per_image)}
- U-Net mean per-image Dice: {per_image.unet_dice.mean():.4f}
- U-Net mean per-image IoU: {per_image.unet_iou.mean():.4f}
- Watershed mean AJI: {per_image.aji.mean():.4f}
- Count MAE: {per_image.count_absolute_error.mean():.2f} nuclei/image
- Baseline mean Dice: {per_image.baseline_dice.mean():.4f}
- Baseline mean IoU: {per_image.baseline_iou.mean():.4f}

Settings were selected on validation images and frozen before test evaluation.
This research workflow is not validated for clinical use.
"""
(ARTIFACTS / "run_report.md").write_text(run_report, encoding="utf-8")
print(run_report)
print("Checkpoint for apps/backend/model:", CHECKPOINT)
print("Configuration for apps/backend/model:", FROZEN_CONFIG)
print("All reports:", ARTIFACTS)

