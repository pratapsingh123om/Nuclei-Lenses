from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


@dataclass
class ModelState:
    ready: bool = False
    status: str = "setup_required"
    message: str = "Copy the trained checkpoint and frozen configuration into the model directory."
    config: dict[str, Any] = field(default_factory=dict)


def _starts(length: int, patch_size: int, overlap: int) -> list[int]:
    if length <= patch_size:
        return [0]
    stride = max(1, patch_size - overlap)
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def _build_unet(tf: Any, patch_size: int, base_filters: int):
    def block(x, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)

    inputs = tf.keras.Input((patch_size, patch_size, 3), name="image")
    c1 = block(inputs, base_filters)
    c2 = block(tf.keras.layers.MaxPooling2D()(c1), base_filters * 2)
    c3 = block(tf.keras.layers.MaxPooling2D()(c2), base_filters * 4)
    c4 = block(tf.keras.layers.MaxPooling2D()(c3), base_filters * 8)
    bridge = block(tf.keras.layers.MaxPooling2D()(c4), base_filters * 16)

    x = bridge
    for skip, filters in ((c4, base_filters * 8), (c3, base_filters * 4), (c2, base_filters * 2), (c1, base_filters)):
        x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
        x = block(tf.keras.layers.Concatenate()([x, skip]), filters)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="nuclei_probability")(x)
    return tf.keras.Model(inputs, outputs, name="nuclei_lens_unet")


class InferenceService:
    def __init__(self, model_dir: str | Path | None = None):
        configured = model_dir or os.getenv("NUCLEI_MODEL_DIR")
        self.model_dir = Path(configured) if configured else Path(__file__).resolve().parents[1] / "model"
        self.weights_path = self.model_dir / "best_unet.weights.h5"
        self.config_path = self.model_dir / "frozen_config.json"
        self.model = None
        self.state = ModelState()
        self._lock = Lock()

    def load(self) -> ModelState:
        missing = [path.name for path in (self.weights_path, self.config_path) if not path.exists()]
        if missing:
            self.state = ModelState(message=f"Missing model artifact(s): {', '.join(missing)}.")
            return self.state

        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            import tensorflow as tf

            patch_size = int(config.get("patch_size", 256))
            base_filters = int(config.get("base_filters", 32))
            model = _build_unet(tf, patch_size, base_filters)
            model.load_weights(self.weights_path)
            self.model = model
            self.state = ModelState(
                ready=True,
                status="ready",
                message="U-Net checkpoint loaded and ready for inference.",
                config=config,
            )
        except Exception as exc:
            self.model = None
            self.state = ModelState(status="model_error", message=f"Model loading failed: {exc}")
        return self.state

    def predict_probability(self, rgb: np.ndarray) -> np.ndarray:
        if not self.state.ready or self.model is None:
            raise RuntimeError(self.state.message)

        config = self.state.config
        patch_size = int(config.get("patch_size", 256))
        overlap = int(config.get("overlap", 64))
        batch_size = int(config.get("inference_batch_size", 8))
        height, width = rgb.shape[:2]
        padded_height = max(height, patch_size)
        padded_width = max(width, patch_size)
        padded = np.pad(
            rgb,
            ((0, padded_height - height), (0, padded_width - width), (0, 0)),
            mode="reflect" if height > 1 and width > 1 else "edge",
        )
        y_starts = _starts(padded_height, patch_size, overlap)
        x_starts = _starts(padded_width, patch_size, overlap)
        coordinates = [(y, x) for y in y_starts for x in x_starts]
        probability_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
        count = np.zeros((padded_height, padded_width), dtype=np.float32)

        with self._lock:
            for index in range(0, len(coordinates), batch_size):
                batch_coordinates = coordinates[index : index + batch_size]
                batch = np.stack(
                    [padded[y : y + patch_size, x : x + patch_size] for y, x in batch_coordinates]
                ).astype(np.float32) / 255.0
                predictions = self.model.predict(batch, verbose=0)[..., 0]
                for (y, x), prediction in zip(batch_coordinates, predictions, strict=True):
                    probability_sum[y : y + patch_size, x : x + patch_size] += prediction
                    count[y : y + patch_size, x : x + patch_size] += 1.0

        return (probability_sum / np.maximum(count, 1.0))[:height, :width]

