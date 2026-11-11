from __future__ import annotations

import base64
import json
import math
import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image

from .model import _starts


@dataclass
class MetastasisState:
    ready: bool = False
    status: str = "setup_required"
    message: str = "Add the evaluated PCam model bundle to enable metastatic-pattern mapping."
    config: dict[str, Any] = field(default_factory=dict)


def _data_uri(array: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def calibrate_probabilities(raw: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(raw, dtype=np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) / max(float(temperature), 1e-6)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def render_probability_outputs(
    rgb: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, str]:
    score = np.clip(probability, 0.0, 1.0)
    # A compact perceptual ramp: deep navy -> teal -> amber -> coral.
    red = np.clip(255 * (1.65 * score - 0.35), 8, 255)
    green = np.clip(255 * (1.45 - np.abs(score - 0.55) * 2.0), 25, 230)
    blue = np.clip(255 * (1.15 - 1.35 * score), 28, 230)
    heatmap = np.stack((red, green, blue), axis=-1).astype(np.uint8)

    alpha = (0.16 + 0.52 * score)[..., None]
    overlay = np.clip(rgb.astype(np.float32) * (1.0 - alpha) + heatmap.astype(np.float32) * alpha, 0, 255)

    evidence = np.zeros_like(rgb, dtype=np.uint8)
    evidence[:] = (5, 18, 17)
    evidence[score >= threshold] = (255, 112, 92)

    return {
        "original": _data_uri(rgb),
        "metastasis_overlay": _data_uri(overlay),
        "metastasis_heatmap": _data_uri(heatmap),
        "evidence_map": _data_uri(evidence),
    }


class MetastasisService:
    def __init__(self, model_dir: str | Path | None = None):
        configured = model_dir or os.getenv("PCAM_MODEL_DIR")
        self.model_dir = Path(configured) if configured else Path(__file__).resolve().parents[1] / "pcam_model"
        self.config_path = self.model_dir / "frozen_pcam_v1.json"
        self.model = None
        self.state = MetastasisState()
        self._lock = Lock()

    def load(self) -> MetastasisState:
        if not self.config_path.exists():
            self.state = MetastasisState(message=f"Missing model artifact: {self.config_path.name}.")
            return self.state

        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            model_name = str(config.get("model", {}).get("file", "nuclei_lens_pcam_v1.keras"))
            model_path = self.model_dir / model_name
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model artifact: {model_name}")

            import tensorflow as tf

            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.state = MetastasisState(
                ready=True,
                status="ready",
                message="PCam metastatic-pattern model loaded and ready for inference.",
                config=config,
            )
        except Exception as exc:
            self.model = None
            self.state = MetastasisState(status="model_error", message=f"PCam model loading failed: {exc}")
        return self.state

    def health_model(self) -> dict[str, Any]:
        config = self.state.config
        output = config.get("output", {})
        evaluation = config.get("evaluation", {})
        return {
            "ready": self.state.ready,
            "status": self.state.status,
            "message": self.state.message,
            "version": config.get("model_id", "pcam_v1"),
            "architecture": config.get("model", {}).get("architecture", "EfficientNetV2-B0"),
            "patch_size": int(config.get("input", {}).get("shape", [96, 96, 3])[0]),
            "analysis_endpoint": "/api/metastasis/analyze",
            "temperature": output.get("temperature"),
            "thresholds": output.get("thresholds", {}),
            "benchmark": {
                "test_patches": evaluation.get("test_patches"),
                "roc_auc": evaluation.get("roc_auc"),
                "pr_auc": evaluation.get("pr_auc"),
            },
            "capabilities": {
                "patch_classification": True,
                "probability_heatmap": True,
                "sensitivity_focused_threshold": True,
            },
        }

    def analyze(self, rgb: np.ndarray, operating_point: str) -> dict[str, Any]:
        if not self.state.ready or self.model is None:
            raise RuntimeError(self.state.message)

        config = self.state.config
        input_shape = config.get("input", {}).get("shape", [96, 96, 3])
        patch_size = int(input_shape[0])
        stride = max(8, int(os.getenv("PCAM_INFERENCE_STRIDE", "32")))
        batch_size = max(1, int(os.getenv("PCAM_INFERENCE_BATCH_SIZE", "32")))
        maximum_patches = max(1, int(os.getenv("PCAM_MAX_PATCHES", "2500")))
        thresholds = config.get("output", {}).get("thresholds", {})
        threshold_key = "sensitivity_focused" if operating_point == "sensitivity" else "balanced_youden"
        threshold = float(thresholds.get(threshold_key, 0.5))
        temperature = float(config.get("output", {}).get("temperature", 1.0))

        height, width = rgb.shape[:2]
        padded_height = max(height, patch_size)
        padded_width = max(width, patch_size)
        padded = np.pad(
            rgb,
            ((0, padded_height - height), (0, padded_width - width), (0, 0)),
            mode="reflect" if height > 1 and width > 1 else "edge",
        )
        y_starts = _starts(padded_height, patch_size, patch_size - stride)
        x_starts = _starts(padded_width, patch_size, patch_size - stride)
        coordinates = [(y, x) for y in y_starts for x in x_starts]
        if len(coordinates) > maximum_patches:
            raise ValueError(
                f"This image requires {len(coordinates):,} PCam patches; the service limit is "
                f"{maximum_patches:,}. Crop a representative field without changing its scale."
            )

        raw_predictions: list[np.ndarray] = []
        with self._lock:
            for index in range(0, len(coordinates), batch_size):
                batch_coordinates = coordinates[index : index + batch_size]
                batch = np.stack(
                    [padded[y : y + patch_size, x : x + patch_size] for y, x in batch_coordinates]
                ).astype(np.float32)
                raw_predictions.append(self.model.predict(batch, verbose=0).reshape(-1))

        raw = np.concatenate(raw_predictions).astype(np.float32)
        probabilities = calibrate_probabilities(raw, temperature)
        probability_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
        coverage = np.zeros((padded_height, padded_width), dtype=np.float32)
        for (y, x), score in zip(coordinates, probabilities, strict=True):
            probability_sum[y : y + patch_size, x : x + patch_size] += float(score)
            coverage[y : y + patch_size, x : x + patch_size] += 1.0
        probability_map = probability_sum / np.maximum(coverage, 1.0)
        probability_map = probability_map[:height, :width]
        positives = probabilities >= threshold
        positive_fraction = float(positives.mean())
        safe_probabilities = np.clip(probabilities.astype(np.float64), 1e-6, 1 - 1e-6)
        normalized_entropy = -(
            safe_probabilities * np.log2(safe_probabilities)
            + (1.0 - safe_probabilities) * np.log2(1.0 - safe_probabilities)
        )
        uncertainty_percent = float(normalized_entropy.mean() * 100.0)
        near_threshold_percent = float((np.abs(probabilities - threshold) <= 0.08).mean() * 100.0)

        minimum_cluster = max(3, math.ceil(len(probabilities) * 0.03))
        if not positives.any():
            finding_code = "not_detected"
            finding_title = "No metastatic tissue pattern detected in this field"
            finding_summary = (
                "None of the analyzed regions crossed the model's review threshold. "
                "This lowers concern for this field, but it does not exclude metastasis elsewhere on a slide."
            )
            evidence_strength = "strong" if float(probabilities.max()) <= threshold - 0.15 else "moderate"
            next_step = (
                "If this image is part of a clinical case, review additional fields or the complete slide "
                "when suspicion remains. A single negative field cannot rule out lymph-node metastasis."
            )
        elif int(positives.sum()) < minimum_cluster:
            finding_code = "limited_evidence"
            finding_title = "Limited metastatic tissue pattern detected"
            finding_summary = (
                f"{int(positives.sum())} of {len(probabilities)} analyzed regions crossed the review threshold. "
                "The evidence is localized and may represent true tumour tissue or a visual mimic."
            )
            evidence_strength = "limited"
            next_step = (
                "Review the highlighted regions first and confirm them against the original H&E image. "
                "A qualified pathologist should decide whether the morphology represents metastasis."
            )
        else:
            finding_code = "detected"
            finding_title = "Metastatic tissue pattern detected"
            finding_summary = (
                f"{int(positives.sum())} of {len(probabilities)} analyzed regions matched patterns learned "
                "from metastatic breast-cancer tissue in sentinel lymph nodes."
            )
            positive_mean = float(probabilities[positives].mean())
            evidence_strength = (
                "strong"
                if positive_fraction >= 0.10 and positive_mean >= threshold + 0.15
                else "moderate"
            )
            next_step = (
                "Prioritize the highlighted regions for pathologist review and confirmation on the full "
                "sentinel-lymph-node slide before any diagnosis or staging decision."
            )

        evidence_label = finding_title

        return {
            "patch_size": patch_size,
            "stride": stride,
            "patch_count": int(len(probabilities)),
            "positive_patch_count": int(positives.sum()),
            "positive_patch_percent": round(positive_fraction * 100.0, 2),
            "mean_probability": round(float(probabilities.mean()), 4),
            "max_probability": round(float(probabilities.max()), 4),
            "heatmap_area_percent": round(float((probability_map >= threshold).mean() * 100.0), 2),
            "threshold": threshold,
            "temperature": temperature,
            "operating_point": operating_point,
            "evidence_label": evidence_label,
            "diagnostic_support": {
                "finding_code": finding_code,
                "finding_title": finding_title,
                "finding_summary": finding_summary,
                "evidence_strength": evidence_strength,
                "uncertainty_percent": round(uncertainty_percent, 1),
                "near_threshold_percent": round(near_threshold_percent, 1),
                "next_step": next_step,
                "uncertainty_note": (
                    "Uncertainty increases with borderline probabilities and with differences in stain, "
                    "focus, magnification, tissue preparation, or tissue types not represented by PCam."
                ),
                "clinical_context": (
                    "Supports review for breast-cancer metastasis in sentinel lymph-node H&E tissue. "
                    "It does not establish patient-level lymph-node status."
                ),
            },
            "visuals": render_probability_outputs(rgb, probability_map, threshold),
        }

