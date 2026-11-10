"""Perform structural and Python-syntax checks on the generated notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "monuseg_unet_colab.ipynb"


def main() -> None:
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 20
    joined = "".join("".join(cell.get("source", [])) for cell in data["cells"])
    required = (
        "canonical_30",
        "best_unet.weights.h5",
        "frozen_config.json",
        "overlapping_predict",
        "watershed_instances",
        "aggregated_jaccard_index",
        "per_image_metrics.csv",
        "summary_metrics.csv",
    )
    for token in required:
        assert token in joined, f"Notebook is missing required token: {token}"
    for index, cell in enumerate(data["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            compile(source, f"cell-{index}", "exec")
            assert not cell.get("outputs"), f"Cell {index} contains saved output"
    print(f"Validated {NOTEBOOK.name}: {len(data['cells'])} cells")


if __name__ == "__main__":
    main()

