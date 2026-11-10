"""Build the Colab notebook from its readable percent-format source."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "monuseg_unet_colab.py"
TARGET = ROOT / "notebooks" / "monuseg_unet_colab.ipynb"


def markdown_source(lines: list[str]) -> str:
    converted: list[str] = []
    for line in lines:
        if line.startswith("# "):
            converted.append(line[2:])
        elif line.rstrip() == "#":
            converted.append("")
        else:
            converted.append(line)
    return "\n".join(converted).rstrip() + "\n"


def build() -> dict:
    cells: list[dict] = []
    kind: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if kind is None:
            return
        source = markdown_source(buffer) if kind == "markdown" else "\n".join(buffer).rstrip() + "\n"
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
        buffer = []

    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line == "# %% [markdown]":
            flush()
            kind = "markdown"
        elif line == "# %%":
            flush()
            kind = "code"
        else:
            buffer.append(line)
    flush()

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "monuseg_unet_colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return notebook


if __name__ == "__main__":
    notebook = build()
    print(f"Wrote {TARGET} with {len(notebook['cells'])} cells")

