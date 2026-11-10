# Phase One — MoNuSeg data inspection

This phase answers whether the supplied data is valid and what cohorts it
contains. It deliberately performs no extraction, mask generation, patching,
splitting, augmentation, or model training.

## Run in Colab

For the no-upload workflow, open `phase1_colab_cell.py`, copy the entire file
into one Colab cell, and run it. It mounts Drive and reads both ZIP archives in
place.

The reusable command-line version can also be run after cloning the repository:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!python inspection.py \
  --train-archive "/content/drive/MyDrive/Nuclei-lens/MoNuSeg 2018 Training Data.zip" \
  --test-archive "/content/drive/MyDrive/Nuclei-lens/MoNuSegTestData.zip" \
  --output-dir "/content/drive/MyDrive/Nuclei-lens/phase1_inspection"
```

The script reads the ZIP files in place. It does not extract them.

## Outputs

- `phase1_report.json` — complete machine-readable verdict and archive hashes.
- `PHASE1_REPORT.md` — concise review report and decision gate.
- `phase1_pairs.csv` — one row for every paired image and XML annotation.
- `phase1_issues.csv` — errors and warnings with affected image IDs.
- `canonical_30_manifest.csv` — published 30-image training cohort.
- `extended_37_manifest.csv` — every valid pair in the training archive.
- `benchmark_14_manifest.csv` — held-out benchmark test cohort.

## Acceptance gate

Do not design the training notebook until:

1. Data integrity is `PASS`.
2. Canonical benchmark readiness is `READY`.
3. The additional seven training IDs have been reviewed.
4. The report records `canonical_30`: use 24 train, 6 validation, and reserve
   the 14 test cases. The seven additional pairs remain listed but excluded.

Finite polygon coordinates outside the image canvas are warnings, not archive
failures. The mask builder clips them to `[0, width - 1]` and
`[0, height - 1]`. Unreadable images/XML, broken pairing, a missing canonical
ID, train/test overlap, or an annotation with no usable polygons still fails
the gate.

## Cohort references

- The MoNuSeg challenge publication defines a 30-image training cohort and a
  14-image test cohort:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC10439521/>
- The canonical 30 TCGA identifiers and organ grouping used by the inspector are
  independently exposed in the `torch-em` MoNuSeg dataset adapter:
  <https://constantinpape.github.io/torch-em/torch_em/data/datasets/histopathology/monuseg.html>

The archive fingerprint and every selected/excluded ID are stored in the output
report, so this cohort rule is auditable rather than dependent on ZIP ordering.


