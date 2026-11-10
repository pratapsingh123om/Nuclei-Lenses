# Held-out benchmark results

These results come from the completed canonical MoNuSeg run. The 30 published
training images were split into 24 development and 6 validation images before
patch sampling. All settings were selected on validation data and frozen before
evaluating the 14 benchmark test images.

## Semantic segmentation

| Method | Mean Dice | Mean IoU |
|---|---:|---:|
| Color/morphology baseline | 0.7603 | 0.6147 |
| U-Net | **0.8113** | **0.6837** |

- U-Net Dice 95% bootstrap CI: 0.7950–0.8277.
- Mean paired Dice improvement: +0.0510 (95% bootstrap CI 0.0388–0.0638).
- U-Net exceeded the baseline on all 14 test images.
- Paired Wilcoxon p-value: 0.000122.

## Instance segmentation and counting

| Measure | Result |
|---|---:|
| Mean AJI | 0.6333 |
| AJI 95% bootstrap CI | 0.6097–0.6574 |
| Count MAE | 114.36 nuclei/image |
| Median absolute count error | 102.5 nuclei/image |
| Mean per-image relative count MAE | 25.26% |
| Signed count bias | +114.36 nuclei/image |

The positive bias equals the MAE, indicating systematic over-counting in this
test run. The likely failure mode is watershed over-separation: one biological
nucleus is sometimes divided into multiple predicted instances. This is an
explicit limitation, not a tuned-away test result. Future work should tune or
learn object centers on validation data and inspect stain- and organ-specific
count errors.

## Reproducibility check

The saved checkpoint was loaded into a fresh model and reproduced a test-image
Dice of 0.7942 within the required tolerance. The complete machine-readable
summary is in [`benchmark_summary.json`](benchmark_summary.json).

Dataset archives, prepared images, patient data, and `best_unet.weights.h5` are
not committed to Git. They remain in the configured private Drive artifact
directory.

