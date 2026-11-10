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

| Measure | Watershed v1 | Component-aware v2 |
|---|---:|---:|
| Mean AJI | 0.6333 | **0.6431** |
| Count MAE | 114.36 | **92.07 nuclei/image** |
| Median absolute count error | 102.5 | **78.5 nuclei/image** |
| Mean per-image relative count MAE | 25.26% | **20.27%** |
| Signed count bias | +114.36 | **+92.07 nuclei/image** |

V2 settings were selected on the six validation images without using benchmark
images, then evaluated once on all 14 test images. It reduced count MAE by
19.5% while preserving 0.8113 Dice and 0.6837 IoU. A v2 AJI confidence interval
has not been calculated and is therefore not claimed. The positive bias still
equals the MAE, indicating systematic over-counting remains a limitation.

## Reproducibility check

The saved checkpoint was loaded into a fresh model and reproduced a test-image
Dice of 0.7942 within the required tolerance. The complete machine-readable
summary is in [`benchmark_summary.json`](benchmark_summary.json).

Dataset archives, prepared images, patient data, and `best_unet.weights.h5` are
not committed to Git. They remain in the configured private Drive artifact
directory.

