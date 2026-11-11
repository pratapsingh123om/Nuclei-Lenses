# Nuclei Lens

Nuclei Lens is a research application for quantifying H&E pathology images. Its nuclei workflow segments and counts individual nuclei; its PCam workflow classifies overlapping lymph-node patches and maps metastatic-tissue probabilities back to their source coordinates for visual review.

> Research use only. Nuclei Lens is not validated for diagnosis, treatment, or clinical decision-making.

## What the app provides

- Drag-and-drop upload for PNG, JPEG, TIFF, and WebP tissue images.
- Semantic nuclei masks from a small, explainable 2D U-Net.
- Full-image inference using overlapping 256 × 256 patches with probability averaging.
- Instance separation and nuclei counting using distance-transform watershed.
- Four synchronized views: original image, binary mask, color overlay, and instance map.
- Calibrated PCam metastatic-pattern mapping with probability, threshold, and overlay views.
- A plain-language preliminary finding—detected, limited evidence, or not detected in this field—with evidence strength, uncertainty, and the recommended review step.
- Balanced and sensitivity-focused operating points frozen on validation data.
- A diagnostic-context panel explaining how patch localization can support breast-cancer sentinel-lymph-node review—and what the model cannot diagnose.
- Readiness checks that disable inference until a real trained checkpoint is available.
- Input safeguards for MIME type, upload size, decoded image size, and model availability.
- A root-level Vercel configuration for the React frontend and a Dockerfile for the TensorFlow API.

## Held-out results

Using the canonical 30-image MoNuSeg development cohort and all 14 held-out
benchmark images, the U-Net achieved **0.8113 mean Dice** and **0.6837 mean
IoU**, compared with 0.7603 Dice and 0.6147 IoU for the color/morphology
baseline. It improved Dice on all 14 test images; the mean paired gain was
0.0510 (95% bootstrap CI 0.0388–0.0638).

Component-aware watershed instance separation achieved **0.6431 mean AJI** and
reduced count MAE from 114.36 to **92.07 nuclei/image** (a 19.5% reduction)
without changing semantic accuracy. Counting remains a documented limitation:
20.27% mean relative count MAE and a +92.07 nuclei/image bias still indicate
systematic over-separation. See the complete
[`results/` report](results/README.md) for confidence intervals, the checkpoint
reload test, and failure analysis.

The EfficientNetV2-B0 PCam classifier achieved **0.9350 ROC-AUC** (95% bootstrap
CI 0.9326–0.9373) and **0.9376 PR-AUC** on the official 32,768-patch held-out
test split. At its validation-frozen sensitivity-focused operating point it
reached **92.73% sensitivity**, 74.31% specificity, and 0.8490 F1. These are
patch-level metastatic-tissue results, not slide-level or patient-level diagnostic
performance.

## Architecture

```text
Browser
  │
  │  POST /api/segment
  ▼
React + Vite frontend ───────────────► FastAPI inference service
                                         │
                                         ├─ nuclei: U-Net + watershed
                                         │             └─ mask · instances · count
                                         │
                                         └─ PCam: EfficientNetV2 + calibration
                                                       └─ patch scores · heatmap · evidence map
```

The frontend and backend are deployed separately. Vercel serves the static
React application. Google Cloud Run hosts FastAPI, and its dedicated service
identity reads the private model bundle from Google Cloud Storage.

## Project structure

```text
apps/
  frontend/
    src/                 React analysis workspace
    public/              static brand assets
    .env.example         frontend API configuration
  backend/
    app/                 FastAPI routes, U-Net loader, image pipeline
    model/               local/deployed model artifacts (ignored by Git)
    tests/               inference-helper tests
    Dockerfile           container deployment
packages/                shared monorepo packages
notebooks/               generated Colab notebook and readable source
scripts/                 notebook build and structural validation
inspection.py            reusable archive inspection command
phase1_colab_cell.py     no-upload Phase One Colab cell
vercel.json              root Vercel build contract
turbo.json               monorepo task configuration
```

## Model artifacts

The nuclei API starts in `setup_required` mode until these files are placed in `apps/backend/model/`:

```text
best_unet.weights.h5
frozen_config.json
```

Model binaries are ignored by Git. This prevents a deployment from appearing to return trained predictions when no validated checkpoint is present.

The optional metastatic-pattern workflow reads a separately verified bundle into
`apps/backend/pcam_model/` containing:

```text
nuclei_lens_pcam_v1.keras
frozen_pcam_v1.json
manifest.json
```

The deployed PCam checkpoint is content-addressed by SHA-256
`1e242a26645dfa982d88b369dc277f5f0d0fa8ca0e9ed1a0589aa729da1ac740`.

The configuration can include:

```json
{
  "patch_size": 256,
  "overlap": 64,
  "base_filters": 32,
  "mask_threshold": 0.5,
  "min_object_size": 12,
  "watershed_min_distance": 5,
  "inference_batch_size": 8
}
```

Use the values frozen on validation data. Do not tune the threshold or watershed settings on the benchmark test set.

## Train and evaluate in Colab

1. Put the original MoNuSeg training and annotated test ZIP files in
   `/content/drive/MyDrive/Nuclei-lens/`.
2. Copy and run [`phase1_colab_cell.py`](phase1_colab_cell.py) in one Colab
   cell. It audits the archives without extracting or training and records the
   canonical 30-image protocol.
3. Continue only when the report says integrity `PASS` and benchmark `READY`.
4. Open [`notebooks/monuseg_unet_colab.ipynb`](notebooks/monuseg_unet_colab.ipynb)
   in a GPU Colab runtime and run it from top to bottom.

The notebook creates a grouped 24/6 split before patch sampling, runs the color
baseline, verifies a four-patch overfit check, trains the U-Net, selects its
checkpoint on reconstructed validation images, freezes semantic and watershed
settings, and evaluates all 14 test images. It writes the checkpoint and
`frozen_config.json` expected by the API.

### Tune instance separation without retraining

After the notebook has produced its checkpoint and split manifest, run
[`scripts/tune_instance_v2.py`](scripts/tune_instance_v2.py) to reduce
watershed over-separation. It keeps the semantic threshold fixed, estimates
nucleus size from the 24 training images, searches component-aware watershed
settings using only the six validation images, freezes the winning settings,
and then evaluates the 14 benchmark images. Existing v1 results are preserved
under `artifacts/`; v2 outputs are written to `artifacts/instance_v2/`.

```bash
python scripts/tune_instance_v2.py --search-profile balanced
```

Use `--validation-only` to freeze the configuration without evaluating the
benchmark images. Full-image U-Net probability maps are cached, so subsequent
searches do not repeat inference.

## Run locally

### Prerequisites

- Node.js 20.19 or newer
- Python 3.11
- A trained checkpoint and frozen configuration for real inference

Install the JavaScript workspace from the repository root:

```powershell
npm ci
```

Create the backend environment and install its dependencies:

```powershell
python -m venv apps\backend\.venv
apps\backend\.venv\Scripts\python -m pip install -r apps\backend\requirements.txt
```

Copy the model artifacts into `apps/backend/model/`, then start the API:

```powershell
apps\backend\.venv\Scripts\python -m uvicorn app.main:app --app-dir apps\backend --reload --port 8000
```

In another terminal, start the frontend:

```powershell
npm --workspace @nuclei-lens/frontend run dev
```

Open [http://localhost:5173](http://localhost:5173). API readiness is available at [http://localhost:8000/api/health](http://localhost:8000/api/health), and the OpenAPI interface is at [http://localhost:8000/docs](http://localhost:8000/docs).

## Configuration

Frontend:

| Variable | Purpose | Default |
|---|---|---|
| `VITE_API_URL` | Public origin of the FastAPI service | `http://127.0.0.1:8000` |

Backend:

| Variable | Purpose | Default |
|---|---|---|
| `NUCLEI_MODEL_DIR` | Directory containing both model artifacts | `apps/backend/model` |
| `NUCLEI_CORS_ORIGINS` | Comma-separated allowed frontend origins | `http://localhost:5173` |
| `NUCLEI_MAX_UPLOAD_MB` | Maximum uploaded file size | `20` |
| `NUCLEI_MAX_IMAGE_PIXELS` | Maximum decoded pixel count | `40000000` |
| `NUCLEI_INFERENCE_BATCH_SIZE` | Patch batch size; use `1` on memory-limited CPU hosting | value frozen by training |
| `PCAM_MODEL_DIR` | Directory containing the evaluated PCam model and configuration | `apps/backend/pcam_model` |
| `PCAM_INFERENCE_BATCH_SIZE` | Number of 96px patches classified per CPU batch | `32` |
| `PCAM_INFERENCE_STRIDE` | Pixel stride used to construct the probability map | `32` |
| `PCAM_MAX_PATCHES` | Maximum patches accepted per request | `2500` |

## API

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/health` | Reports API, model, patch-size, and threshold readiness. |
| `POST` | `/api/segment` | Accepts an image and returns visualizations, instance count, nuclear area, threshold, and processing time. |
| `POST` | `/api/metastasis/analyze` | Returns calibrated PCam patch evidence, probability heatmaps, frozen-threshold maps, and processing metadata. |

The API processes uploads in memory and does not persist tissue images.

## Deploy the frontend to Vercel

Import the GitHub repository into Vercel and leave **Root Directory** set to the repository root (`.`). The tracked `vercel.json` defines the complete build path:

```text
Install command:  npm ci
Build command:    npm --workspace @nuclei-lens/frontend run build
Output directory: apps/frontend/dist
```

Add this project environment variable in Vercel:

```text
VITE_API_URL=https://your-api.example.com
```

Use the backend origin without a trailing slash. Vite embeds this value during the build, so redeploy after changing it.

## Deploy the backend container

Place the two model artifacts in `apps/backend/model/`, then build from the repository root:

```powershell
docker build -t nuclei-lens-api apps/backend
docker run --rm -p 8000:8000 -e NUCLEI_CORS_ORIGINS=https://your-project.vercel.app nuclei-lens-api
```

Deploy the same image to a container platform with enough RAM for TensorFlow. Set `NUCLEI_CORS_ORIGINS` to the final Vercel origin and set `VITE_API_URL` in Vercel to the final HTTPS API origin.

The TensorFlow API is deliberately excluded from the static Vercel build because its runtime and model requirements are better suited to a container service.

### Google Cloud Run — recommended

The recommended backend uses Cloud Run with a dedicated service account and a
private Google Cloud Storage model object. No public model URL or persistent
filesystem is required. Run the tracked
[`Cloud Shell deployment script`](scripts/deploy_cloud_run.sh) or follow the
complete [`Cloud Run deployment guide`](docs/CLOUD_RUN_DEPLOYMENT.md). The
script builds the backend Dockerfile, grants bucket-level read-only access,
checksum-verifies the model, limits concurrency and scaling, and prints the URL
to place in Vercel as `VITE_API_URL`.

### Render Blueprint — legacy alternative

For the deployed demo, use the tracked [`render.yaml`](render.yaml). It creates
a Docker web service with 2 GB RAM, a health check, and a memory-conscious
inference batch size. No Render disk is required: the backend securely downloads
a checksum-verified model bundle from your cloud bucket on startup. Follow the complete
[`Render deployment guide`](docs/RENDER_DEPLOYMENT.md) to upload the checkpoint,
verify readiness, configure CORS, and connect the Vercel frontend.

## Validate

Build the production frontend:

```powershell
npm --workspace @nuclei-lens/frontend run build
```

Run the backend tests:

```powershell
python -m pytest apps/backend/tests -q
```

The test suite checks sliding-window edge coverage, safe behavior when model artifacts are absent, image decoding and mask cleanup, and preservation of separate watershed instance IDs.

## Dataset and evaluation

The nuclei model is designed for the original MoNuSeg 2018 benchmark. Keep the original per-nucleus annotations so semantic segmentation and instance counting can be evaluated separately. The metastatic-pattern classifier uses the official PatchCamelyon splits: 262,144 training, 32,768 validation, and 32,768 test patches drawn from lymph-node histopathology.

Evaluation outputs include:

- mean per-image Dice and IoU for binary masks;
- Aggregated Jaccard Index (AJI) for instances;
- nuclei count mean absolute error;
- per-image results and representative strong, typical, and failure overlays.

Segmentation determines which pixels belong to nuclei. Instance segmentation assigns those pixels to individual nuclei. Touching nuclei may produce a good semantic mask while still being undercounted, which is why the app keeps the two stages explicit.

PCam answers a different question: whether the central region of a 96 × 96
lymph-node patch contains metastatic tissue. Mapping overlapping patch scores can
help prioritize suspicious regions for pathologist review, but it does not confirm
or exclude breast cancer, assign lymph-node status, determine tumour stage or grade,
or recommend treatment.
