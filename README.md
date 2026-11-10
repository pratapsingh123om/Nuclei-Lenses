# Nuclei Lens

Nuclei Lens is a research application for segmenting and counting nuclei in H&E pathology images. A TensorFlow U-Net predicts nuclear pixels, overlapping patch inference reconstructs the full-resolution mask, and distance-transform watershed separates touching nuclei into countable instances.

> Research use only. Nuclei Lens is not validated for diagnosis, treatment, or clinical decision-making.

## What the app provides

- Drag-and-drop upload for PNG, JPEG, TIFF, and WebP tissue images.
- Semantic nuclei masks from a small, explainable 2D U-Net.
- Full-image inference using overlapping 256 × 256 patches with probability averaging.
- Instance separation and nuclei counting using distance-transform watershed.
- Four synchronized views: original image, binary mask, color overlay, and instance map.
- Readiness checks that disable inference until a real trained checkpoint is available.
- Input safeguards for MIME type, upload size, decoded image size, and model availability.
- A root-level Vercel configuration for the React frontend and a Dockerfile for the TensorFlow API.

## Held-out results

Using the canonical 30-image MoNuSeg development cohort and all 14 held-out
benchmark images, the U-Net achieved **0.8113 mean Dice** and **0.6837 mean
IoU**, compared with 0.7603 Dice and 0.6147 IoU for the color/morphology
baseline. It improved Dice on all 14 test images; the mean paired gain was
0.0510 (95% bootstrap CI 0.0388–0.0638).

Watershed instance separation achieved **0.6333 mean AJI**. Counting remains a
documented limitation: 25.26% mean relative count MAE and a +114.36
nuclei/image bias indicate systematic over-separation. See the complete
[`results/` report](results/README.md) for confidence intervals, the checkpoint
reload test, and failure analysis.

## Architecture

```text
Browser
  │
  │  POST /api/segment
  ▼
React + Vite frontend ───────────────► FastAPI inference service
                                         │
                                         ├─ overlapping U-Net prediction
                                         ├─ frozen validation threshold
                                         ├─ morphology cleanup
                                         └─ watershed instance separation
                                                      │
                                                      ▼
                                  mask · overlay · instances · count
```

The frontend and backend are deployed separately. Vercel serves the static React application. The FastAPI service runs in a container with sufficient memory for TensorFlow and the trained model.

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

The API intentionally starts in `setup_required` mode until these files are placed in `apps/backend/model/`:

```text
best_unet.weights.h5
frozen_config.json
```

Model binaries are ignored by Git. This prevents a deployment from appearing to return trained predictions when no validated checkpoint is present.

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

## API

| Method | Route | Description |
|---|---|---|
| `GET` | `/api/health` | Reports API, model, patch-size, and threshold readiness. |
| `POST` | `/api/segment` | Accepts an image and returns visualizations, instance count, nuclear area, threshold, and processing time. |

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

### Render Blueprint

For the deployed demo, use the tracked [`render.yaml`](render.yaml). It creates
a Docker web service with 2 GB RAM, a persistent model disk, a health check, and
a memory-conscious inference batch size. Follow the complete
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

The model is designed for the original MoNuSeg 2018 benchmark. Keep the original per-nucleus annotations so semantic segmentation and instance counting can be evaluated separately.

Recommended held-out reporting:

- mean per-image Dice and IoU for binary masks;
- Aggregated Jaccard Index (AJI) for instances;
- nuclei count mean absolute error;
- per-image results and representative strong, typical, and failure overlays.

Segmentation determines which pixels belong to nuclei. Instance segmentation assigns those pixels to individual nuclei. Touching nuclei may produce a good semantic mask while still being undercounted, which is why the app keeps the two stages explicit.

## Project description

**Short description for GitHub or Vercel:**

> U-Net research app for H&E nuclei segmentation, watershed instance separation, visual overlays, and nuclei counting.
