# Deploy the inference API on Render

The repository includes a root-level [`render.yaml`](../render.yaml) Blueprint.
It builds `apps/backend/Dockerfile`, exposes `/api/health`, and attaches a
persistent 1 GB disk at `/app/model` for the trained artifacts.

The Blueprint selects Render's `1c-2g` plan. TensorFlow is not expected to run
reliably in a 512 MB instance. Inference batch size is capped at one patch to
keep peak CPU memory lower; this changes throughput, not prediction values.

## 1. Create the service

1. In Render, choose **New → Blueprint**.
2. Connect `pratapsingh123om/Nuclei-Lenses` and select `main`.
3. Keep the Blueprint path as `render.yaml`.
4. Set `NUCLEI_CORS_ORIGINS` to the final Vercel origin, for example
   `https://nuclei-lenses.vercel.app`.
5. Apply the Blueprint.

The first deployment can start in `setup_required` mode because trained model
files are intentionally excluded from Git. The health route still responds so
the service can be provisioned.

## 2. Put the model on the persistent disk

Download these files from the private Drive artifact directory:

```text
best_unet.weights.h5
frozen_config.json
```

Open the service's Render **Connect** panel and use its SSH target. From the
directory containing the downloaded files, upload both to the mounted disk:

```bash
scp -s best_unet.weights.h5 frozen_config.json \
  YOUR_SERVICE@ssh.YOUR_REGION.render.com:/app/model/
```

Restart the Render service after the upload. Then verify:

```text
https://YOUR-SERVICE.onrender.com/api/health
```

The response must contain `"ready": true` before enabling uploads in the
frontend. If it reports `model_error`, inspect the Render logs for a checkpoint
architecture or memory mismatch.

## 3. Connect Vercel

Set this variable in the Vercel frontend project:

```text
VITE_API_URL=https://YOUR-SERVICE.onrender.com
```

Redeploy Vercel because Vite embeds this value at build time. Confirm that the
same Vercel origin is present in `NUCLEI_CORS_ORIGINS` on Render without a
trailing slash.

## 4. Smoke test

1. Open `/api/health` and confirm model readiness.
2. Open the Vercel app and upload a small H&E field.
3. Confirm the response includes all four visualizations and a predicted count.
4. Check Render logs for HTTP 200 and confirm no worker restart or memory kill.

The persistent disk is attached to one service instance. This demo therefore
uses a single API instance and is intended for portfolio/research traffic, not
clinical workloads.

