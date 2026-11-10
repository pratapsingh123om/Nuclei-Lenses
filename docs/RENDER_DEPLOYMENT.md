# Deploy the inference API on Render

The repository includes a root-level [`render.yaml`](../render.yaml) Blueprint.
It builds `apps/backend/Dockerfile`, exposes `/api/health`, and downloads the
trained model bundle from cloud object storage into ephemeral runtime storage.
No Render persistent disk is required.

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

## 2. Upload the model bundle to cloud storage

Create or download the bundle from the private Drive artifact directory:

```text
nuclei_lens_render_model.zip
nuclei_lens_render_model.zip.sha256
```

Upload `nuclei_lens_render_model.zip` to your cloud bucket. The object must have
a stable HTTPS download URL that Render can access without an interactive login.
For a private bucket, use an appropriately scoped URL that does not expire while
the service is expected to restart.

```text
NUCLEI_MODEL_BUNDLE_URL=https://YOUR-BUCKET/nuclei_lens_render_model.zip
NUCLEI_MODEL_BUNDLE_SHA256=checksum-from-the-sha256-file
```

Set both values in Render and redeploy. Then verify:

```text
https://YOUR-SERVICE.onrender.com/api/health
```

The response must contain `"ready": true` before enabling uploads in the
frontend. If it reports `model_error`, inspect the Render logs for a checkpoint
architecture or memory mismatch.

On startup, the container downloads the bundle over HTTPS, verifies its SHA-256
checksum and required configuration keys, and installs both files in
`/tmp/nuclei-model` before FastAPI starts. Render's filesystem is ephemeral, so
the bundle is downloaded again after a new instance or restart.

If training is already complete and you do not want to reopen the notebook, run
the standalone `scripts/package_model_bundle.py` script. With its default Drive
layout it needs no arguments and performs no training.

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

This service is intended for research traffic, not clinical workloads.

