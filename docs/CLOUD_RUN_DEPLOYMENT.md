# Deploy Nuclei Lens with Cloud Run, private GCS, and Vercel

This is the recommended production-demo architecture:

```text
GitHub repository ──source build──► Google Cloud Run FastAPI
                                           │ service identity
                                           ▼
                                  private Cloud Storage model bundles

Vercel React frontend ──HTTPS──► public Cloud Run API
```

The model object remains private. Cloud Run downloads the explicitly selected
bundle using a dedicated service account with only `roles/storage.objectViewer`
on that bucket.

## Prerequisites

- A Google Cloud project with billing enabled.
- The verified nuclei and PCam model bundles already uploaded to the `nuclei-lenses` bucket.
- A deployed Vercel frontend origin, such as
  `https://nuclei-lenses.vercel.app`.
- Permission to create service accounts, grant bucket IAM, enable APIs, and
  deploy Cloud Run from source.

## Deploy from Google Cloud Shell

Open Cloud Shell and run:

```bash
git clone https://github.com/pratapsingh123om/Nuclei-Lenses.git
cd Nuclei-Lenses

export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
export VERCEL_ORIGIN="https://YOUR-PROJECT.vercel.app"
export MODEL_OBJECT="nuclei_lens_cloudrun_model_v2.zip"
export PCAM_MODEL_OBJECT="nuclei_lens_pcam_cloudrun_v1_1e242a26645d.zip"

# Compute the exact hexadecimal hashes from the private bucket objects.
export MODEL_SHA256="$(gcloud storage cat gs://nuclei-lenses/${MODEL_OBJECT} | sha256sum | cut -d' ' -f1)"
export PCAM_MODEL_SHA256="$(gcloud storage cat gs://nuclei-lenses/${PCAM_MODEL_OBJECT} | sha256sum | cut -d' ' -f1)"

bash scripts/deploy_cloud_run.sh
```

All four model variables are required. This prevents an old object or checksum
from being selected silently during a later deployment. The PCam bundle is
locked to checkpoint `1e242a26645dfa982d88b369dc277f5f0d0fa8ca0e9ed1a0589aa729da1ac740`,
which achieved 0.9350 ROC-AUC and 0.9376 PR-AUC on the official 32,768-patch
test split.

The script:

1. enables Cloud Run, Cloud Build, Artifact Registry, IAM, and Storage APIs;
2. creates a dedicated `nuclei-lens-api` runtime service account when absent;
3. grants the default build identity the Cloud Run Builder role;
4. grants the runtime account read-only access to the model bucket;
5. builds `apps/backend/Dockerfile` from source;
6. deploys a public API with 2 vCPU, 4 GiB RAM, concurrency one, and a maximum
   of one instance;
7. downloads and checksum-verifies both private model bundles at container startup;
8. prints the Cloud Run URL and calls `/api/health`.

The bucket URLs returning HTTP 403 in an ordinary browser are expected because
the objects remain private. Cloud Run authenticates through its service identity.

## Connect Vercel

Copy the printed Cloud Run URL into the Vercel project environment:

```text
VITE_API_URL=https://YOUR-CLOUD-RUN-SERVICE.run.app
```

Redeploy Vercel. The deployment script sets `NUCLEI_CORS_ORIGINS` to the exact
`VERCEL_ORIGIN` supplied during deployment.

## Verify

Open:

```text
https://YOUR-CLOUD-RUN-SERVICE.run.app/api/health
```

The response must show both `models.nuclei.ready` and
`models.metastasis.ready` as `true`. Verify nuclei segmentation first, then switch
the frontend to **Metastatic pattern** and confirm that the probability overlay,
heatmap, threshold map, patch counts, and clinical-context explanation appear.

For logs:

```bash
gcloud run services logs read nuclei-lens-api \
  --region asia-south1 \
  --limit 100
```

Cold starts include downloading both model bundles and loading two TensorFlow models.
Keeping `min-instances=0` limits idle cost; increasing it to one improves first
request latency but incurs continuous compute charges.

