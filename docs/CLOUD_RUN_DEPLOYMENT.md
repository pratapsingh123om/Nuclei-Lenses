# Deploy Nuclei Lens with Cloud Run, private GCS, and Vercel

This is the recommended production-demo architecture:

```text
GitHub repository ──source build──► Google Cloud Run FastAPI
                                           │ service identity
                                           ▼
                                  private Cloud Storage model

Vercel React frontend ──HTTPS──► public Cloud Run API
```

The model object remains private. Cloud Run downloads
`gs://nuclei-lenses/nuclei_lens_render_model.zip` using a dedicated service
account with only `roles/storage.objectViewer` on that bucket.

## Prerequisites

- A Google Cloud project with billing enabled.
- The verified model bundle already uploaded to the `nuclei-lenses` bucket.
- A deployed Vercel frontend origin, such as
  `https://nuclei-lenses.vercel.app`.
- Permission to create service accounts, grant bucket IAM, enable APIs, and
  deploy Cloud Run from source.

The expected model SHA-256 is:

```text
6273a494092fced07eb355010f231f6eab7d07b8116630c7841699914231c27a
```

## Deploy from Google Cloud Shell

Open Cloud Shell and run:

```bash
git clone https://github.com/pratapsingh123om/Nuclei-Lenses.git
cd Nuclei-Lenses

export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
export VERCEL_ORIGIN="https://YOUR-PROJECT.vercel.app"

bash scripts/deploy_cloud_run.sh
```

The script:

1. enables Cloud Run, Cloud Build, Artifact Registry, IAM, and Storage APIs;
2. creates a dedicated `nuclei-lens-api` runtime service account when absent;
3. grants that account read-only access to the model bucket;
4. builds `apps/backend/Dockerfile` from source;
5. deploys a public API with 2 vCPU, 2 GiB RAM, concurrency one, and a maximum
   of one instance;
6. downloads and checksum-verifies the private model at container startup;
7. prints the Cloud Run URL and calls `/api/health`.

The bucket URL returning HTTP 403 in an ordinary browser is expected because
the object remains private. Cloud Run authenticates through its service identity.

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

The response must show `"ready": true`. Then upload a small H&E image through
the Vercel interface and confirm that the overlay, mask, instance view, and
predicted count all appear.

For logs:

```bash
gcloud run services logs read nuclei-lens-api \
  --region asia-south1 \
  --limit 100
```

Cold starts include downloading the 81.9 MB model bundle and loading TensorFlow.
Keeping `min-instances=0` limits idle cost; increasing it to one improves first
request latency but incurs continuous compute charges.

