#!/usr/bin/env bash
set -euo pipefail

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT to your Google Cloud project ID}"
: "${VERCEL_ORIGIN:?Set VERCEL_ORIGIN to the deployed frontend origin, including https://}"

REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-nuclei-lens-api}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-nuclei-lens-api}"
MODEL_BUCKET="${MODEL_BUCKET:-nuclei-lenses}"
MODEL_OBJECT="${MODEL_OBJECT:-nuclei_lens_render_model.zip}"
MODEL_SHA256="${MODEL_SHA256:-6273a494092fced07eb355010f231f6eab7d07b8116630c7841699914231c27a}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com"

gcloud config set project "${GOOGLE_CLOUD_PROJECT}"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  storage.googleapis.com

if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --display-name="Nuclei Lens Cloud Run runtime"
fi

gcloud storage buckets add-iam-policy-binding "gs://${MODEL_BUCKET}" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectViewer"

gcloud run deploy "${SERVICE}" \
  --source apps/backend \
  --region "${REGION}" \
  --service-account "${SERVICE_ACCOUNT}" \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 2Gi \
  --concurrency 1 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 1 \
  --set-env-vars="NUCLEI_MODEL_DIR=/tmp/nuclei-model,NUCLEI_MODEL_BUNDLE_GCS_URI=gs://${MODEL_BUCKET}/${MODEL_OBJECT},NUCLEI_MODEL_BUNDLE_SHA256=${MODEL_SHA256},NUCLEI_MODEL_BUNDLE_MAX_MB=1024,NUCLEI_INFERENCE_BATCH_SIZE=1,NUCLEI_CORS_ORIGINS=${VERCEL_ORIGIN},NUCLEI_MAX_UPLOAD_MB=20,NUCLEI_MAX_IMAGE_PIXELS=40000000"

SERVICE_URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"
echo "Cloud Run URL: ${SERVICE_URL}"
echo "Set VITE_API_URL=${SERVICE_URL} in Vercel and redeploy the frontend."
curl --fail --show-error --silent "${SERVICE_URL}/api/health"
echo

