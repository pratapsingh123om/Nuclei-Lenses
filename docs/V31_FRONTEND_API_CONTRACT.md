# v3.1 frontend API contract

The frontend remains compatible with the existing v2 response. When the
PanNuke v3.1 model is deployed, the backend enables the typed-nuclei UI by
adding the fields below. Existing fields remain available.

## `GET /api/health`

```json
{
  "service": "nuclei-lens-api",
  "ready": true,
  "status": "ready",
  "message": "PanNuke v3.1 model loaded.",
  "model": {
    "version": "pannuke-v3.1",
    "architecture": "EfficientNetV2 HoVer U-Net",
    "patch_size": 256,
    "instance_pipeline": "hover_v31",
    "analysis_endpoint": "/api/analyze-v3",
    "outputs": ["nucleus", "hover", "type"],
    "class_names": ["neoplastic", "inflammatory", "connective", "dead", "non_neoplastic_epithelial"],
    "capabilities": { "typed_nuclei": true, "typed_overlay": true, "csv_counts": true },
    "benchmark": { "count_relative_mae_percent": null, "bPQ": null, "mPQ": null }
  }
}
```

Until the v3.1 endpoint is available, omit `analysis_endpoint` and the
frontend continues posting to `/api/segment`.

## `POST /api/analyze-v3`

The request remains multipart form data with one field named `image`.

```json
{
  "filename": "field.tif",
  "model_version": "pannuke-v3.1",
  "width": 1000,
  "height": 1000,
  "predicted_count": 533,
  "nuclei_area_percent": 23.97,
  "inference_ms": 13116.5,
  "type_counts": {
    "neoplastic": 211,
    "inflammatory": 96,
    "connective": 122,
    "dead": 17,
    "non_neoplastic_epithelial": 87
  },
  "visuals": {
    "typed_overlay": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,...",
    "original": "data:image/png;base64,...",
    "mask": "data:image/png;base64,...",
    "instances": "data:image/png;base64,...",
    "types": "data:image/png;base64,..."
  },
  "notice": "Research use only. Not for clinical diagnosis."
}
```

The frontend also accepts `class_counts`, `counts_by_type`, or an array named
`cell_types` as aliases for `type_counts`. `typed_overlay` is preferred, while
`type_overlay` is accepted as an alias.

## UI behavior

- v2 health and inference payloads retain the current mask/count interface.
- A five-class health response enables PanNuke and HoVer model copy.
- Typed results add the typed overlay, composition bar, class counts,
  field-level neoplastic-to-inflammatory ratio, and CSV download.
- Cell-type outputs are visual phenotype estimates, never a diagnosis, tumour
  grade, prognosis, or treatment recommendation.
