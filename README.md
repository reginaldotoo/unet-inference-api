# U-Net Satellite Inference API

Dockerized FastAPI service for crop classification from multi-temporal satellite imagery using a trained U-Net model.

Takes Sentinel-2 monthly composites as input, runs tile-based U-Net inference with Hann-window blending, and outputs classified GeoTIFFs.

## Architecture

```
┌────────────────┐       ┌──────────────────┐       ┌────────────────┐
│  Satellite     │       │  FastAPI          │       │  Classified    │
│  Composites    │──────▶│  /predict/{site}  │──────▶│  GeoTIFFs      │
│  (GeoTIFF)     │       │                  │       │  (raw + clean) │
└────────────────┘       │  U-Net model      │       └────────────────┘
                         │  tile-based       │
                         │  inference        │
                         └──────────────────┘
```

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/sites` | List available sites |
| GET | `/model-info` | Model architecture and classes |
| POST | `/predict/{site_name}` | Run inference on a site |

## Quick Start

### 1. Build

```bash
docker build -t unet-api .
```

### 2. Run

Mount your data directory containing the model, norm params, and site folders:

```bash
docker run -p 8000:8000 \
  -v /path/to/your/data:/work_beegfs/sungg1179 \
  unet-api
```

Your data directory should contain:

```
your-data/
├── unet_model.keras
├── unet_norm_params.json
├── Pembina/
│   ├── Pembina_composite_2024_5.tif
│   ├── ...
│   └── Pembina_composite_2024_9.tif
├── Sheridan/
├── Morris/
└── Rose/
```

### 3. Predict

```bash
curl http://localhost:8000/health
curl http://localhost:8000/sites
curl -X POST http://localhost:8000/predict/Pembina
```

## Output

Each prediction produces two GeoTIFFs per site:

- `UNet_{site}_Classified.tif` — raw pixel-wise classification
- `UNet_{site}_Classified_Clean.tif` — post-processed (small patches removed)

## Classes

| ID | Class |
|----|-------|
| 0 | NoData |
| 1 | Cereals |
| 2 | Canola |
| 3 | Soybeans |
| 4 | Corn |

## Technical Details

- **Model**: U-Net (TensorFlow/Keras)
- **Tile size**: 128×128 with stride 64 (overlapping tiles)
- **Blending**: Hann window weighting to reduce tile boundary artifacts
- **Post-processing**: Connected component analysis removes patches smaller than 50 pixels, replacing them with the majority neighboring class
- **Input**: 5 monthly composites × 17 bands = 85 input channels

## Files

| File | Description |
|------|-------------|
| `api.py` | FastAPI endpoints |
| `UNet_Inference.py` | Inference pipeline with tile-based prediction and post-processing |
| `Dockerfile` | Container setup with GDAL dependencies |
| `requirements.txt` | Python dependencies |
