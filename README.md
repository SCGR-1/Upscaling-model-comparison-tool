# Image Upscaler Service

A FastAPI-based image upscaling service with multiple backend support and quality metrics.

## Features

- **Multiple Upscaling Backends:**
  - Real-ESRGAN (4×)
  - PAN (2×, 3×, 4×)
  - EDSR (2×, 3×, 4×)
  - Cascaded upscaling for higher scales (8×, 16×)

- **Quality Metrics:**
  - No-reference metrics: NIQE, BRISQUE, Laplacian Variance
  - Downscale consistency: PSNR, SSIM
  - Comparison with bicubic baseline
  - Delta metrics showing improvement over bicubic

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Optional metrics dependencies (for full metrics support):
```bash
pip install pyiqa scikit-image
```

## Usage

1. Place input images in `pictures/input/` directory

2. Start the server:

   Using uvicorn directly:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

   Using the Makefile:
   ```bash
   make run        # Production mode
   make dev        # Development mode (auto-reload)
   ```

   Or on Windows using PowerShell:
   ```powershell
   .\run-server.ps1        # Production mode
   .\run-server.ps1 -Dev   # Development mode (auto-reload)
   ```

3. Open your browser and navigate to `http://localhost:8000`

4. Select an image, choose backend and scale, then click "Upscale Image"

## API Endpoints

- `GET /` - Web UI
- `POST /upscale` - Upscale an image
  - Query parameters:
    - `filename`: Name of image file in `pictures/input/`
    - `backend`: `realesrgan`, `pan`, or `edsr`
    - `scale`: 2, 3, 4, 8, or 16
    - `metrics`: `true` or `false` (default: `true`)
    - `request_id`: Optional request ID for cancellation
- `POST /cancel` - Cancel an in-progress upscale operation
- `GET /api/images` - List available input images
- `GET /api/backends` - List available backends
- `GET /input/{filename}` - Get input image
- `GET /output/{filename}` - Get output image
- `GET /health` - Health check

## Project Structure

```
upscaler-service/
├── app.py              # FastAPI application
├── metrics_tools.py    # Quality metrics computation
├── requirements.txt    # Python dependencies
├── pictures/
│   ├── input/         # Input images
│   └── output/        # Upscaled images
└── weights/           # Model weights
```

## Notes

- Maximum output size is capped at 16384px on the long edge by default
- Images are automatically resized if they exceed the cap
- Processing can be cancelled using the cancel button or API endpoint
- Metrics computation requires optional dependencies (`pyiqa`, `scikit-image`)

