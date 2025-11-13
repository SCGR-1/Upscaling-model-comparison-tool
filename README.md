# Image Upscaler Service

A FastAPI-based image upscaling service with multiple backend support and quality metrics.

## Features

- **Multiple Upscaling Backends:**
  - Real-ESRGAN (4×)
  - PAN (2×, 3×, 4×)
  - EDSR (2×, 3×, 4×)
  - Cascaded upscaling for higher scales (8×, 16×)

- **Quality Metrics:**
  - No-reference metrics: NIQE, BRISQUE, Laplacian Variance (scaled), Tenengrad
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

### Batch Processing

- Enable "Batch mode" checkbox to process all images in `pictures/input/`
- Outputs are saved to `pictures/{backend}/` folder
- A metrics JSON file is generated with all results and percentile statistics
- Processing time and percentile metrics are displayed in the UI

### Input Metrics

- Click "Compute Input Metrics" to analyze all input images
- Computes no-reference quality metrics (NIQE, BRISQUE, LaplacianVar, Tenengrad)
- Results are saved to `pictures/input/input_metrics_{timestamp}.json`
- Useful for understanding input image quality before upscaling

## API Endpoints

- `GET /` - Web UI
- `POST /upscale` - Upscale a single image
  - Query parameters:
    - `filename`: Name of image file in `pictures/input/`
    - `backend`: `realesrgan`, `pan`, or `edsr`
    - `scale`: 2, 3, 4, 8, or 16
    - `metrics`: `true` or `false` (default: `true`)
    - `request_id`: Optional request ID for cancellation
- `POST /batch` - Batch upscale all images in `pictures/input/`
  - Query parameters:
    - `backend`: `realesrgan`, `pan`, or `edsr`
    - `scale`: 2, 3, 4, 8, or 16
    - `metrics`: `true` or `false` (default: `true`)
    - `request_id`: Optional request ID for cancellation
  - Outputs to `pictures/{backend}/` folder with metrics JSON file
  - Returns percentile statistics (25th, 50th, 75th) for all metrics
- `POST /compute-input-metrics` - Compute quality metrics for all input images
  - Query parameters:
    - `request_id`: Optional request ID for cancellation
  - Saves metrics to `pictures/input/input_metrics_{timestamp}.json`
- `POST /cancel` - Cancel an in-progress operation
  - Query parameters:
    - `request_id`: Request ID to cancel
- `GET /api/images` - List available input images
- `GET /api/backends` - List available backends and their status
- `GET /api/config` - Get configuration (max output size, etc.)
- `GET /input/{filename}` - Get input image
- `GET /output/{filename}` - Get output image
- `GET /health` - Health check endpoint

## Project Structure

```
upscaler-service/
├── app.py                 # Main FastAPI application and routes
├── config.py              # Configuration constants and settings
├── models.py              # Model loading and management
├── image_processing.py    # Image processing and upscaling functions
├── utils.py               # Utility functions (path safety, size calculations, etc.)
├── cancellation.py        # Request cancellation handling
├── frontend.py            # Frontend HTML template loader
├── frontend.html          # HTML/JavaScript frontend template
├── metrics_tools.py       # Quality metrics computation
├── requirements.txt       # Python dependencies
├── pictures/
│   ├── input/            # Input images (place your images here)
│   └── output/           # Upscaled images
└── weights/              # Model weights (downloaded automatically)
```

## Notes

- **Maximum output size** is capped at 16384px on the long edge by default
- Images are automatically resized if they exceed the cap
- **Processing can be cancelled** using the cancel button or API endpoint
- **Metrics computation** requires optional dependencies (`pyiqa`, `scikit-image`)
- **Viewing results**: For best visual assessment, view output images at 100% zoom (full resolution) to see sharpness differences
- **High-scale upscaling (8×/16×)**:
  - Works best on high-detail source images
  - Expect slower processing times
  - Requires more VRAM
  - Benefits are most visible on detailed, high-quality source images
- **Metrics interpretation**:
  - Lower is better: NIQE, BRISQUE
  - Higher is better: LaplacianVar (scaled ×1e6), Tenengrad, PSNR, SSIM
  - LaplacianVar and Tenengrad measure sharpness/high-frequency detail
  - Downscale consistency metrics (PSNR/SSIM) compare how well the upscaled image matches the original when downsampled

