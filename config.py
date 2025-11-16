"""Configuration constants and settings."""
from pathlib import Path
from PIL import Image
import torch

# Image processing limits
Image.MAX_IMAGE_PIXELS = 300_000_000  # ~300 MP cap
MAX_OUT_LONG_EDGE = 16384  # Maximum output long edge
ENFORCE_OUTPUT_CAP = True  # If True, final PNG never exceeds MAX_OUT_LONG_EDGE

# Backend step configurations
BACKEND_STEPS = {
    "realesrgan": [4],       # Only x4 weights loaded
    "pan":        [2, 3, 4],
    "edsr":       [2, 3, 4],
    "swinir":     [2, 3, 4, 8],
}

# Directory paths
PICTURES_DIR = Path("pictures") / "input"
OUTPUT_DIR = Path("pictures") / "output"

# Create directories if they don't exist
PICTURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HTTP headers
COMMON_IMG_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable, no-transform",
    "X-Content-Type-Options": "nosniff",
}

# PyTorch settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.set_num_threads(1)  # Reduce CPU thrash on tile stitching

# LANCZOS resampling
try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

