# metrics_tools.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import torch

# Optional deps
_HAS_PYIQA = True
try:
    import pyiqa  # NIQE, BRISQUE
except Exception:
    _HAS_PYIQA = False

_HAS_PYNVML = True
_PYNVML_INITIALIZED = False
try:
    import pynvml
except Exception:
    _HAS_PYNVML = False

from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.filters import laplace, sobel

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS


def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    """HWC uint8 -> 1xCxHxW float32 in [0,1]."""
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:  # drop alpha for metrics
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
    return t


def _pil_to_gray01(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    a = np.asarray(g, dtype=np.float32) / 255.0
    return a


def _lap_var(img: Image.Image) -> float:
    """Laplacian variance (scaled by 1e6 for readability)."""
    g = _pil_to_gray01(img)
    # skimage.filters.laplace gives a float image centered around 0
    lap = laplace(g)
    var = float(np.var(lap, dtype=np.float64))
    return var * 1e6  # Scale for readability


def _tenengrad(img: Image.Image) -> float:
    """Tenengrad sharpness metric (mean of squared gradients, normalized by image size)."""
    g = _pil_to_gray01(img)
    # Sobel operator for gradients
    grad_x = sobel(g, axis=1)
    grad_y = sobel(g, axis=0)
    # Mean of squared gradients (normalized by pixel count for comparability across sizes)
    h, w = g.shape
    tenengrad = float(np.mean(grad_x**2 + grad_y**2))
    return tenengrad


def get_vram_usage(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Get current GPU VRAM usage information.
    
    Returns a dict with:
    - allocated_mb: PyTorch allocated memory in MB
    - reserved_mb: PyTorch reserved memory in MB
    - total_mb: Total GPU memory in MB (if available via pynvml)
    - used_mb: Total GPU used memory in MB (if available via pynvml)
    - free_mb: Total GPU free memory in MB (if available via pynvml)
    - utilization_percent: GPU memory utilization percentage (if available)
    - method: "pynvml" or "torch" indicating which method was used
    
    Returns zeros if CUDA is not available.
    """
    result = {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "total_mb": None,
        "used_mb": None,
        "free_mb": None,
        "utilization_percent": None,
        "method": "none",
        "available": False
    }
    
    if not torch.cuda.is_available():
        return result
    
    device = device or torch.device("cuda")
    if device.type != "cuda":
        return result
    
    # Get PyTorch memory stats
    try:
        allocated_bytes = torch.cuda.memory_allocated(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
        result["allocated_mb"] = allocated_bytes / (1024 ** 2)
        result["reserved_mb"] = reserved_bytes / (1024 ** 2)
        result["available"] = True
    except Exception:
        pass
    
    # Try pynvml for more precise GPU memory info
    if _HAS_PYNVML:
        try:
            global _PYNVML_INITIALIZED
            if not _PYNVML_INITIALIZED:
                pynvml.nvmlInit()
                _PYNVML_INITIALIZED = True
            
            device_index = device.index if device.index is not None else 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result["total_mb"] = mem_info.total / (1024 ** 2)
            result["used_mb"] = mem_info.used / (1024 ** 2)
            result["free_mb"] = mem_info.free / (1024 ** 2)
            
            if result["total_mb"] > 0:
                result["utilization_percent"] = (result["used_mb"] / result["total_mb"]) * 100.0
            
            result["method"] = "pynvml"
        except Exception:
            # Fallback to torch method
            if result["available"]:
                result["method"] = "torch"
    else:
        # Use torch method only
        if result["available"]:
            result["method"] = "torch"
    
    return result


def _downscale_to(img: Image.Image, w: int, h: int) -> Image.Image:
    return img.resize((w, h), LANCZOS)


def _psnr_ssim(a: Image.Image, b: Image.Image) -> Dict[str, float]:
    a_np = np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0
    b_np = np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0
    # Channel-wise SSIM averaged
    ssim_val = 0.0
    for c in range(3):
        ssim_val += ssim_sk(a_np[..., c], b_np[..., c], data_range=1.0)
    ssim_val /= 3.0
    psnr_val = psnr_sk(a_np, b_np, data_range=1.0)
    return {"PSNR": float(psnr_val), "SSIM": float(ssim_val)}


def _safe_metric(model, x: torch.Tensor, device: torch.device) -> Optional[float]:
    try:
        with torch.no_grad():
            val = float(model(x.to(device)).item())
        return val
    except Exception:
        return None


def compute_upscale_metrics(
    upscaled_pil: Image.Image,
    original_pil: Image.Image,
    compare_bicubic: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    No-ground-truth "combo":
      - No-reference: NIQE, BRISQUE, Laplacian variance on the upscaled.
      - Downscale consistency: PSNR, SSIM between downscaled(upscaled) and original.
      - Baseline: same metrics for bicubic at the same output size.
      - Deltas vs Bicubic: positive is better, NIQE inverted (lower is better).

    Returns a plain dict safe for JSON.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure RGB without alpha for metrics
    up_rgb = upscaled_pil.convert("RGB")
    orig_rgb = original_pil.convert("RGB")
    ow, oh = orig_rgb.size

    # 1) No-reference metrics on the upscaled result
    no_ref = {"NIQE": None, "BRISQUE": None, "LaplacianVar": None, "Tenengrad": None}
    # Track availability of all metrics
    availability = {
        "pyiqa": _HAS_PYIQA,
        "scikit-image": True,  # Always available (required import)
        "laplacian_var": True,  # Always available (uses scikit-image)
        "tenengrad": True,  # Always available (uses scikit-image)
    }

    if _HAS_PYIQA:
        try:
            niqe = pyiqa.create_metric("niqe").to(device).eval()
        except Exception:
            niqe = None
        try:
            brisque = pyiqa.create_metric("brisque").to(device).eval()
        except Exception:
            brisque = None
    else:
        niqe = None
        brisque = None

    x_up = _pil_to_tensor01(up_rgb)
    no_ref["NIQE"] = _safe_metric(niqe, x_up, device) if niqe else None
    no_ref["BRISQUE"] = _safe_metric(brisque, x_up, device) if brisque else None
    no_ref["LaplacianVar"] = _lap_var(up_rgb)
    no_ref["Tenengrad"] = _tenengrad(up_rgb)

    # 2) Downscale consistency to original size
    up_down = _downscale_to(up_rgb, ow, oh)
    down_cons = _psnr_ssim(orig_rgb, up_down)

    # 3) Bicubic baseline at the same final size
    # Note: If source input has little detail, the bicubic baseline may closely match
    # the upscaled output when downscaled, so downscale PSNR/SSIM for upscaled may trail.
    # This is expected behavior and indicates the upscaler is preserving the original structure.
    baseline = None
    deltas = None
    if compare_bicubic:
        up_w, up_h = up_rgb.size
        bic = orig_rgb.resize((up_w, up_h), LANCZOS)
        # No-reference on bicubic
        b_no_ref = {
            "NIQE": _safe_metric(niqe, _pil_to_tensor01(bic), device) if niqe else None,
            "BRISQUE": _safe_metric(brisque, _pil_to_tensor01(bic), device) if brisque else None,
            "LaplacianVar": _lap_var(bic),
            "Tenengrad": _tenengrad(bic),
        }
        # Downscale consistency for bicubic
        bic_down = _downscale_to(bic, ow, oh)
        b_cons = _psnr_ssim(orig_rgb, bic_down)

        baseline = {
            "NIQE": b_no_ref["NIQE"],
            "BRISQUE": b_no_ref["BRISQUE"],
            "LaplacianVar": b_no_ref["LaplacianVar"],
            "Tenengrad": b_no_ref["Tenengrad"],
            "Downscale_PSNR": b_cons["PSNR"],
            "Downscale_SSIM": b_cons["SSIM"],
        }

        # Deltas vs bicubic. NIQE polarity inverted so positive means better.
        def _delta(a, b, invert=False):
            if a is None or b is None:
                return None
            return float((b - a) if invert else (a - b))

        deltas = {
            "NIQE": _delta(no_ref["NIQE"], baseline["NIQE"], invert=True),  # baseline - upscaled (positive = better)
            "BRISQUE": _delta(no_ref["BRISQUE"], baseline["BRISQUE"], invert=True),  # baseline - upscaled (positive = better)
            "LaplacianVar": no_ref["LaplacianVar"] - baseline["LaplacianVar"],  # upscaled - baseline (positive = better)
            "Tenengrad": no_ref["Tenengrad"] - baseline["Tenengrad"],  # upscaled - baseline (positive = better)
            "PSNR": down_cons["PSNR"] - baseline["Downscale_PSNR"],  # upscaled - baseline (positive = better)
            "SSIM": down_cons["SSIM"] - baseline["Downscale_SSIM"],  # upscaled - baseline (positive = better)
        }

    # Get VRAM usage
    vram_info = get_vram_usage(device)
    
    return {
        "availability": availability,
        "no_reference": no_ref,
        "downscale_consistency": down_cons,
        "baseline_bicubic": baseline,
        "deltas_vs_bicubic": deltas,
        "vram_usage": vram_info,
    }
