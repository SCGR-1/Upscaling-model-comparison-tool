"""Main FastAPI application - refactored to use components."""
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import numpy as np
import torch
from pathlib import Path
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

# Import components
from config import (
    PICTURES_DIR, OUTPUT_DIR, MAX_OUT_LONG_EDGE, ENFORCE_OUTPUT_CAP,
    COMMON_IMG_HEADERS, LANCZOS, device
)
from models import have_realesrgan, have_pan, have_edsr, warmup_models
from utils import safe_join, _etag_for, target_size
from image_processing import upscale_multi
from cancellation import (
    CancellationError, check_cancellation, register_cancellation,
    cancel_request, unregister_cancellation, is_cancelled
)
from frontend import get_frontend_html

# Metrics
try:
    from metrics_tools import compute_upscale_metrics
    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False

# Initialize app
app = FastAPI(title="Local Image Upscaler")

# CORS for local UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool executor
_executor = ThreadPoolExecutor(max_workers=1)

# Warmup models at startup
warmup_models()


# Routes
@app.get("/", response_class=HTMLResponse)
def frontend():
    """Serve the frontend HTML."""
    return get_frontend_html()


@app.get("/api/images")
def list_images():
    """List all images in the input directory."""
    images = []
    if PICTURES_DIR.exists():
        for file in PICTURES_DIR.iterdir():
            if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                images.append(file.name)
    return JSONResponse({"images": sorted(images)})


@app.get("/api/backends")
def backends():
    """Get available backend status."""
    return {"realesrgan": have_realesrgan, "pan": have_pan, "edsr": have_edsr}


@app.get("/api/config")
def config():
    """Get configuration."""
    return {"max_out_long_edge": MAX_OUT_LONG_EDGE}


@app.get("/input/{filename}")
def get_input_image(filename: str, request: Request):
    """Serve input images."""
    p = safe_join(PICTURES_DIR, filename)
    if not p.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    etag = _etag_for(p)
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return FileResponse(p, headers={**COMMON_IMG_HEADERS, "ETag": etag})


@app.get("/output/{filename}")
def get_output_image(filename: str, request: Request):
    """Serve output images."""
    p = safe_join(OUTPUT_DIR, filename)
    if not p.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    etag = _etag_for(p)
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return FileResponse(p, headers={**COMMON_IMG_HEADERS, "ETag": etag})


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "device": str(device)}


@app.on_event("shutdown")
def _shutdown():
    """Cleanup on shutdown."""
    _executor.shutdown(wait=False, cancel_futures=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.post("/cancel")
async def cancel_upscale(request_id: str = Query(..., description="Request ID to cancel")):
    """Cancel an ongoing upscale operation."""
    if cancel_request(request_id):
        return JSONResponse({"success": True, "message": "Cancellation requested"})
    return JSONResponse({"success": False, "message": "Request not found"}, status_code=404)


@app.post("/upscale")
async def upscale(
    filename: str = Query(..., description="Name of the image file in /pictures/input directory"),
    backend: str = Query("realesrgan", pattern="^(realesrgan|pan|edsr)$"),
    scale: int = Query(4, ge=2, le=16),
    request_id: str = Query(None, description="Request ID for cancellation"),
    metrics: bool = Query(True, description="Compute quality metrics"),
):
    """Upscale a single image."""
    start_time = time.time()
    input_path = safe_join(PICTURES_DIR, filename)
    if not input_path.exists():
        return JSONResponse({"error": f"File not found: {filename}"}, status_code=404)
    
    if request_id:
        register_cancellation(request_id)
    
    try:
        check_cancellation(request_id)
        with Image.open(input_path) as im:
            im = ImageOps.exif_transpose(im)
            has_alpha = im.mode in ("RGBA", "LA")
            alpha = im.getchannel("A") if has_alpha else None
            img = im.convert("RGB").copy()
        orig_w, orig_h = img.size
        check_cancellation(request_id)
    except CancellationError:
        if request_id:
            unregister_cancellation(request_id)
        return JSONResponse({"error": "Processing cancelled by user", "cancelled": True}, status_code=499)
    except Exception as e:
        if request_id:
            unregister_cancellation(request_id)
        return JSONResponse({"error": f"Failed to open image: {str(e)}"}, status_code=400)
    
    def run_upscale():
        try:
            check_cancellation(request_id)
            
            result = upscale_multi(img, backend, scale, request_id, check_cancellation)
            if isinstance(result, dict) and not result.get("success", True):
                return result
            
            out = result
            
            tw, th = target_size(orig_w, orig_h, scale, MAX_OUT_LONG_EDGE, ENFORCE_OUTPUT_CAP)
            out = out.resize((tw, th), LANCZOS)
            
            if has_alpha:
                a_resized = alpha.resize((tw, th), LANCZOS)
                out = out.convert("RGBA")
                out.putalpha(a_resized)
            
            check_cancellation(request_id)
            
            base_name = Path(filename).stem
            output_filename = f"{base_name}_upscaled_{backend}_{scale}x.png"
            output_path = OUTPUT_DIR / output_filename
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            out.save(tmp_path, format="PNG", optimize=True, compress_level=6)
            os.replace(tmp_path, output_path)
            out_w, out_h = out.size
            
            tgt_long = max(orig_w, orig_h) * scale
            warning = None
            capped = False
            if ENFORCE_OUTPUT_CAP and tgt_long > MAX_OUT_LONG_EDGE:
                warning = f"Final output capped to long edge {MAX_OUT_LONG_EDGE}px (requested {tgt_long}px)."
                capped = True
            
            return {
                "success": True, "output": out, "output_path": output_path,
                "output_filename": output_filename, "output_size": [out_w, out_h],
                "warning": warning, "capped": capped, "filename": filename
            }
        except CancellationError:
            raise
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    try:
        future = _executor.submit(run_upscale)
        
        while True:
            try:
                result = future.result(timeout=0.5)
                break
            except FutureTimeoutError:
                if request_id and is_cancelled(request_id):
                    try:
                        result = future.result(timeout=0.1)
                        break
                    except (FutureTimeoutError, Exception):
                        if request_id:
                            unregister_cancellation(request_id)
                        return JSONResponse({"error": "Processing cancelled by user", "cancelled": True}, status_code=499)
        
        if request_id:
            unregister_cancellation(request_id)
        
        if not result["success"]:
            status_code = result.get("status_code", 400)
            return JSONResponse({"error": result["error"], "success": False}, status_code=status_code)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        processing_time = time.time() - start_time
        
        eff_w, eff_h = result["output_size"]
        eff_scale = round(max(eff_w / orig_w, eff_h / orig_h), 3)
        
        m = None
        if metrics and _HAS_METRICS:
            m = compute_upscale_metrics(result["output"], img, compare_bicubic=True, device=device)
        
        response_data = {
            "success": True,
            "input_file": str(input_path),
            "output_file": str(result["output_path"]),
            "output_filename": result["output_filename"],
            "filename": result.get("filename", filename),
            "backend": backend,
            "scale": scale,
            "effective_scale": eff_scale,
            "output_size": result["output_size"],
            "processing_time": processing_time
        }
        if result.get("warning"):
            response_data["warning"] = result["warning"]
        if result.get("capped") is not None:
            response_data["capped"] = result["capped"]
        if m is not None:
            response_data["metrics"] = m
        headers = {"Server-Timing": f"process;dur={processing_time*1000:.2f}"}
        return JSONResponse(response_data, headers=headers)
    except CancellationError:
        if request_id:
            unregister_cancellation(request_id)
        return JSONResponse({"error": "Processing cancelled by user", "cancelled": True}, status_code=499)
    except Exception as e:
        if request_id:
            unregister_cancellation(request_id)
        return JSONResponse({"error": str(e), "success": False}, status_code=500)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _calculate_percentiles(values: list, percentiles: list = [25, 50, 75]) -> list:
    """Calculate percentiles for a list of values, filtering out None/NaN."""
    valid = [v for v in values if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
    if not valid:
        return [None, None, None]
    sorted_vals = sorted(valid)
    result = []
    for p in percentiles:
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        result.append(float(sorted_vals[idx]))
    return result


@app.post("/batch")
async def batch_upscale(
    backend: str = Query("realesrgan", pattern="^(realesrgan|pan|edsr)$"),
    scale: int = Query(4, ge=2, le=16),
    request_id: str = Query(None, description="Request ID for cancellation"),
    metrics: bool = Query(True, description="Compute quality metrics"),
):
    """Batch process all images in input/ directory."""
    start_time = time.time()
    
    if request_id:
        register_cancellation(request_id)
    
    image_files = []
    if PICTURES_DIR.exists():
        for file in PICTURES_DIR.iterdir():
            if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                image_files.append(file.name)
    
    if not image_files:
        return JSONResponse({"error": "No images found in input directory", "success": False}, status_code=400)
    
    output_folder = Path("pictures") / backend
    output_folder.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    processed_count = 0
    errors = []
    
    def process_single_image(filename: str):
        """Process a single image and return result with metrics."""
        try:
            check_cancellation(request_id)
            input_path = safe_join(PICTURES_DIR, filename)
            
            with Image.open(input_path) as im:
                im = ImageOps.exif_transpose(im)
                has_alpha = im.mode in ("RGBA", "LA")
                alpha = im.getchannel("A") if has_alpha else None
                img = im.convert("RGB").copy()
            orig_w, orig_h = img.size
            
            # Start timing for actual upscaling
            img_start_time = time.time()
            result = upscale_multi(img, backend, scale, request_id, check_cancellation)
            if isinstance(result, dict) and not result.get("success", True):
                return {"success": False, "error": result.get("error", "Upscaling failed"), "filename": filename}
            
            out = result
            tw, th = target_size(orig_w, orig_h, scale, MAX_OUT_LONG_EDGE, ENFORCE_OUTPUT_CAP)
            out = out.resize((tw, th), LANCZOS)
            
            if has_alpha:
                a_resized = alpha.resize((tw, th), LANCZOS)
                out = out.convert("RGBA")
                out.putalpha(a_resized)
            
            check_cancellation(request_id)
            
            base_name = Path(filename).stem
            output_filename = f"{base_name}_upscaled_{scale}x.png"
            output_path = output_folder / output_filename
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            out.save(tmp_path, format="PNG", optimize=True, compress_level=6)
            os.replace(tmp_path, output_path)
            
            # Stop timing after upscaling and saving (before metrics)
            img_processing_time = time.time() - img_start_time
            
            # Compute metrics after timing (not included in processing_time)
            m = None
            if metrics and _HAS_METRICS:
                m = compute_upscale_metrics(out, img, compare_bicubic=True, device=device)
            
            return {
                "success": True,
                "filename": filename,
                "output_filename": output_filename,
                "output_path": str(output_path),
                "processing_time": img_processing_time,
                "metrics": m
            }
        except CancellationError:
            raise
        except Exception as e:
            return {"success": False, "error": str(e), "filename": filename}
    
    try:
        for filename in image_files:
            check_cancellation(request_id)
            result = process_single_image(filename)
            if result["success"]:
                processed_count += 1
                if result.get("metrics"):
                    all_metrics.append({
                        "filename": result["filename"],
                        "output_filename": result["output_filename"],
                        "processing_time": result.get("processing_time"),
                        "metrics": result["metrics"]
                    })
            else:
                errors.append({"filename": filename, "error": result.get("error", "Unknown error")})
        
        processing_time = time.time() - start_time
        
        percentiles = None
        if all_metrics and metrics:
            metric_arrays = {
                "no_reference": {
                    "NIQE": [], "BRISQUE": [], "LaplacianVar": [], "Tenengrad": []
                },
                "downscale_consistency": {
                    "PSNR": [], "SSIM": []
                },
                "deltas_vs_bicubic": {
                    "NIQE": [], "BRISQUE": [], "LaplacianVar": [], "Tenengrad": [],
                    "PSNR": [], "SSIM": []
                }
            }
            
            processing_times = []
            
            for item in all_metrics:
                m = item["metrics"]
                if item.get("processing_time") is not None:
                    processing_times.append(item["processing_time"])
                
                if m.get("no_reference"):
                    nr = m["no_reference"]
                    metric_arrays["no_reference"]["NIQE"].append(nr.get("NIQE"))
                    metric_arrays["no_reference"]["BRISQUE"].append(nr.get("BRISQUE"))
                    metric_arrays["no_reference"]["LaplacianVar"].append(nr.get("LaplacianVar"))
                    metric_arrays["no_reference"]["Tenengrad"].append(nr.get("Tenengrad"))
                
                if m.get("downscale_consistency"):
                    dc = m["downscale_consistency"]
                    metric_arrays["downscale_consistency"]["PSNR"].append(dc.get("PSNR"))
                    metric_arrays["downscale_consistency"]["SSIM"].append(dc.get("SSIM"))
                
                if m.get("deltas_vs_bicubic"):
                    deltas = m["deltas_vs_bicubic"]
                    metric_arrays["deltas_vs_bicubic"]["NIQE"].append(deltas.get("NIQE"))
                    metric_arrays["deltas_vs_bicubic"]["BRISQUE"].append(deltas.get("BRISQUE"))
                    metric_arrays["deltas_vs_bicubic"]["LaplacianVar"].append(deltas.get("LaplacianVar"))
                    metric_arrays["deltas_vs_bicubic"]["Tenengrad"].append(deltas.get("Tenengrad"))
                    metric_arrays["deltas_vs_bicubic"]["PSNR"].append(deltas.get("PSNR"))
                    metric_arrays["deltas_vs_bicubic"]["SSIM"].append(deltas.get("SSIM"))
            
            percentiles = {
                "processing_time": _calculate_percentiles(processing_times) if processing_times else None,
                "no_reference": {
                    "NIQE": _calculate_percentiles(metric_arrays["no_reference"]["NIQE"]),
                    "BRISQUE": _calculate_percentiles(metric_arrays["no_reference"]["BRISQUE"]),
                    "LaplacianVar": _calculate_percentiles(metric_arrays["no_reference"]["LaplacianVar"]),
                    "Tenengrad": _calculate_percentiles(metric_arrays["no_reference"]["Tenengrad"])
                },
                "downscale_consistency": {
                    "PSNR": _calculate_percentiles(metric_arrays["downscale_consistency"]["PSNR"]),
                    "SSIM": _calculate_percentiles(metric_arrays["downscale_consistency"]["SSIM"])
                },
                "deltas_vs_bicubic": {
                    "NIQE": _calculate_percentiles(metric_arrays["deltas_vs_bicubic"]["NIQE"]),
                    "BRISQUE": _calculate_percentiles(metric_arrays["deltas_vs_bicubic"]["BRISQUE"]),
                    "LaplacianVar": _calculate_percentiles(metric_arrays["deltas_vs_bicubic"]["LaplacianVar"]),
                    "Tenengrad": _calculate_percentiles(metric_arrays["deltas_vs_bicubic"]["Tenengrad"]),
                    "PSNR": _calculate_percentiles(metric_arrays["deltas_vs_bicubic"]["PSNR"]),
                    "SSIM": _calculate_percentiles(metric_arrays["deltas_vs_bicubic"]["SSIM"])
                }
            }
        
        metrics_file = None
        percentiles_file = None
        if all_metrics:
            timestamp = int(time.time())
            metrics_file = f"metrics_{backend}_{scale}x_{timestamp}.json"
            metrics_path = output_folder / metrics_file
            with open(metrics_path, 'w') as f:
                json.dump({
                    "backend": backend,
                    "scale": scale,
                    "processed_count": processed_count,
                    "total_images": len(image_files),
                    "errors": errors,
                    "timestamp": timestamp,
                    "processing_time": processing_time,
                    "metrics": all_metrics,
                    "percentiles": percentiles
                }, f, indent=2)
            
            if percentiles:
                percentiles_file = f"percentiles_{backend}_{scale}x_{timestamp}.json"
                percentiles_path = output_folder / percentiles_file
                with open(percentiles_path, 'w') as f:
                    json.dump({
                        "backend": backend,
                        "scale": scale,
                        "processed_count": processed_count,
                        "total_images": len(image_files),
                        "timestamp": timestamp,
                        "total_processing_time": processing_time,
                        "percentiles": percentiles
                    }, f, indent=2)
        
        response_data = {
            "success": True,
            "batch_results": True,
            "processed_count": processed_count,
            "total_images": len(image_files),
            "output_folder": str(output_folder),
            "metrics_file": metrics_file,
            "percentiles_file": percentiles_file,
            "backend": backend,
            "scale": scale,
            "processing_time": processing_time,
            "percentiles": percentiles,
            "errors": errors if errors else None
        }
        
        return JSONResponse(response_data)
        
    except CancellationError:
        if request_id:
            unregister_cancellation(request_id)
        return JSONResponse({"error": "Processing cancelled by user", "cancelled": True}, status_code=499)
    except Exception as e:
        if request_id:
            unregister_cancellation(request_id)
        return JSONResponse({"error": str(e), "success": False}, status_code=500)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/compute-input-metrics")
async def compute_input_metrics(
    request_id: str = Query(None, description="Request ID for cancellation"),
):
    """Compute metrics for all input images and save to JSON file in input/ directory."""
    start_time = time.time()
    
    if not _HAS_METRICS:
        return JSONResponse({"error": "Metrics tools not available", "success": False}, status_code=400)
    
    if request_id:
        register_cancellation(request_id)
    
    image_files = []
    if PICTURES_DIR.exists():
        for file in PICTURES_DIR.iterdir():
            if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                image_files.append(file.name)
    
    if not image_files:
        return JSONResponse({"error": "No images found in input directory", "success": False}, status_code=400)
    
    all_metrics = []
    processed_count = 0
    errors = []
    
    try:
        for filename in image_files:
            check_cancellation(request_id)
            try:
                img_start_time = time.time()
                input_path = safe_join(PICTURES_DIR, filename)
                
                with Image.open(input_path) as im:
                    im = ImageOps.exif_transpose(im)
                    img = im.convert("RGB").copy()
                
                metrics_result = compute_upscale_metrics(img, img, compare_bicubic=False, device=device)
                
                img_processing_time = time.time() - img_start_time
                
                input_metrics = {
                    "filename": filename,
                    "processing_time": img_processing_time,
                    "no_reference": metrics_result.get("no_reference", {}),
                    "image_size": list(img.size)
                }
                
                all_metrics.append(input_metrics)
                processed_count += 1
                
            except CancellationError:
                raise
            except Exception as e:
                errors.append({"filename": filename, "error": str(e)})
        
        processing_time = time.time() - start_time
        
        percentiles = None
        if all_metrics:
            metric_arrays = {
                "no_reference": {
                    "NIQE": [], "BRISQUE": [], "LaplacianVar": [], "Tenengrad": []
                }
            }
            processing_times = []
            
            for item in all_metrics:
                if item.get("processing_time") is not None:
                    processing_times.append(item["processing_time"])
                
                if item.get("no_reference"):
                    nr = item["no_reference"]
                    metric_arrays["no_reference"]["NIQE"].append(nr.get("NIQE"))
                    metric_arrays["no_reference"]["BRISQUE"].append(nr.get("BRISQUE"))
                    metric_arrays["no_reference"]["LaplacianVar"].append(nr.get("LaplacianVar"))
                    metric_arrays["no_reference"]["Tenengrad"].append(nr.get("Tenengrad"))
            
            percentiles = {
                "processing_time": _calculate_percentiles(processing_times) if processing_times else None,
                "no_reference": {
                    "NIQE": _calculate_percentiles(metric_arrays["no_reference"]["NIQE"]),
                    "BRISQUE": _calculate_percentiles(metric_arrays["no_reference"]["BRISQUE"]),
                    "LaplacianVar": _calculate_percentiles(metric_arrays["no_reference"]["LaplacianVar"]),
                    "Tenengrad": _calculate_percentiles(metric_arrays["no_reference"]["Tenengrad"])
                }
            }
        
        timestamp = int(time.time())
        metrics_file = f"input_metrics_{timestamp}.json"
        metrics_path = PICTURES_DIR / metrics_file
        
        with open(metrics_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "processed_count": processed_count,
                "total_images": len(image_files),
                "errors": errors if errors else None,
                "processing_time": processing_time,
                "metrics": all_metrics,
                "percentiles": percentiles
            }, f, indent=2)
        
        percentiles_file = None
        if percentiles:
            percentiles_file = f"input_percentiles_{timestamp}.json"
            percentiles_path = PICTURES_DIR / percentiles_file
            with open(percentiles_path, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "processed_count": processed_count,
                    "total_images": len(image_files),
                    "total_processing_time": processing_time,
                    "percentiles": percentiles
                }, f, indent=2)
        
        return JSONResponse({
            "success": True,
            "processed_count": processed_count,
            "total_images": len(image_files),
            "metrics_file": metrics_file,
            "percentiles_file": percentiles_file,
            "processing_time": processing_time,
            "percentiles": percentiles,
            "errors": errors if errors else None
        })
        
    except CancellationError:
        if request_id:
            unregister_cancellation(request_id)
        return JSONResponse({"error": "Processing cancelled by user", "cancelled": True}, status_code=499)
    except Exception as e:
        if request_id:
            unregister_cancellation(request_id)
        return JSONResponse({"error": str(e), "success": False}, status_code=500)

