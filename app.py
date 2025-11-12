from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from PIL import Image, ImageOps
import numpy as np
import torch
from pathlib import Path
import time
import threading
import os
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms.functional import to_pil_image

try:
    from metrics_tools import compute_upscale_metrics
    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

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

# Bound image size to avoid pathological inputs
Image.MAX_IMAGE_PIXELS = 300_000_000  # ~300 MP cap

MAX_OUT_LONG_EDGE = 16384  # 提高以支持 8×/16× 放大
ENFORCE_OUTPUT_CAP = True  # if True, the final PNG never exceeds MAX_OUT_LONG_EDGE

BACKEND_STEPS = {
    "realesrgan": [4],       # 只加载了 x4 权重
    "pan":        [2, 3, 4],
    "edsr":       [2, 3, 4],
}

def plan_steps(total: int, avail: list[int]) -> list[int]:
    """规划级联步长（例如 8 -> [4,2]；16 -> [4,4]）"""
    steps = []
    remain = total
    while remain > 1:
        cand = [s for s in avail if s <= remain and remain % s == 0]
        step = max(cand) if cand else max([s for s in avail if s <= remain])
        steps.append(step)
        remain //= step
    return steps

COMMON_IMG_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable, no-transform",
    "X-Content-Type-Options": "nosniff",
}

def target_size(orig_w, orig_h, scale, cap, enforce=True):
    tw, th = orig_w * scale, orig_h * scale
    if not enforce:
        return tw, th
    long_edge = max(tw, th)
    if long_edge <= cap:
        return tw, th
    r = cap / long_edge
    return max(1, int(tw * r)), max(1, int(th * r))

# --- Optional imports (loaded lazily) ---
have_realesrgan = have_pan = have_edsr = False
try:
    from py_real_esrgan.model import RealESRGAN as RealESRGANModel  # py-real-esrgan
    have_realesrgan = True
except Exception:
    pass
try:
    from super_image import PanModel
    have_pan = True
except Exception:
    pass
try:
    from super_image import EdsrModel
    have_edsr = True
except Exception:
    pass

app = FastAPI(title="Local Image Upscaler")

# CORS for local UI
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PICTURES_DIR = Path("pictures") / "input"
OUTPUT_DIR = Path("pictures") / "output"

PICTURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _etag_for(path: Path) -> str:
    st = path.stat()
    return f'W/"{st.st_mtime_ns}-{st.st_size}"'

def _choose_tile(target_bytes=256<<20):  # ~256MB budget
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        budget = max(64<<20, int(0.2 * free))
    else:
        budget = target_bytes
    side = int((budget / (3 * 4)) ** 0.5)  # 3ch * float32
    return max(192, min(768, side))

def _hann2d(h, w, device, dtype):
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype).unsqueeze(1)
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype).unsqueeze(0)
    return (wy * wx).clamp_(min=1e-6)

def _tile_infer(model, tensor, scale, tile=512, overlap=32, request_id=None):
    _, _, H, W = tensor.shape
    out_h, out_w = H * scale, W * scale
    # accumulate on CPU to keep VRAM low
    out_acc = torch.zeros(3, out_h, out_w, dtype=torch.float32)
    w_acc   = torch.zeros(3, out_h, out_w, dtype=torch.float32)
    step = max(1, tile - overlap)

    for ys in range(0, H, step):
        for xs in range(0, W, step):
            check_cancellation(request_id)
            ye, xe = min(ys + tile, H), min(xs + tile, W)
            patch = tensor[:, :, ys:ye, xs:xe]
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                pred = model(patch).squeeze(0)  # C x (ph) x (pw), device = cuda if available
            ph, pw = pred.shape[-2:]
            oy, ox = ys * scale, xs * scale

            wmask = _hann2d(ph, pw, pred.device, pred.dtype).expand_as(pred)
            predw = (pred * wmask).float().cpu()
            out_acc[:, oy:oy+ph, ox:ox+pw] += predw
            w_acc[:,  oy:oy+ph, ox:ox+pw] += wmask.float().cpu()

    out = out_acc / torch.clamp_min(w_acc, 1e-6)
    return out.unsqueeze(0)  # 1 x 3 x H*scale x W*scale (cpu float32)

def safe_join(base: Path, name: str) -> Path:
    p = (base / Path(name).name).resolve()
    if base.resolve() not in p.parents and base.resolve() != p:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return p

_pan_models = {}       # cache: {scale:int -> PanModel}
_edsr_models = {}      # cache: {scale:int -> EdsrModel}
_realesrgan_model = None

_cancellation_flags = {}  # {request_id: threading.Event}
_executor = ThreadPoolExecutor(max_workers=1)  # Single worker to prevent concurrent GPU operations

# Device/health log at startup
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Warmup Real-ESRGAN at startup
if have_realesrgan and torch.cuda.is_available():
    try:
        _realesrgan_model = RealESRGANModel(device, scale=4)
        _realesrgan_model.load_weights("weights/RealESRGAN_x4.pth", download=True)
        print("[Warmup] Real-ESRGAN ready")
    except Exception as e:
        print(f"[Warmup] Real-ESRGAN skipped: {e}")
        _realesrgan_model = None

def cap_input_for_scale(img, scale):
    w, h = img.size
    tgt_long = max(w, h) * scale
    if tgt_long <= MAX_OUT_LONG_EDGE:
        return img
    s = MAX_OUT_LONG_EDGE / tgt_long
    nw, nh = max(64, int(w * s)), max(64, int(h * s))
    return img.resize((nw, nh), LANCZOS)

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0  # HWC, 0..1
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1xCxHxW
    return t

@app.get("/", response_class=HTMLResponse)
def frontend():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Image Upscaler</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 { color: #333; margin-bottom: 30px; }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
        }
        select, input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            background: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        button.cancel {
            background: #dc3545;
            margin-left: 10px;
        }
        button.cancel:hover { background: #c82333; }
        .button-group {
            display: flex;
            gap: 10px;
        }
        #status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        #status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        #status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        #status.loading {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        #result {
            margin-top: 20px;
            text-align: center;
        }
        #preview {
            margin-top: 20px;
            text-align: center;
        }
        #preview img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        #comparison {
            margin-top: 20px;
            display: none;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }
        .comparison-item {
            text-align: center;
        }
        .comparison-item h3 {
            margin-bottom: 10px;
            color: #333;
            font-size: 16px;
        }
        .comparison-item img {
            max-width: 100%;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            image-rendering: crisp-edges;
        }
        #comparison img {
            image-rendering: crisp-edges;
        }
        #timer {
            font-size: 18px;
            font-weight: bold;
            color: #007bff;
            margin-top: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .comparison-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖼️ Image Upscaler</h1>
        
        <form id="upscaleForm">
            <div class="form-group">
                <label for="filename">Image File:</label>
                <select id="filename" name="filename" required>
                    <option value="">Loading...</option>
                </select>
            </div>
            
            <div class="grid">
                <div class="form-group">
                    <label for="backend">Backend:</label>
                    <select id="backend" name="backend">
                        <option value="realesrgan">Real-ESRGAN</option>
                        <option value="pan">PAN</option>
                        <option value="edsr">EDSR</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="scale">Scale:</label>
                    <select id="scale" name="scale">
                        <option value="2">2×</option>
                        <option value="3">3×</option>
                        <option value="4" selected>4×</option>
                        <option value="8">8×</option>
                        <option value="16">16×</option>
                    </select>
                </div>
            </div>
            
            <div class="button-group">
                <button type="submit" id="submitBtn">Upscale Image</button>
                <button type="button" id="cancelBtn" class="cancel" style="display: none;">Cancel</button>
            </div>
        </form>
        
        <div id="preview"></div>
        <div id="status"></div>
        <div id="timer"></div>
        <div id="comparison"></div>
    </div>
    
    <script>
        let startTime = null;
        let timerInterval = null;
        let currentRequestId = null;
        let abortController = null;
        
        function updateTimer() {
            if (startTime) {
                const elapsed = (Date.now() - startTime) / 1000;
                const minutes = Math.floor(elapsed / 60);
                const seconds = Math.floor(elapsed % 60);
                const ms = Math.floor((elapsed - Math.floor(elapsed)) * 1000);
                document.getElementById('timer').textContent = 
                    `Processing time: ${minutes}:${seconds.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
            }
        }
        
        function startTimer() {
            startTime = Date.now();
            timerInterval = setInterval(updateTimer, 50);
        }
        
        function stopTimer() {
            if (timerInterval) {
                clearInterval(timerInterval);
                timerInterval = null;
            }
        }
        
        async function loadImages() {
            try {
                const res = await fetch('/api/images');
                const data = await res.json();
                const select = document.getElementById('filename');
                select.innerHTML = '<option value="">Select an image...</option>';
                data.images.forEach(img => {
                    const option = document.createElement('option');
                    option.value = img;
                    option.textContent = img;
                    select.appendChild(option);
                });
            } catch (e) {
                console.error('Failed to load images:', e);
            }
        }
        
        document.getElementById('filename').addEventListener('change', function() {
            const filename = this.value;
            const preview = document.getElementById('preview');
            if (filename) {
                preview.innerHTML = `<h3>Preview:</h3><img src="/input/${encodeURIComponent(filename)}" alt="Preview">`;
            } else {
                preview.innerHTML = '';
            }
        });
        
        document.getElementById('cancelBtn').addEventListener('click', async () => {
            if (currentRequestId) {
                const status = document.getElementById('status');
                const timer = document.getElementById('timer');
                stopTimer();
                status.className = 'loading';
                status.textContent = 'Cancelling...';
                try {
                    await fetch(`/cancel?request_id=${encodeURIComponent(currentRequestId)}`, { method: 'POST' });
                    if (abortController) {
                        abortController.abort();
                    }
                } catch (e) {
                    console.error('Failed to cancel:', e);
                }
            }
        });

        document.getElementById('upscaleForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const cancelBtn = document.getElementById('cancelBtn');
            const status = document.getElementById('status');
            const comparison = document.getElementById('comparison');
            const timer = document.getElementById('timer');
            
            const filename = document.getElementById('filename').value;
            const backend = document.getElementById('backend').value;
            const scale = document.getElementById('scale').value;
            
            currentRequestId = Date.now().toString() + Math.random().toString(36).substr(2, 9);
            abortController = new AbortController();
            
            btn.disabled = true;
            cancelBtn.style.display = 'block';
            status.style.display = 'block';
            status.className = 'loading';
            status.textContent = 'Processing... This may take a while.';
            comparison.style.display = 'none';
            comparison.innerHTML = '';
            timer.textContent = '';
            startTimer();
            
            try {
                const res = await fetch(`/upscale?filename=${encodeURIComponent(filename)}&backend=${backend}&scale=${scale}&metrics=true&request_id=${encodeURIComponent(currentRequestId)}`, {
                    method: 'POST',
                    signal: abortController.signal
                });
                const data = await res.json();
                
                stopTimer();
                
                if (data.success) {
                    let infoText = `Scale: ${data.scale}x`;
                    if (data.effective_scale !== undefined && data.effective_scale !== data.scale) {
                        infoText += ` (effective: ${data.effective_scale}x)`;
                    }
                    if (data.capped) {
                        infoText += ' [CAPPED]';
                    }
                    if (data.warning) {
                        infoText += ` - ${data.warning}`;
                    }
                    status.className = 'success';
                    status.textContent = `Success! Output saved as: ${data.output_filename}. ${infoText}`;
                    timer.textContent = `Processing time: ${data.processing_time.toFixed(2)} seconds`;
                    comparison.style.display = 'block';
                    const afterTitle = data.effective_scale !== undefined && data.effective_scale !== data.scale 
                        ? `After (${data.effective_scale}x${data.capped ? ' - CAPPED' : ''})` 
                        : `After (${scale}x${data.capped ? ' - CAPPED' : ''})`;
                    comparison.innerHTML = `
                        <div class="comparison-grid">
                            <div class="comparison-item">
                                <h3>Before</h3>
                                <img src="/input/${encodeURIComponent(filename)}" alt="Original">
                            </div>
                            <div class="comparison-item">
                                <h3>${afterTitle}</h3>
                                <img src="/output/${encodeURIComponent(data.output_filename)}" alt="Upscaled">
                            </div>
                        </div>
                    `;
                    
                    if (data.metrics) {
                        try {
                            const m = data.metrics;
                            const pretty = s => {
                                if (s === null || s === undefined) return '—';
                                if (typeof s === 'number' && (isNaN(s) || !isFinite(s))) return '—';
                                if (typeof s === 'number') return s.toFixed(4);
                                return String(s);
                            };
                            
                            let metricsHtml = `
                                <div style="margin-top:20px;padding:16px;background:#f8f9fa;border-radius:8px;border:1px solid #dee2e6">
                                    <h3 style="margin-top:0;margin-bottom:16px;color:#333">📊 Quality Metrics</h3>
                                    <pre style="background:#ffffff;padding:12px;border-radius:6px;overflow:auto;margin:0;font-size:13px;line-height:1.5">
Availability: ${JSON.stringify(m.availability || {}, null, 2)}

Note: Lower is better for NIQE and BRISQUE. Higher is better for all others.

No-reference on Upscaled:
  NIQE: ${pretty(m.no_reference?.NIQE)} (lower is better)
  BRISQUE: ${pretty(m.no_reference?.BRISQUE)} (lower is better)
  LaplacianVar: ${pretty(m.no_reference?.LaplacianVar)}

Downscale Consistency (compare downscaled-upscaled vs original):
  PSNR: ${pretty(m.downscale_consistency?.PSNR)} dB
  SSIM: ${pretty(m.downscale_consistency?.SSIM)}
`;
                            
                            if (m.baseline_bicubic) {
                                metricsHtml += `
Baseline Bicubic @ same size:
  NIQE: ${pretty(m.baseline_bicubic.NIQE)}
  BRISQUE: ${pretty(m.baseline_bicubic.BRISQUE)}
  LaplacianVar: ${pretty(m.baseline_bicubic.LaplacianVar)}
  PSNR: ${pretty(m.baseline_bicubic.Downscale_PSNR)} dB
  SSIM: ${pretty(m.baseline_bicubic.Downscale_SSIM)}
`;
                            }
                            
                            if (m.deltas_vs_bicubic) {
                                metricsHtml += `
Deltas vs Bicubic (positive = your upscaler better; NIQE/BRISQUE inverted):
  ΔNIQE: ${pretty(m.deltas_vs_bicubic.NIQE)}
  ΔBRISQUE: ${pretty(m.deltas_vs_bicubic.BRISQUE)}
  ΔLaplacianVar: ${pretty(m.deltas_vs_bicubic.LaplacianVar)}
  ΔPSNR: ${pretty(m.deltas_vs_bicubic.PSNR)} dB
  ΔSSIM: ${pretty(m.deltas_vs_bicubic.SSIM)}
`;
                            }
                            
                            metricsHtml += `                                </pre>
                                </div>`;
                            
                            document.getElementById('comparison').insertAdjacentHTML('beforeend', metricsHtml);
                        } catch (e) {
                            console.error('Error displaying metrics:', e, data.metrics);
                        }
                    } else {
                        console.log('No metrics in response:', data);
                    }
                } else {
                    throw new Error(data.error || 'Upscaling failed');
                }
            } catch (error) {
                stopTimer();
                if (error.name === 'AbortError' || error.message.includes('cancelled')) {
                    status.className = 'error';
                    status.textContent = 'Processing cancelled by user.';
                } else {
                    status.className = 'error';
                    status.textContent = 'Error: ' + error.message;
                }
                timer.textContent = '';
            } finally {
                btn.disabled = false;
                cancelBtn.style.display = 'none';
                currentRequestId = null;
                abortController = null;
            }
        });
        
        loadImages();
    </script>
</body>
</html>
    """

@app.get("/api/images")
def list_images():
    images = []
    if PICTURES_DIR.exists():
        for file in PICTURES_DIR.iterdir():
            if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                images.append(file.name)
    return JSONResponse({"images": sorted(images)})

@app.get("/api/backends")
def backends():
    return {"realesrgan": have_realesrgan, "pan": have_pan, "edsr": have_edsr}

@app.get("/api/config")
def config():
    return {"max_out_long_edge": MAX_OUT_LONG_EDGE}

@app.get("/input/{filename}")
def get_input_image(filename: str, request: Request):
    p = safe_join(PICTURES_DIR, filename)
    if not p.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    etag = _etag_for(p)
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return FileResponse(p, headers={**COMMON_IMG_HEADERS, "ETag": etag})

@app.get("/output/{filename}")
def get_output_image(filename: str, request: Request):
    p = safe_join(OUTPUT_DIR, filename)
    if not p.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    etag = _etag_for(p)
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return FileResponse(p, headers={**COMMON_IMG_HEADERS, "ETag": etag})

@app.get("/health")
def health():
    return {"ok": True, "device": str(device)}

@app.on_event("shutdown")
def _shutdown():
    _executor.shutdown(wait=False, cancel_futures=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class CancellationError(Exception):
    pass

def check_cancellation(request_id: str):
    if request_id and _cancellation_flags.get(request_id, threading.Event()).is_set():
        raise CancellationError("Processing cancelled by user")

@app.post("/cancel")
async def cancel_upscale(request_id: str = Query(..., description="Request ID to cancel")):
    if request_id in _cancellation_flags:
        _cancellation_flags[request_id].set()
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
    start_time = time.time()
    input_path = safe_join(PICTURES_DIR, filename)
    if not input_path.exists():
        return JSONResponse({"error": f"File not found: {filename}"}, status_code=404)
    
    if request_id:
        _cancellation_flags[request_id] = threading.Event()
    
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
            _cancellation_flags.pop(request_id, None)
        return JSONResponse({"error": "Processing cancelled by user", "cancelled": True}, status_code=499)
    except Exception as e:
        if request_id:
            _cancellation_flags.pop(request_id, None)
        return JSONResponse({"error": f"Failed to open image: {str(e)}"}, status_code=400)
    
    def apply_once(img_in, backend_name, step_scale):
        """单次放大，封装每个后端的逻辑"""
        if backend_name == "realesrgan":
            if not have_realesrgan:
                raise Exception("Install py-real-esrgan")
            global _realesrgan_model
            if _realesrgan_model is None:
                _realesrgan_model = RealESRGANModel(device, scale=4)
                _realesrgan_model.load_weights("weights/RealESRGAN_x4.pth", download=True)
            check_cancellation(request_id)
            # soft cap before running 4x
            img2 = cap_input_for_scale(img_in, 4)
            try:
                out = _realesrgan_model.predict(img2)
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            return out
        
        elif backend_name == "pan":
            if not have_pan:
                raise Exception("Install super-image")
            if step_scale not in (2, 3, 4):
                raise Exception("PAN supports scales 2,3,4")
            model = _pan_models.get(step_scale)
            if model is None:
                model = PanModel.from_pretrained("eugenesiow/pan-bam", scale=step_scale).to(device)
                model.eval()
                _pan_models[step_scale] = model
            check_cancellation(request_id)
            inp = pil_to_tensor(img_in)
            if torch.cuda.is_available():
                inp = inp.contiguous(memory_format=torch.channels_last)
            inp = inp.to(device, non_blocking=True)
            # 直接分块推理，自动选 tile 尺寸
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                preds = _tile_infer(model, inp, step_scale, tile=_choose_tile(), request_id=request_id)
            out = to_pil_image(preds.squeeze(0).clamp(0, 1).float().cpu())
            return out
        
        elif backend_name == "edsr":
            if not have_edsr:
                raise Exception("Install super-image")
            if step_scale not in (2, 3, 4):
                raise Exception("EDSR supports scales 2,3,4")
            model = _edsr_models.get(step_scale)
            if model is None:
                model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=step_scale).to(device)
                model.eval()
                _edsr_models[step_scale] = model
            check_cancellation(request_id)
            inp = pil_to_tensor(img_in)
            if torch.cuda.is_available():
                inp = inp.contiguous(memory_format=torch.channels_last)
            inp = inp.to(device, non_blocking=True)
            # 直接分块推理，自动选 tile 尺寸
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                preds = _tile_infer(model, inp, step_scale, tile=_choose_tile(), request_id=request_id)
            out = to_pil_image(preds.squeeze(0).clamp(0, 1).float().cpu())
            return out
        else:
            raise Exception(f"Unknown backend: {backend_name}")
    
    def upscale_multi(img_in, backend_name, total_scale, request_id):
        """级联放大：将总倍数分解为多次放大"""
        steps = plan_steps(total_scale, BACKEND_STEPS[backend_name])
        out = img_in
        for s in steps:
            check_cancellation(request_id)
            # 每步前做输入软限，以减少 OOM
            img2 = cap_input_for_scale(out, s)
            try:
                out = apply_once(img2, backend_name, s)
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {"success": False, "error": "CUDA out of memory. Reduce scale or image size.", "status_code": 507}
        return out
    
    def run_upscale():
        try:
            check_cancellation(request_id)
            
            # 使用级联放大
            result = upscale_multi(img, backend, scale, request_id)
            if isinstance(result, dict) and not result.get("success", True):
                return result
            
            out = result
            
            # 应用最终尺寸限制
            tw, th = target_size(orig_w, orig_h, scale, MAX_OUT_LONG_EDGE, ENFORCE_OUTPUT_CAP)
            out = out.resize((tw, th), LANCZOS)
            
            # 统一处理 alpha 通道
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
            os.replace(tmp_path, output_path)  # atomic on POSIX and Windows
            out_w, out_h = out.size
            
            tgt_long = max(orig_w, orig_h) * scale
            warning = None
            capped = False
            if ENFORCE_OUTPUT_CAP and tgt_long > MAX_OUT_LONG_EDGE:
                warning = f"Final output capped to long edge {MAX_OUT_LONG_EDGE}px (requested {tgt_long}px)."
                capped = True
            
            return {"success": True, "output": out, "output_path": output_path, "output_filename": output_filename, "output_size": [out_w, out_h], "warning": warning, "capped": capped, "filename": filename}
        except CancellationError:
            raise
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    try:
        future = _executor.submit(run_upscale)
        
        # Poll for result with cancellation checks
        from concurrent.futures import TimeoutError as FutureTimeoutError
        while True:
            try:
                result = future.result(timeout=0.5)
                break
            except FutureTimeoutError:
                # Check if cancelled while waiting
                if request_id and _cancellation_flags.get(request_id, threading.Event()).is_set():
                    # Try to get result if it completed, otherwise return cancelled
                    try:
                        result = future.result(timeout=0.1)
                        break
                    except (FutureTimeoutError, Exception):
                        if request_id:
                            _cancellation_flags.pop(request_id, None)
                        return JSONResponse({"error": "Processing cancelled by user", "cancelled": True}, status_code=499)
        
        if request_id:
            _cancellation_flags.pop(request_id, None)
        
        if not result["success"]:
            status_code = result.get("status_code", 400)
            return JSONResponse({"error": result["error"], "success": False}, status_code=status_code)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        processing_time = time.time() - start_time
        
        # Calculate effective scale
        eff_w, eff_h = result["output_size"]
        eff_scale = round(max(eff_w / orig_w, eff_h / orig_h), 3)
        
        # Compute metrics after timer stops (not included in processing_time)
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
            _cancellation_flags.pop(request_id, None)
        return JSONResponse({"error": "Processing cancelled by user", "cancelled": True}, status_code=499)
    except Exception as e:
        if request_id:
            _cancellation_flags.pop(request_id, None)
        return JSONResponse({"error": str(e), "success": False}, status_code=500)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
