"""Image processing and upscaling functions."""
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from config import device, LANCZOS, MAX_OUT_LONG_EDGE, BACKEND_STEPS
from utils import cap_input_for_scale, _choose_tile, plan_steps
from models import get_realesrgan_model, get_pan_model, get_edsr_model

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to PyTorch tensor."""
    arr = np.array(img).astype(np.float32) / 255.0  # HWC, 0..1
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1xCxHxW
    return t

def _hann2d(h, w, device, dtype):
    """Create 2D Hann window for blending."""
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype).unsqueeze(1)
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype).unsqueeze(0)
    return (wy * wx).clamp_(min=1e-6)

def _tile_infer(model, tensor, scale, tile=512, overlap=32, request_id=None, check_cancellation=None):
    """Tile-based inference to handle large images."""
    if check_cancellation is None:
        def check_cancellation(x): pass
    
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

def apply_once(img_in, backend_name, step_scale, request_id, check_cancellation):
    """Single upscale step, encapsulating each backend's logic."""
    if backend_name == "realesrgan":
        from models import have_realesrgan
        if not have_realesrgan:
            raise Exception("Install py-real-esrgan")
        model = get_realesrgan_model()
        check_cancellation(request_id)
        # soft cap before running 4x
        img2 = cap_input_for_scale(img_in, 4)
        try:
            out = model.predict(img2)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        return out
    
    elif backend_name == "pan":
        from models import have_pan
        if not have_pan:
            raise Exception("Install super-image")
        if step_scale not in (2, 3, 4):
            raise Exception("PAN supports scales 2,3,4")
        model = get_pan_model(step_scale)
        if model is None:
            raise Exception("PAN model not available")
        check_cancellation(request_id)
        inp = pil_to_tensor(img_in)
        if torch.cuda.is_available():
            inp = inp.contiguous(memory_format=torch.channels_last)
        inp = inp.to(device, non_blocking=True)
        # 直接分块推理，自动选 tile 尺寸
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
            preds = _tile_infer(model, inp, step_scale, tile=_choose_tile(), request_id=request_id, check_cancellation=check_cancellation)
        out = to_pil_image(preds.squeeze(0).clamp(0, 1).float().cpu())
        return out
    
    elif backend_name == "edsr":
        from models import have_edsr
        if not have_edsr:
            raise Exception("Install super-image")
        if step_scale not in (2, 3, 4):
            raise Exception("EDSR supports scales 2,3,4")
        model = get_edsr_model(step_scale)
        if model is None:
            raise Exception("EDSR model not available")
        check_cancellation(request_id)
        inp = pil_to_tensor(img_in)
        if torch.cuda.is_available():
            inp = inp.contiguous(memory_format=torch.channels_last)
        inp = inp.to(device, non_blocking=True)
        # 直接分块推理，自动选 tile 尺寸
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
            preds = _tile_infer(model, inp, step_scale, tile=_choose_tile(), request_id=request_id, check_cancellation=check_cancellation)
        out = to_pil_image(preds.squeeze(0).clamp(0, 1).float().cpu())
        return out
    else:
        raise Exception(f"Unknown backend: {backend_name}")

def upscale_multi(img_in, backend_name, total_scale, request_id, check_cancellation):
    """级联放大：将总倍数分解为多次放大"""
    steps = plan_steps(total_scale, BACKEND_STEPS[backend_name])
    out = img_in
    for s in steps:
        check_cancellation(request_id)
        # 每步前做输入软限，以减少 OOM
        img2 = cap_input_for_scale(out, s)
        try:
            out = apply_once(img2, backend_name, s, request_id, check_cancellation)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"success": False, "error": "CUDA out of memory. Reduce scale or image size.", "status_code": 507}
    return out

