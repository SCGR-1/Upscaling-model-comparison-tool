"""Image processing and upscaling functions."""
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from config import device, LANCZOS, MAX_OUT_LONG_EDGE, BACKEND_STEPS
from utils import cap_input_for_scale, _choose_tile, plan_steps
from models import get_realesrgan_model, get_pan_model, get_edsr_model, get_swinir_model

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

def _swinir_infer(model, tensor, scale, request_id=None, check_cancellation=None):
    """SwinIR inference with proper padding for window-based architecture."""
    if check_cancellation is None:
        def check_cancellation(x): pass
    
    # Verify input tensor range (should be [0, 1])
    input_min, input_max = tensor.min().item(), tensor.max().item()
    if input_min < -0.1 or input_max > 1.1:
        raise ValueError(f"SwinIR input tensor out of expected range [0,1]: min={input_min:.4f}, max={input_max:.4f}")
    
    _, _, H, W = tensor.shape
    window_size = 8  # SwinIR window size
    
    # For large images, use tiling with proper padding
    # Estimate if we need tiling (roughly > 1024px on a side)
    tile_size = 512
    if H > 1024 or W > 1024:
        # Use tiling approach with padding
        out_h, out_w = H * scale, W * scale
        out_acc = torch.zeros(3, out_h, out_w, dtype=torch.float32)
        w_acc = torch.zeros(3, out_h, out_w, dtype=torch.float32)
        overlap = 32
        step = max(window_size, tile_size - overlap)
        
        for ys in range(0, H, step):
            for xs in range(0, W, step):
                check_cancellation(request_id)
                ye, xe = min(ys + tile_size, H), min(xs + tile_size, W)
                patch = tensor[:, :, ys:ye, xs:xe]
                
                # Pad patch to be divisible by window_size
                ph, pw = patch.shape[-2:]
                mod_pad_h = (window_size - ph % window_size) % window_size
                mod_pad_w = (window_size - pw % window_size) % window_size
                if mod_pad_h > 0 or mod_pad_w > 0:
                    patch = torch.nn.functional.pad(patch, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
                
                # Run inference
                # Note: SwinIR may not work well with autocast, so we use full precision
                with torch.inference_mode():
                    # Ensure patch is on correct device and dtype
                    patch_float = patch.float()
                    if patch_float.device != next(model.parameters()).device:
                        patch_float = patch_float.to(next(model.parameters()).device)
                    pred = model(patch_float)
                    # Ensure output is in correct format (1, C, H, W)
                    if pred.dim() == 3:
                        pred = pred.unsqueeze(0)
                    # Convert to float32 for consistency and ensure on CPU for accumulation
                    pred = pred.float()
                
                # Crop back to original patch size
                pred = pred[:, :, :ph*scale, :pw*scale]
                # Ensure values are in [0, 1] range
                pred = pred.clamp(0, 1)
                
                # Diagnostic: check output range (only for first patch to avoid spam)
                if ys == 0 and xs == 0:
                    pred_min, pred_max = pred.min().item(), pred.max().item()
                    pred_mean = pred.mean().item()
                    print(f"[SwinIR Debug] First patch output: min={pred_min:.4f}, max={pred_max:.4f}, mean={pred_mean:.4f}, shape={pred.shape}")
                    if pred_min < -0.1 or pred_max > 1.1 or (pred_max - pred_min) < 0.01:
                        import warnings
                        warnings.warn(f"SwinIR output range suspicious: min={pred_min:.4f}, max={pred_max:.4f}")
                
                # Blend with Hann window
                oy, ox = ys * scale, xs * scale
                ph_out, pw_out = pred.shape[-2:]
                
                # Ensure pred is on CPU for accumulation
                pred_cpu = pred.float().cpu()
                # Remove batch dimension if present: (1, C, H, W) -> (C, H, W)
                if pred_cpu.dim() == 4:
                    pred_cpu = pred_cpu.squeeze(0)
                
                wmask = _hann2d(ph_out, pw_out, torch.device('cpu'), torch.float32)
                if wmask.dim() == 2:
                    wmask = wmask.unsqueeze(0).expand(3, -1, -1)  # (H, W) -> (C, H, W)
                elif wmask.dim() == 3 and wmask.shape[0] == 1:
                    wmask = wmask.expand(3, -1, -1)  # (1, H, W) -> (C, H, W)
                
                predw = pred_cpu * wmask
                out_acc[:, oy:oy+ph_out, ox:ox+pw_out] += predw
                w_acc[:, oy:oy+ph_out, ox:ox+pw_out] += wmask
        
        # Normalize by accumulated weights, ensuring no division by zero
        out = out_acc / torch.clamp_min(w_acc, 1e-6)
        # Ensure output is in [0, 1] range
        out = out.clamp(0, 1)
        
        # Diagnostic: check final accumulated output
        out_min, out_max = out.min().item(), out.max().item()
        out_mean = out.mean().item()
        print(f"[SwinIR Debug] Tiled final output: min={out_min:.4f}, max={out_max:.4f}, mean={out_mean:.4f}, shape={out.shape}")
        if out_max < 0.01:
            import warnings
            warnings.warn(f"SwinIR tiled output appears black: min={out_min:.4f}, max={out_max:.4f}")
        
        return out.unsqueeze(0)  # Add batch dimension: CxHxW -> 1xCxHxW
    else:
        # For smaller images, process directly with padding
        mod_pad_h = (window_size - H % window_size) % window_size
        mod_pad_w = (window_size - W % window_size) % window_size
        
        if mod_pad_h > 0 or mod_pad_w > 0:
            tensor = torch.nn.functional.pad(tensor, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        
        check_cancellation(request_id)
        # Note: SwinIR may not work well with autocast, so we use full precision
        with torch.inference_mode():
            # Ensure tensor is on correct device and dtype
            tensor_float = tensor.float()
            if tensor_float.device != next(model.parameters()).device:
                tensor_float = tensor_float.to(next(model.parameters()).device)
            pred = model(tensor_float)
            # Ensure output is in correct format (1, C, H, W)
            if pred.dim() == 3:
                pred = pred.unsqueeze(0)
            # Convert to float32 for consistency
            pred = pred.float()
        
        # Diagnostic: check output range
        pred_min, pred_max = pred.min().item(), pred.max().item()
        pred_mean = pred.mean().item()
        print(f"[SwinIR Debug] Direct output: min={pred_min:.4f}, max={pred_max:.4f}, mean={pred_mean:.4f}, shape={pred.shape}")
        if pred_min < -0.1 or pred_max > 1.1 or (pred_max - pred_min) < 0.01:
            import warnings
            warnings.warn(f"SwinIR output range suspicious: min={pred_min:.4f}, max={pred_max:.4f}")
        
        out_h, out_w = H * scale, W * scale
        if mod_pad_h > 0 or mod_pad_w > 0:
            pred = pred[:, :, :out_h, :out_w]
        
        # Ensure values are in [0, 1] range
        pred = pred.clamp(0, 1)
        return pred.float().cpu()

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
    
    elif backend_name == "swinir":
        from models import have_swinir, _swinir_error
        if not have_swinir:
            error_msg = "SwinIR is not available. "
            if _swinir_error:
                if "numpy" in _swinir_error.lower() or "scipy" in _swinir_error.lower():
                    error_msg += "NumPy/scipy compatibility issue detected. "
                    error_msg += "Try: pip install 'numpy<2.0' or upgrade scipy. "
                elif "torchvision" in _swinir_error.lower() or "functional_tensor" in _swinir_error.lower():
                    error_msg += "Torchvision compatibility issue detected. "
                    error_msg += "Try: pip install 'torchvision>=0.15.0'. "
                error_msg += f"Error: {_swinir_error}"
            else:
                error_msg += "Install basicsr package: pip install basicsr"
            raise Exception(error_msg)
        if step_scale not in (2, 3, 4, 8):
            raise Exception("SwinIR supports scales 2,3,4,8")
        try:
            model = get_swinir_model(step_scale)
            if model is None:
                raise Exception("SwinIR model not available")
        except (ImportError, ModuleNotFoundError) as e:
            error_msg = str(e)
            if "torchvision" in error_msg.lower() or "functional_tensor" in error_msg.lower():
                raise Exception(
                    f"SwinIR model creation failed: {error_msg}\n"
                    "This is a torchvision compatibility issue. Try:\n"
                    "  pip install 'torchvision>=0.15.0'"
                ) from e
            raise
        check_cancellation(request_id)
        
        # ===== 2. Check input tensor scaling & range =====
        inp = pil_to_tensor(img_in)
        inp_min, inp_max, inp_mean = inp.min().item(), inp.max().item(), inp.mean().item()
        print(f"[SwinIR Debug] Input tensor: min={inp_min:.4f}, max={inp_max:.4f}, mean={inp_mean:.4f}, shape={inp.shape}, dtype={inp.dtype}")
        if inp_min < 0 or inp_max > 1:
            raise ValueError(f"Input tensor out of range [0,1]: min={inp_min:.4f}, max={inp_max:.4f}")
        if inp_max < 0.01:
            raise ValueError(f"Input tensor appears to be all zeros (max={inp_max:.4f})")
        
        # ===== 6. Check device/dtype consistency =====
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        print(f"[SwinIR Debug] Model device: {model_device}, dtype: {model_dtype}")
        
        if torch.cuda.is_available():
            inp = inp.contiguous(memory_format=torch.channels_last)
        inp = inp.to(device, non_blocking=True)
        inp = inp.float()  # Ensure float32
        print(f"[SwinIR Debug] Input after device move: device={inp.device}, dtype={inp.dtype}")
        
        # ===== 7. Check scale factor =====
        print(f"[SwinIR Debug] Scale factor: {step_scale}, input size: {inp.shape[-2]}x{inp.shape[-1]}")
        expected_out_h, expected_out_w = inp.shape[-2] * step_scale, inp.shape[-1] * step_scale
        print(f"[SwinIR Debug] Expected output size: {expected_out_h}x{expected_out_w}")
        
        # SwinIR requires input dimensions to be divisible by window_size (8)
        # Use specialized inference that handles padding
        preds = _swinir_infer(model, inp, step_scale, request_id=request_id, check_cancellation=check_cancellation)
        
        # ===== 1. Check output tensor values =====
        if preds.numel() == 0:
            raise ValueError(f"SwinIR output tensor is empty: shape={preds.shape}")
        
        raw_min, raw_max, raw_mean = preds.min().item(), preds.max().item(), preds.mean().item()
        raw_std = preds.std().item()
        print(f"[SwinIR Debug] Raw model output: min={raw_min:.4f}, max={raw_max:.4f}, mean={raw_mean:.4f}, std={raw_std:.4f}, shape={preds.shape}")
        
        # Check for NaNs or Infs
        if torch.isnan(preds).any():
            nan_count = torch.isnan(preds).sum().item()
            raise ValueError(f"SwinIR output contains {nan_count} NaN values")
        if torch.isinf(preds).any():
            inf_count = torch.isinf(preds).sum().item()
            raise ValueError(f"SwinIR output contains {inf_count} Inf values")
        
        # ===== 7. Inspect individual channels =====
        if preds.dim() >= 3:
            for c in range(min(3, preds.shape[-3] if preds.dim() == 3 else preds.shape[1])):
                if preds.dim() == 4:
                    ch_tensor = preds[0, c, :, :]
                else:
                    ch_tensor = preds[c, :, :]
                ch_min, ch_max, ch_mean = ch_tensor.min().item(), ch_tensor.max().item(), ch_tensor.mean().item()
                print(f"[SwinIR Debug] Channel {c}: min={ch_min:.4f}, max={ch_max:.4f}, mean={ch_mean:.4f}")
        
        # Ensure output is in [0, 1] range and properly formatted
        preds = preds.squeeze(0) if preds.dim() == 4 else preds
        # Clamp to [0, 1] range (SwinIR should output in this range already)
        preds = preds.clamp(0, 1)
        
        # ===== 1. Check output after clamp =====
        out_min, out_max, out_mean = preds.min().item(), preds.max().item(), preds.mean().item()
        print(f"[SwinIR Debug] After clamp: min={out_min:.4f}, max={out_max:.4f}, mean={out_mean:.4f}, shape={preds.shape}")
        
        # Check if output is suspiciously constant or out of range
        if out_max < 0.01:
            raise ValueError(f"SwinIR output appears to be black (max={out_max:.4f}). Check model weights and input.")
        if out_min > 0.99:
            raise ValueError(f"SwinIR output appears to be white (min={out_min:.4f}). Check model weights and input.")
        
        # ===== 8. Verify output dimensions =====
        actual_out_h, actual_out_w = preds.shape[-2], preds.shape[-1]
        print(f"[SwinIR Debug] Actual output size: {actual_out_h}x{actual_out_w}")
        if actual_out_h != expected_out_h or actual_out_w != expected_out_w:
            print(f"[SwinIR Debug] WARNING: Output size mismatch! Expected {expected_out_h}x{expected_out_w}, got {actual_out_h}x{actual_out_w}")
        
        # Ensure proper format: C x H x W for to_pil_image
        if preds.dim() == 4:
            preds = preds.squeeze(0)  # Remove batch dimension: 1xCxHxW -> CxHxW
        elif preds.dim() != 3:
            raise ValueError(f"Unexpected SwinIR output shape: {preds.shape}, expected 3D (CxHxW) or 4D (1xCxHxW)")
        
        # Ensure values are in [0, 1] and convert to PIL
        preds = preds.clamp(0, 1).float().cpu()
        
        # ===== 4. Check output conversion to image =====
        # Convert to numpy to verify before PIL conversion
        out_np = preds.permute(1, 2, 0).cpu().numpy()  # CxHxW -> HxWxC
        out_np_uint8 = (out_np * 255.0).astype(np.uint8)
        print(f"[SwinIR Debug] NumPy array (float): min={out_np.min():.4f}, max={out_np.max():.4f}, mean={out_np.mean():.4f}")
        print(f"[SwinIR Debug] NumPy array (uint8): min={out_np_uint8.min()}, max={out_np_uint8.max()}, mean={out_np_uint8.mean():.2f}")
        
        if out_np_uint8.max() == 0:
            raise ValueError(f"Output numpy array is all zeros (max={out_np_uint8.max()})")
        
        out = to_pil_image(preds)
        
        # ===== 4. Verify PIL image =====
        out_arr = np.array(out)
        if out_arr.size > 0:
            pil_min, pil_max, pil_mean = out_arr.min(), out_arr.max(), out_arr.mean()
            print(f"[SwinIR Debug] PIL image: min={pil_min}, max={pil_max}, mean={pil_mean:.2f}, shape={out_arr.shape}, dtype={out_arr.dtype}")
            if pil_max == 0:
                raise ValueError(f"PIL image is all zeros (max={pil_max})")
        
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

