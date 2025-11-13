"""Model loading and management."""
import torch
from config import device

# Optional model imports
have_realesrgan = have_pan = have_edsr = False
RealESRGANModel = None
PanModel = None
EdsrModel = None

try:
    from py_real_esrgan.model import RealESRGAN as RealESRGANModel
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

# Model caches
_pan_models = {}       # {scale: int -> PanModel}
_edsr_models = {}      # {scale: int -> EdsrModel}
_realesrgan_model = None

def get_realesrgan_model():
    """Get or create Real-ESRGAN model."""
    global _realesrgan_model
    if _realesrgan_model is None and have_realesrgan:
        _realesrgan_model = RealESRGANModel(device, scale=4)
        _realesrgan_model.load_weights("weights/RealESRGAN_x4.pth", download=True)
    return _realesrgan_model

def get_pan_model(scale: int):
    """Get or create PAN model for given scale."""
    if scale not in _pan_models:
        if have_pan:
            model = PanModel.from_pretrained("eugenesiow/pan-bam", scale=scale).to(device)
            model.eval()
            _pan_models[scale] = model
        else:
            return None
    return _pan_models.get(scale)

def get_edsr_model(scale: int):
    """Get or create EDSR model for given scale."""
    if scale not in _edsr_models:
        if have_edsr:
            model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale).to(device)
            model.eval()
            _edsr_models[scale] = model
        else:
            return None
    return _edsr_models.get(scale)

def warmup_models():
    """Warmup models at startup."""
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    
    if have_realesrgan and torch.cuda.is_available():
        try:
            get_realesrgan_model()
            print("[Warmup] Real-ESRGAN ready")
        except Exception as e:
            print(f"[Warmup] Real-ESRGAN skipped: {e}")

