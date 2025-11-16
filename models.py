"""Model loading and management."""
import torch
import torch.nn as nn
from pathlib import Path
from config import device

# Patch torchvision.transforms for compatibility before any imports
# This fixes the functional_tensor import issue in newer torchvision versions
def _patch_torchvision_transforms():
    """Patch torchvision.transforms to add functional_tensor alias."""
    try:
        import torchvision.transforms as transforms
        if not hasattr(transforms, 'functional_tensor'):
            try:
                from torchvision.transforms import _functional_tensor
                transforms.functional_tensor = _functional_tensor
                # Also patch in sys.modules to ensure it's visible to dynamically loaded modules
                import sys
                if 'torchvision.transforms' in sys.modules:
                    sys.modules['torchvision.transforms'].functional_tensor = _functional_tensor
            except ImportError:
                try:
                    from torchvision.transforms import functional
                    transforms.functional_tensor = functional
                    import sys
                    if 'torchvision.transforms' in sys.modules:
                        sys.modules['torchvision.transforms'].functional_tensor = functional
                except ImportError:
                    pass
    except Exception:
        pass  # If patching fails, continue anyway

# Apply patch immediately
_patch_torchvision_transforms()

# Optional model imports
have_realesrgan = have_pan = have_edsr = have_swinir = False
RealESRGANModel = None
PanModel = None
EdsrModel = None
SwinIRModel = None

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

# SwinIR - try basicsr first, then fallback to manual implementation
_swinir_error = None
try:
    # Try to import SwinIR architecture directly using importlib to avoid basicsr.__init__ issues
    import importlib.util
    import sys
    basicsr_path = None
    for p in sys.path:
        potential = Path(p) / "basicsr" / "archs" / "swinir_arch.py"
        if potential.exists():
            basicsr_path = potential
            break
    
    if basicsr_path:
        # Temporarily add basicsr parent to path, then restore
        basicsr_parent = str(basicsr_path.parent.parent)
        original_path = sys.path.copy()
        try:
            if basicsr_parent not in sys.path:
                sys.path.insert(0, basicsr_parent)
            
            # Ensure torchvision patch is applied before loading the module
            _patch_torchvision_transforms()
            
            # Pre-import and patch torchvision to ensure it's available when module loads
            import torchvision
            import torchvision.transforms
            _patch_torchvision_transforms()  # Apply again after import
            
            # Also create a mock functional_tensor module if needed
            # This allows "from torchvision.transforms import functional_tensor" to work
            if 'torchvision.transforms.functional_tensor' not in sys.modules:
                try:
                    from torchvision.transforms import _functional_tensor
                    # Create a mock module for functional_tensor
                    import types
                    functional_tensor_module = types.ModuleType('torchvision.transforms.functional_tensor')
                    functional_tensor_module.__dict__.update(_functional_tensor.__dict__)
                    functional_tensor_module.__file__ = getattr(_functional_tensor, '__file__', '')
                    functional_tensor_module.__package__ = 'torchvision.transforms'
                    sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor_module
                except (ImportError, AttributeError):
                    pass
            
            spec = importlib.util.spec_from_file_location("basicsr.archs.swinir_arch", basicsr_path)
            swinir_module = importlib.util.module_from_spec(spec)
            # Set package context for relative imports
            swinir_module.__package__ = 'basicsr.archs'
            swinir_module.__name__ = 'basicsr.archs.swinir_arch'
            # Ensure parent packages exist and are properly configured
            import types
            if 'basicsr' not in sys.modules:
                basicsr_pkg = types.ModuleType('basicsr')
                basicsr_pkg.__path__ = [str(basicsr_path.parent.parent)]
                sys.modules['basicsr'] = basicsr_pkg
            if 'basicsr.archs' not in sys.modules:
                basicsr_archs = types.ModuleType('basicsr.archs')
                basicsr_archs.__path__ = [str(basicsr_path.parent)]
                basicsr_archs.__package__ = 'basicsr.archs'
                sys.modules['basicsr.archs'] = basicsr_archs
            # Register the module in sys.modules before execution so relative imports work
            sys.modules['basicsr.archs.swinir_arch'] = swinir_module
            try:
                spec.loader.exec_module(swinir_module)
                SwinIRModel = swinir_module.SwinIR
                have_swinir = True
            except (ImportError, ModuleNotFoundError) as e:
                error_msg = str(e)
                if "torchvision" in error_msg.lower() or "functional_tensor" in error_msg.lower():
                    _swinir_error = f"torchvision compatibility: {error_msg}"
                    raise ImportError(
                        f"SwinIR import failed: {error_msg}\n"
                        "This is a torchvision compatibility issue. Try:\n"
                        "  pip install 'torchvision>=0.15.0' or patch torchvision.transforms"
                    ) from e
                raise
        finally:
            # Restore original sys.path to avoid affecting other imports
            sys.path[:] = original_path
    else:
        # Fallback: try normal import
        from basicsr.archs.swinir_arch import SwinIR as SwinIRModel
        have_swinir = True
except Exception as e:
    _swinir_error = str(e)
    # Fallback: try importing from local swinir module if available
    try:
        swinir_path = Path(__file__).parent / "swinir"
        if swinir_path.exists():
            sys.path.insert(0, str(swinir_path.parent))
            from swinir.models.network_swinir import SwinIR as SwinIRModel
            have_swinir = True
            _swinir_error = None
    except Exception as e2:
        if _swinir_error is None:
            _swinir_error = str(e2)

# Model caches
_pan_models = {}       # {scale: int -> PanModel}
_edsr_models = {}      # {scale: int -> EdsrModel}
_swinir_models = {}    # {scale: int -> SwinIRModel}
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

def get_swinir_model(scale: int):
    """Get or create SwinIR model for given scale."""
    if scale not in _swinir_models:
        if not have_swinir:
            return None
        
        # Check if we have a stored error that prevents model creation
        global _swinir_error
        if _swinir_error and ("torchvision" in _swinir_error.lower() or "functional_tensor" in _swinir_error.lower()):
            raise ImportError(
                f"SwinIR model creation failed: {_swinir_error}\n"
                "This is likely a torchvision compatibility issue. Try:\n"
                "  pip install 'torchvision>=0.15.0' or downgrade basicsr"
            )
        
        # SwinIR model parameters
        # For classical SR: img_size=48, window_size=8, depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6]
        # For lightweight: img_size=64, window_size=8, depths=[6,6,6,6], embed_dim=60, num_heads=[6,6,6,6]
        # We'll use the medium model (classical SR) as default
        
        if scale in [2, 3, 4, 8]:
            # Model configuration for classical SR
            img_size = 48
            window_size = 8
            depths = [6, 6, 6, 6, 6, 6]
            embed_dim = 180
            num_heads = [6, 6, 6, 6, 6, 6]
            mlp_ratio = 2.0
            upsampler = 'pixelshuffle'
            resi_connection = '1conv'
            
            try:
                model = SwinIRModel(
                    upscale=scale,
                    in_chans=3,
                    img_size=img_size,
                    window_size=window_size,
                    img_range=1.0,
                    depths=depths,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    upsampler=upsampler,
                    resi_connection=resi_connection
                )
            except (ImportError, ModuleNotFoundError) as e:
                error_msg = str(e)
                if "torchvision" in error_msg.lower() or "functional_tensor" in error_msg.lower():
                    raise ImportError(
                        f"SwinIR model creation failed: {error_msg}\n"
                        "This is a torchvision compatibility issue. Try:\n"
                        "  pip install 'torchvision>=0.15.0'"
                    ) from e
                raise
            
            # Load weights
            weights_dir = Path("weights") / "swinir"
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            # Weight file naming: 001_classicalSR_DIV2K_s48w8_SwinIR-M_x{scale}.pth
            weight_file = weights_dir / f"001_classicalSR_DIV2K_s48w8_SwinIR-M_x{scale}.pth"
            
            if not weight_file.exists():
                raise FileNotFoundError(
                    f"SwinIR weights not found: {weight_file}\n"
                    f"Please download from: https://github.com/JingyunLiang/SwinIR/releases\n"
                    f"Or use the download script: python download_swinir_weights.py"
                )
            
            # Load weights
            checkpoint = torch.load(weight_file, map_location='cpu')
            try:
                if 'params_ema' in checkpoint:
                    model.load_state_dict(checkpoint['params_ema'], strict=True)
                elif 'params' in checkpoint:
                    model.load_state_dict(checkpoint['params'], strict=True)
                else:
                    model.load_state_dict(checkpoint, strict=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load SwinIR weights from {weight_file}: {e}") from e
            
            # ===== 3. Check model weight loading =====
            # Verify model has loaded weights (check parameter count)
            param_count = sum(p.numel() for p in model.parameters())
            param_count_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[SwinIR Debug] Model parameters: total={param_count:,}, trainable={param_count_trainable:,}")
            if param_count == 0:
                raise RuntimeError(f"SwinIR model has no parameters after loading weights. Check weight file: {weight_file}")
            
            # Check a few sample parameters to verify they're not all zeros
            sample_params = list(model.parameters())[:5]
            for i, param in enumerate(sample_params):
                param_min, param_max = param.min().item(), param.max().item()
                param_mean = param.mean().item()
                print(f"[SwinIR Debug] Sample param {i}: shape={param.shape}, min={param_min:.6f}, max={param_max:.6f}, mean={param_mean:.6f}")
            
            model = model.to(device)
            model.eval()
            print(f"[SwinIR Debug] Model moved to device: {device}, eval mode: {not model.training}")
            _swinir_models[scale] = model
        else:
            raise ValueError(f"SwinIR supports scales 2, 3, 4, 8, got {scale}")
    
    return _swinir_models.get(scale)

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
    
    if have_swinir:
        try:
            # Try to warmup x4 model (most common)
            get_swinir_model(4)
            print("[Warmup] SwinIR ready")
        except Exception as e:
            print(f"[Warmup] SwinIR skipped: {e}")

