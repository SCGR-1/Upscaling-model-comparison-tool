"""Utility functions."""
from pathlib import Path
from fastapi import HTTPException
import torch
from config import MAX_OUT_LONG_EDGE, LANCZOS, BACKEND_STEPS

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

def target_size(orig_w, orig_h, scale, cap, enforce=True):
    """Calculate target size with optional capping."""
    tw, th = orig_w * scale, orig_h * scale
    if not enforce:
        return tw, th
    long_edge = max(tw, th)
    if long_edge <= cap:
        return tw, th
    r = cap / long_edge
    return max(1, int(tw * r)), max(1, int(th * r))

def safe_join(base: Path, name: str) -> Path:
    """Safely join path, preventing directory traversal."""
    p = (base / Path(name).name).resolve()
    if base.resolve() not in p.parents and base.resolve() != p:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return p

def _etag_for(path: Path) -> str:
    """Generate ETag for file."""
    st = path.stat()
    return f'W/"{st.st_mtime_ns}-{st.st_size}"'

def cap_input_for_scale(img, scale):
    """Cap input image size for given scale."""
    w, h = img.size
    tgt_long = max(w, h) * scale
    if tgt_long <= MAX_OUT_LONG_EDGE:
        return img
    s = MAX_OUT_LONG_EDGE / tgt_long
    nw, nh = max(64, int(w * s)), max(64, int(h * s))
    return img.resize((nw, nh), LANCZOS)

def _choose_tile(target_bytes=256<<20):  # ~256MB budget
    """Choose tile size based on available memory."""
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        budget = max(64<<20, int(0.2 * free))
    else:
        budget = target_bytes
    side = int((budget / (3 * 4)) ** 0.5)  # 3ch * float32
    return max(192, min(768, side))

