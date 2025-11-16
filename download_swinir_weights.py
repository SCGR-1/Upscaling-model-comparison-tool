"""Download SwinIR pretrained weights from official releases."""
import os
import sys
from pathlib import Path
import urllib.request
from urllib.error import URLError

# SwinIR model weights URLs (from official GitHub releases)
SWINIR_WEIGHTS = {
    2: "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth",
    3: "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth",
    4: "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
    8: "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth",
}

def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress indication."""
    print(f"Downloading {desc}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")
    
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded * 100) / total_size) if total_size > 0 else 0
            print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest, reporthook=show_progress)
        print("\n  ✓ Download complete!")
        return True
    except URLError as e:
        print(f"\n  ✗ Download failed: {e}")
        return False
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return False

def main():
    """Download SwinIR weights."""
    weights_dir = Path("weights") / "swinir"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SwinIR Weight Downloader")
    print("=" * 60)
    print(f"Weights will be saved to: {weights_dir.absolute()}")
    print()
    
    if len(sys.argv) > 1:
        # Download specific scales
        scales = []
        for arg in sys.argv[1:]:
            try:
                scale = int(arg)
                if scale in SWINIR_WEIGHTS:
                    scales.append(scale)
                else:
                    print(f"Warning: Scale {scale} not supported. Supported scales: {list(SWINIR_WEIGHTS.keys())}")
            except ValueError:
                print(f"Warning: Invalid scale '{arg}', skipping")
        
        if not scales:
            print("No valid scales specified. Downloading all scales...")
            scales = list(SWINIR_WEIGHTS.keys())
    else:
        # Download all scales
        print("No scales specified. Downloading all available scales (2x, 3x, 4x, 8x)...")
        scales = list(SWINIR_WEIGHTS.keys())
    
    print()
    print(f"Scales to download: {scales}")
    print()
    
    success_count = 0
    skip_count = 0
    
    for scale in scales:
        weight_file = weights_dir / f"001_classicalSR_DIV2K_s48w8_SwinIR-M_x{scale}.pth"
        
        if weight_file.exists():
            size_mb = weight_file.stat().st_size / (1024 * 1024)
            print(f"Scale {scale}x: Already exists ({size_mb:.1f} MB), skipping...")
            skip_count += 1
            continue
        
        url = SWINIR_WEIGHTS[scale]
        if download_file(url, weight_file, f"SwinIR x{scale} model"):
            success_count += 1
        print()
    
    print("=" * 60)
    print("Summary:")
    print(f"  Downloaded: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Total: {len(scales)}")
    print("=" * 60)
    
    if success_count > 0:
        print("\n✓ SwinIR weights are ready to use!")
    elif skip_count > 0:
        print("\n✓ All requested weights are already downloaded.")
    else:
        print("\n✗ No weights were downloaded. Check your internet connection and try again.")

if __name__ == "__main__":
    main()

