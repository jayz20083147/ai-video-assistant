#!/usr/bin/env python3
"""
Environment and system dependency verification script.
Checks FFmpeg availability, Python version, and hardware acceleration support.
"""

import sys
import shutil
import subprocess

def check_python_version() -> bool:
    print(f"[+] Python Version: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  [!] WARNING: Python 3.10+ recommended.")
        return False
    print("  [OK] Python version compatible.")
    return True

def check_ffmpeg() -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    
    if not ffmpeg_path or not ffprobe_path:
        print("  [X] ERROR: FFmpeg or FFprobe not found in PATH.")
        print("      macOS: brew install ffmpeg")
        print("      Ubuntu/Debian: sudo apt update && sudo apt install -y ffmpeg")
        return False
    
    try:
        res = subprocess.run([ffmpeg_path, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        first_line = res.stdout.splitlines()[0] if res.stdout else "Unknown version"
        print(f"[+] FFmpeg: {ffmpeg_path} ({first_line})")
        print(f"[+] FFprobe: {ffprobe_path}")
        print("  [OK] FFmpeg and FFprobe verified.")
        return True
    except Exception as e:
        print(f"  [X] Error executing FFmpeg: {e}")
        return False

def check_hardware_acceleration():
    print("[+] Checking hardware acceleration:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  [OK] Apple Silicon MPS (Metal Performance Shaders) detected.")
        else:
            print("  [i] CPU mode active (No CUDA/MPS detected).")
    except ImportError:
        print("  [i] PyTorch not yet installed. Hardware acceleration check deferred.")

def main():
    print("=" * 60)
    print("  AI Video Assistant — Environment Verification")
    print("=" * 60)
    py_ok = check_python_version()
    ffmpeg_ok = check_ffmpeg()
    check_hardware_acceleration()
    print("=" * 60)
    
    if py_ok and ffmpeg_ok:
        print(">> System environment is READY for AI Video Assistant.")
        return 0
    else:
        print(">> System environment requires attention before running.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
