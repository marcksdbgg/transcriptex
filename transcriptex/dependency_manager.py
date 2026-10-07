"""
dependency_manager.py — Auto-install missing Python packages into the project's venv.

This module is imported by commands that need heavy optional dependencies
(faster-whisper, torch, pyaudiowpatch, etc.).  It never mutates the *system*
Python; it always targets the interpreter that is currently running
transcriptex (usually ``.venv\\Scripts\\python.exe`` on Windows).
"""
from __future__ import annotations

import importlib
import csv
import os
import platform
import shutil
import subprocess
import sys
import ctypes
from pathlib import Path
from typing import Iterable

_NVIDIA_CUDA_PACKAGES = (
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cuda-nvrtc-cu12",
)
_CUDA_DLL_DIRS_PREPENDED = False


def _has_nvidia_gpu() -> bool:
    """Light-weight GPU probe that does **not** need torch."""
    if sys.platform == "win32":
        try:
            ctypes.CDLL("nvcuda.dll")
            return True
        except OSError:
            return False
    # Linux / macOS
    return shutil.which("nvidia-smi") is not None


def _nvidia_site_package_dirs() -> list[Path]:
    """Return nvidia/*/bin (Windows) or nvidia/*/lib (Linux) paths from site-packages."""
    lib_subdir = "bin" if platform.system() == "Windows" else "lib"
    paths: list[Path] = []
    seen: set[Path] = set()

    for site_dir in {Path(p) for p in sys.path if p}:
        nvidia_root = site_dir / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for pkg_dir in sorted(nvidia_root.iterdir()):
            lib_dir = pkg_dir / lib_subdir
            if lib_dir.is_dir() and lib_dir not in seen:
                seen.add(lib_dir)
                paths.append(lib_dir)

    return paths


def prepend_nvidia_cuda_paths() -> None:
    """
    Expose NVIDIA pip runtime DLLs to CTranslate2 on Windows.

    CTranslate2 loads cublas/cudnn via native LoadLibrary, which does not
    always honor os.add_dll_directory(). Prepending PATH is required.
    """
    global _CUDA_DLL_DIRS_PREPENDED
    if _CUDA_DLL_DIRS_PREPENDED:
        return

    paths = _nvidia_site_package_dirs()
    if not paths:
        return

    path_prefix = os.pathsep.join(str(p) for p in paths)
    os.environ["PATH"] = path_prefix + os.pathsep + os.environ.get("PATH", "")

    if sys.platform == "win32":
        for lib_dir in paths:
            try:
                os.add_dll_directory(str(lib_dir))
            except (OSError, AttributeError):
                pass

    _CUDA_DLL_DIRS_PREPENDED = True


def ensure_nvidia_cuda_runtime() -> None:
    """Install CUDA 12 runtime wheels when a GPU is present."""
    if not _has_nvidia_gpu():
        return

    if not _nvidia_site_package_dirs():
        ensure(*_NVIDIA_CUDA_PACKAGES)

    prepend_nvidia_cuda_paths()


def cuda_available() -> bool:
    """
    Return True when CUDA can be used for faster-whisper inference.

    faster-whisper runs on CTranslate2, not PyTorch, so we probe CT2 first.
    This avoids false negatives when torch is installed as a CPU-only wheel
    (common on Python versions without official CUDA torch builds).
    """
    if _has_nvidia_gpu():
        try:
            ensure_nvidia_cuda_runtime()
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return True
        except Exception:
            pass

    return False


def nvidia_gpu_info() -> tuple[str, float] | None:
    """Read device-zero VRAM without importing or installing PyTorch."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
        name, memory = next(csv.reader(result.stdout.splitlines()))
        return name.strip(), float(memory.strip())
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration):
        return None


def _pip_install(
    packages: Iterable[str],
    *,
    index_url: str | None = None,
    extra_index_url: str | None = None,
) -> None:
    """Run `python -m pip install <packages>` in the current interpreter."""
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    if index_url:
        cmd += ["--index-url", index_url]
    if extra_index_url:
        cmd += ["--extra-index-url", extra_index_url]

    print(f"[dep-mgr] Installing missing packages: {', '.join(packages)} ...", file=sys.stderr)
    result = subprocess.run(cmd, check=False, stdout=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to install packages: {list(packages)}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Make sure you have internet access and write permissions to the environment."
        )


def _try_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def ensure(
    *specs: str,
    auto_install: bool = True,
) -> None:
    """
    Ensure one or more packages are importable.

    Each *spec* is either:
        "import_name"               # pip name == import name
        "import_name:pip_name"     # different pip name (e.g. "faster_whisper:faster-whisper")

    If a package is missing and *auto_install* is True, the function will call
    `python -m pip install <pip_name>` automatically.  If *auto_install* is
    False (or installation fails) a `RuntimeError` is raised.
    """
    missing_pip: list[str] = []
    missing_human: list[str] = []

    for spec in specs:
        if ":" in spec:
            import_name, pip_name = spec.split(":", 1)
        else:
            import_name = pip_name = spec

        if not _try_import(import_name):
            missing_pip.append(pip_name)
            missing_human.append(spec)

    if not missing_pip:
        return

    if not auto_install:
        raise RuntimeError(
            f"Missing required packages: {missing_pip}. "
            f"Run: {sys.executable} -m pip install {' '.join(missing_pip)}"
        )

    # Special-case torch so we pick the correct wheel index.
    torch_specs = [p for p in missing_pip if p == "torch" or p.startswith("torch")]
    other_specs = [p for p in missing_pip if p not in torch_specs]

    if other_specs:
        _pip_install(other_specs)

    for pkg in torch_specs:
        installed = False
        # Try CUDA 12.1 first, then CUDA 12.4, then CPU as fallback
        indices_to_try: list[tuple[str, str | None]] = []
        if _has_nvidia_gpu():
            indices_to_try.append(
                ("https://download.pytorch.org/whl/cu121", None)
            )
            indices_to_try.append(
                ("https://download.pytorch.org/whl/cu124", None)
            )
        # Always try CPU as last resort
        indices_to_try.append(
            ("https://download.pytorch.org/whl/cpu", None)
        )

        last_error: Exception | None = None
        for index_url, extra_index in indices_to_try:
            try:
                _pip_install([pkg], index_url=index_url, extra_index_url=extra_index)
                installed = True
                break
            except RuntimeError as e:
                last_error = e
                continue

        if not installed and last_error:
            raise last_error

    # Verify everything imported successfully after installation
    still_missing: list[str] = []
    for spec in missing_human:
        import_name = spec.split(":", 1)[0]
        if not _try_import(import_name):
            still_missing.append(spec)

    if still_missing:
        raise RuntimeError(
            f"Packages were installed but still cannot be imported: {still_missing}. "
            f"Try restarting the terminal / re-activating the virtual environment."
        )


def ensure_torch() -> None:
    """Shorthand: make sure torch is available (CPU or CUDA wheel)."""
    ensure("torch")


def ensure_faster_whisper() -> None:
    """Shorthand: make sure faster-whisper is available."""
    ensure_nvidia_cuda_runtime()
    ensure("faster_whisper:faster-whisper")


def ensure_yt_dlp() -> None:
    """Shorthand: make sure yt-dlp and EJS scripts are available."""
    from transcriptex.yt_dlp_utils import ensure_yt_dlp_ejs

    ensure_yt_dlp_ejs()


def ensure_pyaudiowpatch() -> None:
    """Shorthand: make sure pyaudiowpatch is available."""
    ensure("pyaudiowpatch")


def ensure_numpy() -> None:
    """Shorthand: make sure numpy is available."""
    ensure("numpy")
