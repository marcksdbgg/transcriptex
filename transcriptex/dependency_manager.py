"""
dependency_manager.py — Auto-install missing Python packages into the project's venv.

This module is imported by commands that need heavy optional dependencies
(faster-whisper, torch, pyaudiowpatch, etc.).  It never mutates the *system*
Python; it always targets the interpreter that is currently running
transcriptex (usually ``.venv\\Scripts\\python.exe`` on Windows).
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import ctypes
from typing import Iterable


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

    print(f"[dep-mgr] Installing missing packages: {', '.join(packages)} ...")
    result = subprocess.run(cmd, check=False)
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
