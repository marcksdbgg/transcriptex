import os
import warnings
from typing import Any

# Suppress HuggingFace symlinks warning on Windows (non-admin users)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Suppress ctranslate2 / huggingface_hub noise
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Xet Storage is enabled", category=UserWarning)


class ASREngine:
    """
    Lazy-loading wrapper around faster-whisper WhisperModel.

    Accepts device="auto" and compute_type="auto":
      - auto device  → uses CUDA if available, otherwise CPU.
      - auto compute → float16 on GPU, int8 on CPU.
    """

    def __init__(self, model_size: str, device: str = "auto", compute_type: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def _resolve_device_and_compute(self) -> None:
        """Resolve 'auto' device/compute_type to concrete values, detecting GPU availability."""
        need_resolve = self.device == "auto" or self.compute_type == "auto"
        if not need_resolve:
            return

        gpu_available = False
        if self.device == "auto" or (self.compute_type == "auto" and self.device in ("auto", "cuda")):
            from transcriptex.dependency_manager import ensure_torch
            ensure_torch()
            import torch
            gpu_available = torch.cuda.is_available()

        if self.device == "auto":
            self.device = "cuda" if gpu_available else "cpu"
        if self.compute_type == "auto":
            self.compute_type = "float16" if self.device == "cuda" else "int8"

    def load(self) -> Any:
        if self.model is not None:
            return self.model

        from transcriptex.dependency_manager import ensure_faster_whisper
        ensure_faster_whisper()
        from faster_whisper import WhisperModel

        # Resolve auto values first (this will install torch if needed)
        self._resolve_device_and_compute()

        # If CUDA was explicitly requested but is not available, fall back gracefully
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("  [WARN] CUDA unavailable. Falling back to CPU (int8).")
                self.device = "cpu"
                self.compute_type = "int8"

        print(f"  [ASR] Loading faster-whisper ({self.model_size}) on {self.device} ({self.compute_type})...")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        return self.model

    def transcribe(self, audio, **kwargs) -> tuple[Any, Any]:
        """Proxy to faster-whisper's transcribe method, loading the model on-demand."""
        if self.model is None:
            self.load()
        return self.model.transcribe(audio, **kwargs)
