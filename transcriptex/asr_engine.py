import os
import sys
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
        self.batched = None

    def _resolve_device_and_compute(self) -> None:
        """Resolve 'auto' device/compute_type to concrete values, detecting GPU availability."""
        need_resolve = self.device == "auto" or self.compute_type == "auto"
        if not need_resolve:
            return

        gpu_available = False
        if self.device in ("auto", "cuda"):
            from transcriptex.dependency_manager import cuda_available

            gpu_available = cuda_available()

        if self.device == "auto":
            self.device = "cuda" if gpu_available else "cpu"
        elif self.device == "cuda" and not gpu_available:
            print("  [WARN] CUDA unavailable. Falling back to CPU (int8).", file=sys.stderr)
            self.device = "cpu"
            self.compute_type = "int8"
        if self.compute_type == "auto":
            self.compute_type = "float16" if self.device == "cuda" else "int8"

    def load(self) -> Any:
        if self.model is not None:
            return self.model

        from transcriptex.dependency_manager import ensure_faster_whisper
        ensure_faster_whisper()
        from faster_whisper import WhisperModel

        # Resolve auto values using CTranslate2, the inference backend
        self._resolve_device_and_compute()

        print(f"  [ASR] Loading faster-whisper ({self.model_size}) on {self.device} ({self.compute_type})...", file=sys.stderr)
        try:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        except RuntimeError as exc:
            # A visible GPU does not imply usable cuBLAS/cuDNN libraries.
            if self.device != "cuda" or not any(x in str(exc).lower() for x in ("cuda", "cublas", "cudnn")):
                raise
            print(f"  [WARN] CUDA runtime failed: {exc}. Falling back to CPU (int8).", file=sys.stderr)
            self.device, self.compute_type = "cpu", "int8"
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self.model

    def transcribe(self, audio, **kwargs) -> tuple[Any, Any]:
        """Proxy to faster-whisper's transcribe method, loading the model on-demand."""
        if self.model is None:
            self.load()
        return self.model.transcribe(audio, **kwargs)

    def transcribe_file(self, audio, *, batch_size: int = 0, vad_filter: bool = True, **kwargs):
        """Materialize one file; reduce batch size on GPU OOM before writing output.

        A zero batch size selects 8 on CUDA and sequential inference on CPU.
        The same engine can process several inputs without loading weights again.
        """
        self.load()
        size = batch_size or (8 if self.device == "cuda" else 1)
        if not vad_filter:
            size = 1
        if size > 1 and self.batched is None:
            from faster_whisper import BatchedInferencePipeline
            self.batched = BatchedInferencePipeline(model=self.model)
        while True:
            try:
                if size > 1:
                    # The batched decoder stores alignment state between calls.
                    self.batched.last_speech_timestamp = 0.0
                    segments, info = self.batched.transcribe(audio, batch_size=size, vad_filter=True,
                        vad_parameters={"min_silence_duration_ms": 500}, **kwargs)
                else:
                    segments, info = self.model.transcribe(audio, vad_filter=vad_filter,
                        vad_parameters={"min_silence_duration_ms": 500}, **kwargs)
                return list(segments), info, size
            except RuntimeError as exc:
                if self.device != "cuda" or size <= 1 or "out of memory" not in str(exc).lower():
                    raise
                size = max(1, size // 2)
                print(f"  [WARN] GPU memory exhausted; retrying with batch size {size}.", file=sys.stderr)
