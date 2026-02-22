import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RealtimeConfig:
    model: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    chunk_duration: float = 3.0


def initialize_whisper(config: RealtimeConfig):
    print(f"  🎙 Initializing faster-whisper ({config.model})...")
    from faster_whisper import WhisperModel
    model = WhisperModel(
        config.model,
        device=config.device,
        compute_type=config.compute_type,
    )
    return model


def get_loopback_device() -> Optional[int]:
    try:
        import pyaudiowpatch  # Ensure WASAPI is available
        import sounddevice as sd
    except ImportError:
        print("    ✗ sounddevice/pyaudiowpatch not installed. Run: pip install pyaudiowpatch sounddevice numpy")
        sys.exit(1)

    # find default loopback
    devices = sd.query_devices()
    try:
        default_output = sd.default.device[1]  # output device index
        # We need the loopback version of the default output
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels") > 0 and dev.get("is_loopback"):
                # Simplification: return the first WASAPI loopback device
                # A more robust approach checks if it matches default_output name
                return i
    except Exception:
        pass
        
    for i, dev in enumerate(devices):
        if dev.get("max_input_channels") > 0 and dev.get("is_loopback"):
            return i
            
    return None


def run_realtime(config: RealtimeConfig):
    print("=================================================================")
    print("  Realtime Transcription (WASAPI Loopback)")
    print("=================================================================")
    print(f"  Model             : {config.model}")
    print(f"  Device            : {config.device} ({config.compute_type})")
    print("=================================================================\n")

    import sounddevice as sd

    loopback_idx = get_loopback_device()
    if loopback_idx is None:
        print("  ✗ No WASAPI Loopback device found. Are you on Windows?")
        print("    Please ensure your speakers are enabled and PyAudioWPatch is installed.")
        return {"success": False, "error": "No loopback device"}

    device_info = sd.query_devices(loopback_idx)
    print(f"  ✓ Using audio device: {device_info['name']}")

    model = initialize_whisper(config)
    print("  ✓ Whisper ready. Listening for system audio... (Press Ctrl+C to stop)")

    # Faster Whisper requires 16000Hz mono numpy arrays (float32)
    sample_rate = 16000
    chunk_samples = int(sample_rate * config.chunk_duration)
    
    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Stream warning: {status}")
        # convert to mono by averaging channels
        mono = np.mean(indata, axis=1)
        audio_queue.put(mono.copy())

    # We use a running buffer to avoid cutting words
    buffer = np.array([], dtype=np.float32)
    
    # We maintain previous text to prevent overlapping outputs
    last_text = ""

    try:
        with sd.InputStream(
            device=loopback_idx,
            channels=device_info["max_input_channels"],
            samplerate=sample_rate,
            dtype='float32',
            callback=callback,
            blocksize=chunk_samples  # Send blocks at chunk_duration
        ):
            while True:
                # Wait for next audio chunk
                chunk = audio_queue.get()
                buffer = np.concatenate((buffer, chunk))
                
                # Only process if we have at least chunk_duration audio
                if len(buffer) >= sample_rate * config.chunk_duration:
                    try:
                        # Process current buffer
                        segments, info = model.transcribe(
                            buffer,
                            beam_size=config.beam_size,
                            language=config.language if config.language else None,
                            condition_on_previous_text=False,
                        )
                        
                        text = " ".join([seg.text.strip() for seg in segments]).strip()
                        
                        if text and text != last_text:
                            # Clear line and print new text
                            sys.stdout.write(f"\r\033[K{text}")
                            sys.stdout.flush()
                            last_text = text
                            
                    except Exception as e:
                        print(f"\n[Error processing chunk: {e}]", end="")
                    
                    # keep last second to avoid hard cuts
                    buffer = buffer[-sample_rate:] 
                        
    except KeyboardInterrupt:
        print("\n\n  Stopped by user.")
        return {"success": True}
    except Exception as e:
        print(f"\n  ✗ Error streaming audio: {e}")
        return {"success": False, "error": str(e)}
