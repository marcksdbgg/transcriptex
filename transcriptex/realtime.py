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
    print(f"  [ASR] Initializing faster-whisper ({config.model})...")
    from faster_whisper import WhisperModel
    model = WhisperModel(
        config.model,
        device=config.device,
        compute_type=config.compute_type,
    )
    return model


def get_wasapi_loopback():
    import pyaudiowpatch as pa
    p = pa.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(pa.paWASAPI)
    except OSError:
        return None, None
        
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    
    if not default_speakers.get("isLoopbackDevice", False):
        found = False
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                default_speakers = loopback
                found = True
                break
        if not found:
            # fallback to first loopback
            for loopback in p.get_loopback_device_info_generator():
                default_speakers = loopback
                break
                
    if not default_speakers.get("isLoopbackDevice", False):
        return None, None
        
    return p, default_speakers


def resample_np(data: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if orig_sr == target_sr:
        return data
    n_samples = int(len(data) * target_sr / orig_sr)
    indices = np.linspace(0, len(data) - 1, n_samples)
    return np.interp(indices, np.arange(len(data)), data)


def run_realtime(config: RealtimeConfig):
    print("=================================================================")
    print("  Realtime Transcription (WASAPI Loopback)")
    print("=================================================================")
    print(f"  Model             : {config.model}")
    print(f"  Device            : {config.device} ({config.compute_type})")
    print("=================================================================\n")

    try:
        import pyaudiowpatch as pa
    except ImportError:
        print("  [ERR] pyaudiowpatch not installed. Run: pip install pyaudiowpatch numpy")
        return {"success": False, "error": "Missing dependencies"}

    p, loopback_device = get_wasapi_loopback()
    
    if loopback_device is None:
        print("  [ERR] No WASAPI Loopback device found. Are you on Windows?")
        print("    Please ensure your speakers are enabled.")
        return {"success": False, "error": "No loopback device"}

    print(f"  [OK] Using audio device: {loopback_device['name']}")

    model = initialize_whisper(config)
    print("  [OK] Whisper ready. Listening for system audio... (Press Ctrl+C to stop)")

    native_rate = int(loopback_device["defaultSampleRate"])
    channels = loopback_device["maxInputChannels"]
    
    audio_queue = queue.Queue()

    def callback(in_data, frame_count, time_info, status):
        if in_data:
            # in_data is bytes
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            # convert to mono
            if channels > 1:
                audio_data = audio_data.reshape((-1, channels)).mean(axis=1)
            audio_queue.put(audio_data.copy())
        return (in_data, pa.paContinue)

    buffer = np.array([], dtype=np.float32)
    last_text = ""
    
    WHISPER_SR = 16000

    try:
        stream = p.open(
            format=pa.paFloat32,
            channels=channels,
            rate=native_rate,
            frames_per_buffer=int(native_rate * 0.5), # 0.5s blocks
            input=True,
            input_device_index=loopback_device["index"],
            stream_callback=callback
        )
        
        stream.start_stream()
        
        while stream.is_active():
            try:
                chunk = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            buffer = np.concatenate((buffer, chunk))
            
            if len(buffer) >= native_rate * config.chunk_duration:
                # Resample buffer to 16000Hz for whisper
                resampled = resample_np(buffer, native_rate, WHISPER_SR)
                
                try:
                    segments, info = model.transcribe(
                        resampled,
                        beam_size=config.beam_size,
                        language=config.language if config.language else None,
                        condition_on_previous_text=False,
                        vad_filter=True, # Prevent hallucinating background noise
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    # Restrict languages to English or Spanish if not explicitly set
                    if not config.language and info.language not in ["en", "es"]:
                        text = ""
                    else:
                        text = " ".join([seg.text.strip() for seg in segments]).strip()
                    
                    if text:
                        # Clear line and print ongoing chunk
                        sys.stdout.write(f"\r\033[K{text}")
                        sys.stdout.flush()
                        last_text = text
                        
                    # Calculate simple RMS energy of the most recent ~1s chunk to detect silence
                    # This allows us to "lock in" the text and move to a new line when the user finishes speaking.
                    recent_audio = buffer[-native_rate:] if len(buffer) >= native_rate else buffer
                    rms = np.sqrt(np.mean(recent_audio**2))
                    is_silence = bool(rms < 0.005) # Empirically low threshold for silence detection
                    
                    # Lock text if they stopped talking, OR if buffer is getting huge (>25 seconds)
                    if (is_silence or len(buffer) > native_rate * 25) and last_text != "":
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        last_text = ""
                        # If silence, we can safely clear the buffer
                        if is_silence:
                            buffer = np.array([], dtype=np.float32)
                        else:
                            # Keep 1.5s overlap if we are just flushing due to max length
                            buffer = buffer[-int(native_rate * 1.5):]
                            
                except Exception as e:
                    print(f"\n[Error processing chunk: {e}]", end="")
                    # keep some overlap to avoid harsh cuts if transcription failed
                    buffer = buffer[-int(native_rate * 1.5):]
                
    except KeyboardInterrupt:
        print("\n\n  Stopped by user.")
    except Exception as e:
        print(f"\n  [ERR] Error streaming audio: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'stream' in locals() and stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

    return {"success": True}
