import queue
import sys
import shutil
import textwrap
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RealtimeConfig:
    model: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    chunk_duration: float = 2.0


class RealtimeTranscriber:
    """
    Object-Oriented Transcriber designed for blazing fast, low-latency WASAPI loopback capture.
    Maintains a heavily capped buffer coupled with rolling initial_prompt text history to 
    maintain semantic structure across chunks without suffering from infinite CPU lag.
    """
    
    WHISPER_SR = 16000
    SILENCE_THRESHOLD_RMS = 0.005
    
    def __init__(self, config: RealtimeConfig):
        self.config = config
        self.model = self._init_whisper()
        self.audio_queue = queue.Queue()
        self.buffer = np.array([], dtype=np.float32)
        
        # UI & State tracking
        self.is_running = False
        self.lines_to_clear = 0
        self.last_text = ""
        self.committed_history = ""
        
    def _init_whisper(self):
        print(f"  [ASR] Initializing faster-whisper ({self.config.model})...")
        from faster_whisper import WhisperModel
        return WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def _get_wasapi_loopback(self):
        import pyaudiowpatch as pa
        p = pa.PyAudio()
        try:
            wasapi_info = p.get_host_api_info_by_type(pa.paWASAPI)
        except OSError:
            return None, None
            
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        if not default_speakers.get("isLoopbackDevice", False):
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            if not default_speakers.get("isLoopbackDevice", False):
                # Fallback
                for loopback in p.get_loopback_device_info_generator():
                    default_speakers = loopback
                    break
                    
        if not default_speakers.get("isLoopbackDevice", False):
            return None, None
            
        return p, default_speakers

    def _resample(self, data: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.WHISPER_SR:
            return data.astype(np.float32)
        n_samples = int(len(data) * self.WHISPER_SR / orig_sr)
        indices = np.linspace(0, len(data) - 1, n_samples)
        return np.interp(indices, np.arange(len(data)), data).astype(np.float32)

    def _audio_callback(self, in_data, frame_count, time_info, status, pa, channels):
        if in_data:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            if channels > 1:
                audio_data = audio_data.reshape((-1, channels)).mean(axis=1)
            self.audio_queue.put(audio_data.copy())
        return (in_data, pa.paContinue)

    def _update_terminal_ui(self, text: str):
        if not text or text == self.last_text:
            return
            
        term_width = shutil.get_terminal_size().columns or 80
        wrapped_lines = textwrap.wrap(text, width=max(10, term_width - 4))
        
        if self.lines_to_clear > 0:
            for _ in range(self.lines_to_clear):
                sys.stdout.write("\033[F\033[K")
                
        sys.stdout.write("\n".join(wrapped_lines) + "\n")
        sys.stdout.flush()
        
        self.lines_to_clear = len(wrapped_lines)
        self.last_text = text

    def run(self) -> dict:
        print("=================================================================")
        print("  Realtime Transcription (WASAPI Loopback)")
        print("=================================================================")
        print(f"  Model             : {self.config.model}")
        print(f"  Device            : {self.config.device} ({self.config.compute_type})")
        print("=================================================================\n")

        try:
            import pyaudiowpatch as pa
        except ImportError:
            print("  [ERR] pyaudiowpatch not installed. Run: pip install pyaudiowpatch numpy")
            return {"success": False, "error": "Missing dependencies"}

        p, loopback_device = self._get_wasapi_loopback()
        
        if not loopback_device:
            print("  [ERR] No WASAPI Loopback device found. Are you on Windows?")
            return {"success": False, "error": "No loopback device"}

        print(f"  [OK] Using audio device: {loopback_device['name']}")
        print("  [OK] Whisper ready. Listening for system audio... (Press Ctrl+C to stop)")

        native_rate = int(loopback_device["defaultSampleRate"])
        channels = loopback_device["maxInputChannels"]
        
        # Max buffer frames set VERY LOW (1.5x of chunk_duration) to enforce low-latency real-time
        # E.g. 2.0s duration -> 3.0s max context. Prevents quadratic processing slowdowns!
        max_buffer_frames = int(native_rate * self.config.chunk_duration * 1.5)
        
        # Set processing interval natively to 1.0s (accumulate new audio before transcribing)
        processing_interval = int(native_rate * 1.0)
        unprocessed_frames = 0
        
        try:
            stream = p.open(
                format=pa.paFloat32,
                channels=channels,
                rate=native_rate,
                frames_per_buffer=int(native_rate * 0.5), # 0.5s chunks
                input=True,
                input_device_index=loopback_device["index"],
                stream_callback=lambda *args: self._audio_callback(*args, pa=pa, channels=channels)
            )
            
            stream.start_stream()
            self.is_running = True
            
            import time
            while self.is_running and stream.is_active():
                try:
                    chunk = self.audio_queue.get(timeout=1.0)
                    self.buffer = np.concatenate((self.buffer, chunk))
                    unprocessed_frames += len(chunk)
                    
                    # Drain queue to catch up perfectly if the GPU took slightly longer
                    while not self.audio_queue.empty():
                        ch = self.audio_queue.get_nowait()
                        self.buffer = np.concatenate((self.buffer, ch))
                        unprocessed_frames += len(ch)
                        
                except queue.Empty:
                    # If empty, but buffer hasn't reached even processing interval, wait more
                    if len(self.buffer) < native_rate * self.config.chunk_duration:
                        continue
                
                # Check if we have gathered enough context (initial startup delay)
                if len(self.buffer) < native_rate * self.config.chunk_duration:
                    continue
                    
                # Ensure we only compute if new audio arrived (prevents repetitive loop lag)
                if unprocessed_frames < processing_interval:
                    continue
                    
                unprocessed_frames = 0
                resampled = self._resample(self.buffer, native_rate)
                
                try:
                    t0 = time.time()
                    prompt = self.committed_history[-150:] if self.committed_history else None
                    
                    segments, info = self.model.transcribe(
                        resampled,
                        beam_size=self.config.beam_size,
                        initial_prompt=prompt,
                        language=self.config.language if self.config.language else None,
                        condition_on_previous_text=False,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=400)
                    )
                    
                    t1 = time.time()
                    audio_dur = len(self.buffer) / native_rate
                    proc_time = t1 - t0
                    rtf = proc_time / audio_dur if audio_dur > 0 else 0.0
                    
                    if not self.config.language and info.language not in ["en", "es"]:
                        text = ""
                    else:
                        text = " ".join([seg.text.strip() for seg in segments]).strip()
                    
                    # Add metrics string behind the scenes to UI
                    metrics_str = f" [{(len(self.buffer)/native_rate):.1f}s | Q:{self.audio_queue.qsize()} | RT:{rtf:.2f}x]"
                    
                    if text and text != self.last_text:
                        # Incorporates text + metrics temporarily
                        self._update_terminal_ui(f"{text} \033[90m{metrics_str}\033[0m")
                        self.last_text = text
                        
                    recent_audio = self.buffer[-native_rate:] if len(self.buffer) >= native_rate else self.buffer
                    rms = np.sqrt(np.mean(recent_audio**2))
                    is_silence = bool(rms < self.SILENCE_THRESHOLD_RMS)
                    
                    if (is_silence or len(self.buffer) >= max_buffer_frames) and self.last_text != "":
                        # Overwrite metric line without metrics to make it permanent
                        self._update_terminal_ui(self.last_text)
                        
                        self.committed_history += (" " + self.last_text)
                        self.lines_to_clear = 0
                        self.last_text = ""
                        
                        if is_silence:
                            self.buffer = np.array([], dtype=np.float32)
                        else:
                            self.buffer = self.buffer[-int(native_rate * 0.5):]
                            
                except Exception as e:
                    self.buffer = self.buffer[-int(native_rate * 0.5):]
                
        except KeyboardInterrupt:
            print("\n\n  Stopped by user.")
        except Exception as e:
            print(f"\n  [ERR] {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.is_running = False
            if 'stream' in locals() and stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

        return {"success": True}


def run_realtime(config: RealtimeConfig) -> dict:
    transcriber = RealtimeTranscriber(config)
    return transcriber.run()
