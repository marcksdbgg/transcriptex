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
    process: str = ""
    list_devices: bool = False
    device_index: int = -1


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
        from transcriptex.dependency_manager import ensure_faster_whisper, ensure_torch
        ensure_faster_whisper()
        from faster_whisper import WhisperModel

        device = self.config.device
        compute_type = self.config.compute_type

        # Ensure torch is available when CUDA is requested so we can probe it
        if device == "cuda":
            ensure_torch()
            import torch
            if not torch.cuda.is_available():
                print("  [WARN] CUDA unavailable. Falling back to CPU (int8).")
                device = "cpu"
                compute_type = "int8"

        try:
            model = WhisperModel(
                self.config.model,
                device=device,
                compute_type=compute_type,
            )
            return model
        except Exception as e:
            if device == "cuda" and ("cublas" in str(e).lower() or "cuda" in str(e).lower() or "gpu" in str(e).lower()):
                print(f"  [WARN] CUDA libraries not available ({e})")
                print("  [WARN] Falling back to CPU (int8). Install CUDA toolkit for GPU support.")
                print("         Download from: https://developer.nvidia.com/cuda-downloads")
                model = WhisperModel(
                    self.config.model,
                    device="cpu",
                    compute_type="int8",
                )
                return model
            raise

    def _list_devices(self):
        from transcriptex.dependency_manager import ensure_pyaudiowpatch, ensure_numpy
        ensure_pyaudiowpatch()
        ensure_numpy()
        import pyaudiowpatch as pa
        p = pa.PyAudio()
        print("=== WASAPI Loopback Devices ===\n")
        for i, d in enumerate(p.get_loopback_device_info_generator()):
            print(f"  {i}: {d['name']}")
            print(f"     Channels: {d['maxInputChannels']}, Rate: {int(d['defaultSampleRate'])} Hz\n")
        p.terminate()

    def _get_wasapi_loopback(self, device_index: int = -1):
        from transcriptex.dependency_manager import ensure_pyaudiowpatch, ensure_numpy
        ensure_pyaudiowpatch()
        ensure_numpy()
        import pyaudiowpatch as pa
        p = pa.PyAudio()
        try:
            wasapi_info = p.get_host_api_info_by_type(pa.paWASAPI)
        except OSError:
            return None, None

        if device_index >= 0:
            loopback_devices = list(p.get_loopback_device_info_generator())
            if device_index < len(loopback_devices):
                return p, loopback_devices[device_index]
            else:
                print(f"  [ERR] Device index {device_index} out of range (0-{len(loopback_devices)-1})")
                return None, None

        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        if not default_speakers.get("isLoopbackDevice", False):
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            if not default_speakers.get("isLoopbackDevice", False):
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

    def _update_terminal_ui(self, text: str, metrics: str = ""):
        if not text:
            return
            
        if text == self.last_text and metrics == self.last_metrics:
            return
            
        display_str = f"{text} \033[90m{metrics}\033[0m" if metrics else text
        term_width = shutil.get_terminal_size().columns or 80
        wrapped_lines = textwrap.wrap(display_str, width=max(10, term_width - 4))
        
        if self.lines_to_clear > 0:
            for _ in range(self.lines_to_clear):
                sys.stdout.write("\033[F\033[K")
                
        sys.stdout.write("\n".join(wrapped_lines) + "\n")
        sys.stdout.flush()
        
        self.lines_to_clear = len(wrapped_lines)
        self.last_text = text
        self.last_metrics = metrics

    def run(self) -> dict:
        if self.config.list_devices:
            self._list_devices()
            return {"success": True}

        print("=================================================================")
        print("  Realtime Transcription (WASAPI Loopback)")
        print("=================================================================")
        print(f"  Model             : {self.config.model}")
        print(f"  Device            : {self.config.device} ({self.config.compute_type})")
        if self.config.process:
            print(f"  Target Process    : {self.config.process}")
        print("=================================================================\n")

        if self.config.process:
            return self._run_pac_loop()
        else:
            return self._run_wasapi_loop()

    def _run_pac_loop(self) -> dict:
        try:
            import process_audio_capture as pac
        except ImportError:
            print("  [ERR] process_audio_capture not installed. Run: pip install process-audio-capture pywin32")
            return {"success": False, "error": "Missing dependencies"}

        try:
            import win32pipe, win32file, pywintypes
        except ImportError:
            print("  [ERR] pywin32 not installed. Run: pip install pywin32")
            return {"success": False, "error": "Missing pywin32"}

        procs = pac.ProcessAudioCapture.enumerate_audio_processes()
        target_pid = None
        target_name = None
        for p in procs:
            if self.config.process.lower() in p.name.lower() or self.config.process.lower() in p.window_title.lower():
                target_pid = p.pid
                target_name = p.name
                break

        if not target_pid:
            print(f"  [ERR] Process '{self.config.process}' not found among processes outputting audio.")
            print("        Currently active processes:")
            for p in procs:
                print(f"        - {p.name} (PID: {p.pid})")
            return {"success": False, "error": "Process not found"}

        print(f"  [OK] Found Target Process: {target_name} (PID: {target_pid})")

        import struct
        pipe_name = r'\\.\pipe\transcriptex_app_loopback'
        read_thread_err = None

        def pipe_reader():
            nonlocal read_thread_err
            # Create Named Pipe
            handle = win32pipe.CreateNamedPipe(
                pipe_name,
                win32pipe.PIPE_ACCESS_INBOUND,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                1, 65536 * 4, 65536 * 4, 0, None)
            
            try:
                win32pipe.ConnectNamedPipe(handle, None)
                
                # Parse WAV header
                # RIFF header
                _, buffer = win32file.ReadFile(handle, 12)
                
                channels = 2
                native_rate = 48000
                bits = 32
                
                while True:
                    _, chunk_header = win32file.ReadFile(handle, 8)
                    if len(chunk_header) < 8:
                        break
                    chunk_id, chunk_size = struct.unpack('<4sI', chunk_header)
                    
                    if chunk_id == b'fmt ':
                        _, fmt_data = win32file.ReadFile(handle, chunk_size)
                        _, channels, native_rate, _, _, bits = struct.unpack('<HHIIHH', fmt_data[:16])
                    elif chunk_id == b'data':
                        break
                    else:
                        _, _ = win32file.ReadFile(handle, chunk_size)
                
                bytes_per_sample = bits // 8
                # Read chunks of ~0.5s
                chunk_bytes = int(native_rate * 0.5) * channels * bytes_per_sample
                
                while self.is_running:
                    err, raw_data = win32file.ReadFile(handle, chunk_bytes)
                    if not raw_data:
                        break
                    
                    if bits == 32:
                        audio_data = np.frombuffer(raw_data, dtype=np.float32)
                    elif bits == 16:
                        audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        continue # Unsupported depth
                        
                    if channels > 1:
                        audio_data = audio_data.reshape((-1, channels)).mean(axis=1)
                        
                    self.audio_queue.put(audio_data.copy(), timeout=2.0)
                    
            except Exception as e:
                if self.is_running:
                    read_thread_err = e
            finally:
                win32file.CloseHandle(handle)

        import threading
        t = threading.Thread(target=pipe_reader, daemon=True)
        t.start()
        
        # Give pipe_reader a moment to setup server
        import time
        time.sleep(0.5)

        print("  [OK] Whisper ready. Listening for target application audio... (Press Ctrl+C to stop)")
        
        self.is_running = True
        
        try:
            with pac.ProcessAudioCapture(pid=target_pid, output_path=pipe_name) as capture:
                capture.start()
                
                # Assume default rate since we process dynamic chunk sizing anyway 
                native_rate = 48000
                
                max_buffer_frames = int(native_rate * self.config.chunk_duration * 1.5)
                processing_interval = int(native_rate * 0.75)
                unprocessed_frames = 0
                
                while self.is_running and t.is_alive():
                    if read_thread_err:
                        raise read_thread_err
                        
                    try:
                        chunk = self.audio_queue.get(timeout=1.0)
                        self.buffer = np.concatenate((self.buffer, chunk))
                        unprocessed_frames += len(chunk)
                        
                        while not self.audio_queue.empty():
                            ch = self.audio_queue.get_nowait()
                            self.buffer = np.concatenate((self.buffer, ch))
                            unprocessed_frames += len(ch)
                            
                    except queue.Empty:
                        if len(self.buffer) < native_rate * self.config.chunk_duration:
                            continue
                    
                    if len(self.buffer) < native_rate * self.config.chunk_duration:
                        continue
                        
                    if unprocessed_frames < processing_interval:
                        continue
                        
                    unprocessed_frames = 0
                    resampled = self._resample(self.buffer, native_rate)
                    
                    self._process_whisper_chunk(resampled, audio_dur=len(self.buffer)/native_rate, 
                                                native_rate=native_rate, max_buffer_frames=max_buffer_frames)
                    
        except KeyboardInterrupt:
            print("\n\n  Stopped by user.")
        except Exception as e:
            print(f"\n  [ERR] {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.is_running = False
            t.join(timeout=1.0)

        return {"success": True}

    def _process_whisper_chunk(self, resampled: np.ndarray, audio_dur: float, native_rate: int, max_buffer_frames: int):
        import time
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
            proc_time = t1 - t0
            rtf = proc_time / audio_dur if audio_dur > 0 else 0.0
            
            if not self.config.language and info.language not in ["en", "es"]:
                text = ""
            else:
                text = " ".join([seg.text.strip() for seg in segments]).strip()
            
            metrics_str = f"[{audio_dur:.1f}s | Q:{self.audio_queue.qsize()} | RT:{rtf:.2f}x]"
            
            self._update_terminal_ui(text, metrics_str)
                
            recent_audio = self.buffer[-native_rate:] if len(self.buffer) >= native_rate else self.buffer
            rms = np.sqrt(np.mean(recent_audio**2))
            is_silence = bool(rms < self.SILENCE_THRESHOLD_RMS)
            
            if (is_silence or len(self.buffer) >= max_buffer_frames) and self.last_text != "":
                self._update_terminal_ui(self.last_text, metrics="")
                
                self.committed_history += (" " + self.last_text)
                self.lines_to_clean = 0
                self.last_text = ""
                self.last_metrics = ""
                
                if is_silence:
                    self.buffer = np.array([], dtype=np.float32)
                else:
                    self.buffer = self.buffer[-int(native_rate * 0.5):]
                    
        except Exception as e:
            self.buffer = self.buffer[-int(native_rate * 0.5):]


    def _run_wasapi_loop(self) -> dict:
        from transcriptex.dependency_manager import ensure_pyaudiowpatch, ensure_numpy
        ensure_pyaudiowpatch()
        ensure_numpy()
        import pyaudiowpatch as pa

        p, loopback_device = self._get_wasapi_loopback(self.config.device_index)
        
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
        
        # Set processing interval natively to 0.75s (accumulate new audio before transcribing)
        processing_interval = int(native_rate * 0.75)
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
                
                self._process_whisper_chunk(resampled, audio_dur=len(self.buffer)/native_rate, 
                                            native_rate=native_rate, max_buffer_frames=max_buffer_frames)
                                                
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
