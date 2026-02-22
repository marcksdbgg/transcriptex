import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transcriptex.youtube_client import download_single_audio


@dataclass
class TranscribeSingleConfig:
    input_path: str
    output_dir: Path
    output_format: str
    model: str
    device: str
    compute_type: str
    language: str
    beam_size: int
    verbose: bool


def format_timestamp(seconds: float, separator: str = ",") -> str:
    """Format seconds into HH:MM:SS,mmm or HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"


def write_srt(segments: list[Any], file_path: Path):
    with open(file_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start, ",")
            end = format_timestamp(segment.end, ",")
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment.text.strip()}\n\n")


def write_vtt(segments: list[Any], file_path: Path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in segments:
            start = format_timestamp(segment.start, ".")
            end = format_timestamp(segment.end, ".")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment.text.strip()}\n\n")


def write_json(segments: list[Any], file_path: Path):
    data = []
    for segment in segments:
        data.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_txt(segments: list[Any], file_path: Path):
    with open(file_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(f"{segment.text.strip()}\n")


def run_transcribe_single(config: TranscribeSingleConfig) -> dict[str, Any]:
    print("=================================================================")
    print("  Single Transcription")
    print("=================================================================")
    print(f"  Input             : {config.input_path}")
    print(f"  Model             : {config.model}")
    print(f"  Device            : {config.device} ({config.compute_type})")
    print(f"  Format            : {config.output_format.upper()}")
    print("=================================================================\n")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    is_url = config.input_path.startswith("http://") or config.input_path.startswith("https://")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = None
        temp_audio = None
        
        try:
            if is_url:
                print(f"  ↓ Downloading audio from URL...")
                start_dw = time.time()
                try:
                    temp_audio = download_single_audio(
                        video_url=config.input_path,
                        output_dir=Path(temp_dir),
                        verbose=config.verbose,
                    )
                    audio_path = temp_audio
                    dw_time = time.time() - start_dw
                    print(f"    ✓ Downloaded in {dw_time:.1f}s")
                except Exception as e:
                    print(f"    ✗ Download failed: {e}")
                    return {"success": False, "error": str(e)}
            else:
                audio_path = Path(config.input_path)
                if not audio_path.exists():
                    print(f"  ✗ Local file not found: {audio_path}")
                    return {"success": False, "error": "File not found"}

            print(f"  🎙 Initializing faster-whisper...")
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                print("    ✗ faster-whisper is not installed. Run: pip install faster-whisper")
                return {"success": False, "error": "faster-whisper not installed"}
            
            model = WhisperModel(
                config.model,
                device=config.device,
                compute_type=config.compute_type,
            )

            print(f"  🎙 Transcribing...")
            start_tr = time.time()
            segments_gen, info = model.transcribe(
                str(audio_path),
                beam_size=config.beam_size,
                language=config.language if config.language else None,
            )
            
            segments = list(segments_gen)
            tr_time = time.time() - start_tr
            print(f"    ✓ Transcribed in {tr_time:.1f}s (Detected language: {info.language})")
            
        finally:
            if temp_audio and Path(temp_audio).exists():
                try:
                    os.remove(temp_audio)
                except Exception:
                    pass

    # Save to file
    base_name = "transcript"
    if not is_url:
        base_name = Path(config.input_path).stem
    else:
        # Try to sanitize URL or use timestamp
        base_name = f"transcript_{int(time.time())}"
        
    out_file = config.output_dir / f"{base_name}.{config.output_format.lower()}"
    
    fmt = config.output_format.lower()
    if fmt == "srt":
        write_srt(segments, out_file)
    elif fmt == "vtt":
        write_vtt(segments, out_file)
    elif fmt == "json":
        write_json(segments, out_file)
    else:
        write_txt(segments, out_file)
        
    print(f"  ✓ Saved to: {out_file}")
    
    return {
        "success": True,
        "output_file": str(out_file),
        "language": info.language,
        "segments": len(segments),
    }
