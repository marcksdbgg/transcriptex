import json
import os
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Suppress HuggingFace symlinks warning on Windows (non-admin users)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Suppress ctranslate2 pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
# Suppress hf_xet suggestion warning
warnings.filterwarnings("ignore", message="Xet Storage is enabled", category=UserWarning)

from transcriptex.asr_engine import ASREngine
from transcriptex.youtube_client import YouTubeClient


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


class SingleTranscriber:
    def __init__(self, config: TranscribeSingleConfig):
        self.config = config
        self.client = YouTubeClient(verbose=config.verbose)
        self.engine = ASREngine(
            model_size=config.model,
            device=config.device,
            compute_type=config.compute_type
        )

    def _format_timestamp(self, seconds: float, separator: str = ",") -> str:
        """Format seconds into HH:MM:SS,mmm or HH:MM:SS.mmm format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{milliseconds:03d}"

    def _write_srt(self, segments: list[Any], file_path: Path):
        with open(file_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                start = self._format_timestamp(segment.start, ",")
                end = self._format_timestamp(segment.end, ",")
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{segment.text.strip()}\n\n")

    def _write_vtt(self, segments: list[Any], file_path: Path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for segment in segments:
                start = self._format_timestamp(segment.start, ".")
                end = self._format_timestamp(segment.end, ".")
                f.write(f"{start} --> {end}\n")
                f.write(f"{segment.text.strip()}\n\n")

    def _write_json(self, segments: list[Any], file_path: Path):
        data = []
        for segment in segments:
            data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _write_txt(self, segments: list[Any], file_path: Path):
        """
        Write clean plain-text transcript.

        Segments are joined into flowing text. A blank line (paragraph break)
        is inserted whenever there is a silence gap longer than PARAGRAPH_GAP seconds,
        so natural pauses in speech become readable paragraph separations.
        """
        PARAGRAPH_GAP = 3.0  # seconds of silence → new paragraph

        paragraphs: list[list[str]] = [[]]
        prev_end: float | None = None

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            # Long pause → start a new paragraph
            if prev_end is not None and (segment.start - prev_end) >= PARAGRAPH_GAP:
                paragraphs.append([])
            paragraphs[-1].append(text)
            prev_end = segment.end

        # Join words within each paragraph with a space; join paragraphs with blank line
        output = "\n\n".join(" ".join(p) for p in paragraphs if p)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(output)
            f.write("\n")

    def run(self) -> dict[str, Any]:
        fmt = self.config.output_format.lower()
        fmt_labels = {
            "txt":  "TXT  — plain text (default)",
            "srt":  "SRT  — subtitles with timestamps",
            "vtt":  "VTT  — WebVTT subtitles",
            "json": "JSON — full segment data (start/end/text)",
        }
        print("=================================================================")
        print("  Single Transcription")
        print("=================================================================")
        print(f"  Input   : {self.config.input_path}")
        print(f"  Model   : {self.config.model}")
        print(f"  Device  : {self.config.device} ({self.config.compute_type})")
        print(f"  Format  : {fmt_labels.get(fmt, fmt.upper())}")
        print(f"  Output  : {self.config.output_dir}")
        print("="*65)
        print("  Formats: --format txt | srt | vtt | json")
        print("=================================================================")
        print()

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        is_url = self.config.input_path.startswith("http://") or self.config.input_path.startswith("https://")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = None
            
            try:
                if is_url:
                    print(f"  ↓ Downloading audio from URL...")
                    start_dw = time.time()
                    try:
                        audio_path = self.client.download_audio(
                            url=self.config.input_path,
                            tmp_dir=Path(temp_dir),
                        )
                        dw_time = time.time() - start_dw
                        print(f"    ✓ Downloaded in {dw_time:.1f}s")
                    except Exception as e:
                        print(f"    ✗ Download failed: {e}")
                        return {"success": False, "error": str(e)}
                else:
                    audio_path = Path(self.config.input_path)
                    if not audio_path.exists():
                        print(f"  ✗ Local file not found: {audio_path}")
                        return {"success": False, "error": "File not found"}

                print(f"  🎙 Transcribing...")
                start_tr = time.time()
                segments_gen, info = self.engine.transcribe(
                    str(audio_path),
                    beam_size=self.config.beam_size,
                    language=self.config.language if self.config.language else None,
                )
                
                segments = list(segments_gen)
                tr_time = time.time() - start_tr
                print(f"    ✓ Transcribed in {tr_time:.1f}s (Detected language: {info.language})")
                
            finally:
                pass # Temp file cleans up automatically

        # Save to file
        base_name = "transcript"
        if not is_url:
            base_name = Path(self.config.input_path).stem
        else:
            base_name = f"transcript_{int(time.time())}"
            
        out_file = self.config.output_dir / f"{base_name}.{self.config.output_format.lower()}"
        
        fmt = self.config.output_format.lower()
        if fmt == "srt":
            self._write_srt(segments, out_file)
        elif fmt == "vtt":
            self._write_vtt(segments, out_file)
        elif fmt == "json":
            self._write_json(segments, out_file)
        else:
            self._write_txt(segments, out_file)
            
        print(f"  ✓ Saved to: {out_file}")
        
        return {
            "success": True,
            "output_file": str(out_file),
            "language": info.language,
            "segments": len(segments),
        }

def run_transcribe_single(config: TranscribeSingleConfig) -> dict[str, Any]:
    transcriber = SingleTranscriber(config)
    return transcriber.run()
