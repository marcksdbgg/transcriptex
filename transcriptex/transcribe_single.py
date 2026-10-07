import json
import os
import tempfile
import time
import warnings
import hashlib
import sys
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
    cookies: str = ""
    quiet: bool = False
    print_summary: bool = True
    source: str = "auto"
    lang_priority: str = ""
    batch_size: int = 0
    vad_filter: bool = True
    output_stem: str = ""


class SingleTranscriber:
    def __init__(self, config: TranscribeSingleConfig, engine: ASREngine | None = None):
        self.config = config
        self.client = YouTubeClient(verbose=config.verbose, cookies=config.cookies)
        self.engine = engine or ASREngine(
            model_size=config.model,
            device=config.device,
            compute_type=config.compute_type
        )

    def _format_timestamp(self, seconds: float, separator: str = ",") -> str:
        """Format seconds into HH:MM:SS,mmm or HH:MM:SS.mmm format."""
        total_ms = max(0, round(seconds * 1000))
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, milliseconds = divmod(remainder, 1000)
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

    def _try_subtitles(self):
        from transcriptex.transcriber import select_subtitle_track
        from transcriptex.subtitle_segments import parse_subtitle_segments
        info = self.client.extract_video_info(self.config.input_path)
        priorities = [p.strip() for p in self.config.lang_priority.split(',') if p.strip()]
        priorities = priorities or ([self.config.language] if self.config.language else ['es', 'en'])
        track = select_subtitle_track(info, priorities)
        if track is None:
            raise ValueError('No subtitle tracks available')
        segments = parse_subtitle_segments(self.client.fetch_binary(track.url), track.ext)
        return segments, track.language, track.source, info.get('duration')

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        config = self.config
        is_url = config.input_path.startswith(('http://', 'https://'))
        if config.source == 'subtitles' and not is_url:
            raise ValueError('--source subtitles requires a video URL')
        if not is_url and not Path(config.input_path).is_file():
            return {'success': False, 'error': 'File not found', 'input': config.input_path}
        config.output_dir.mkdir(parents=True, exist_ok=True)
        segments = None
        subtitle_error = ''
        download_s = 0.0
        recognition_s = 0.0
        actual_batch = 0
        if is_url and config.source != 'asr':
            try:
                segments, language, source, duration = self._try_subtitles()
            except Exception as exc:
                if config.source == 'subtitles':
                    raise
                subtitle_error = str(exc)
                if not config.quiet:
                    print(f'  [INFO] Subtitles unavailable ({exc}); using local ASR.', file=sys.stderr)
        if segments is None:
            with tempfile.TemporaryDirectory(prefix='transcriptex-') as temporary:
                if is_url:
                    download_started = time.perf_counter()
                    audio = self.client.download_audio(config.input_path, Path(temporary))
                    download_s = time.perf_counter() - download_started
                else:
                    audio = Path(config.input_path)
                asr_started = time.perf_counter()
                segments, info, actual_batch = self.engine.transcribe_file(str(audio),
                    batch_size=config.batch_size, vad_filter=config.vad_filter,
                    beam_size=config.beam_size, language=config.language or None)
                recognition_s = time.perf_counter() - asr_started
                language = info.language
                duration = getattr(info, 'duration', None)
                source = 'asr'
        if config.output_stem:
            base_name = config.output_stem
        elif not is_url:
            base_name = Path(config.input_path).stem
        else:
            from transcriptex.url_utils import extract_video_id
            video_id = extract_video_id(config.input_path)
            base_name = 'transcript_' + (video_id or hashlib.sha256(config.input_path.encode()).hexdigest()[:12])
        out_file = config.output_dir / f'{base_name}.{config.output_format.lower()}'
        # Write in the destination filesystem and replace only after a complete export.
        fd, temporary_name = tempfile.mkstemp(prefix='.' + base_name + '-', suffix='.tmp', dir=config.output_dir)
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            writer = getattr(self, '_write_' + config.output_format.lower())
            writer(segments, temporary_path)
            os.replace(temporary_path, out_file)
        finally:
            temporary_path.unlink(missing_ok=True)
        if config.print_summary and not config.quiet:
            print(f'  Saved: {out_file} ({source}, {language}, {len(segments)} segments)')
        return {
            'success': True, 'input': config.input_path, 'output_file': str(out_file),
            'language': language, 'source': source, 'segments': len(segments),
            'duration_seconds': duration, 'batch_size': actual_batch,
            'device': self.engine.device if source == 'asr' else None,
            'timing': {'download_s': download_s, 'recognition_with_load_s': recognition_s,
                       'total_s': time.perf_counter() - started},
            'subtitle_fallback_reason': subtitle_error,
        }


def run_transcribe_single(config: TranscribeSingleConfig, engine: ASREngine | None = None) -> dict[str, Any]:
    return SingleTranscriber(config, engine=engine).run()
