"""
asr.py — ASR fallback transcription using faster-whisper (GPU/CPU).

Reads the channel's video_index.csv, downloads audio for videos with
status='no_subtitles' (or 'failed' if requested), transcribes them locally
with faster-whisper, and writes .txt files in the same format as the main
transcriber. Updates video_index.csv with the new status.
"""
from __future__ import annotations

import csv
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# FIX: Disable symlinks on Windows to avoid [WinError 1314] if not Admin/DevMode
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from yt_channel_transcriber.fs_utils import (
    build_transcript_filename,
    ensure_dir,
    extract_video_id_from_stem,
    now_utc_iso,
    sanitize_name,
)

META_FOLDER = "_meta"
STATUS_OK = "ok"
STATUS_NO_SUBTITLES = "no_subtitles"
STATUS_FAILED = "failed"
STATUS_EXISTING = "existing"
SOURCE_ASR = "asr"

# Video index CSV column order (must match transcriber.py)
VIDEO_INDEX_HEADERS = [
    "video_id",
    "title",
    "url",
    "assigned_playlist_id",
    "assigned_playlist_title",
    "all_playlist_ids",
    "all_playlist_titles",
    "folder",
    "transcript_path",
    "status",
    "language",
    "source",
    "error",
]


@dataclass
class AsrConfig:
    """Configuration for the ASR transcription command."""

    channel_dir: Path
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "es"
    beam_size: int = 5
    batch_size: int = 8
    force: bool = False
    limit: int = 0  # 0 = no limit
    sleep_seconds: float = 0.0
    include_failed: bool = False
    verbose: bool = False


@dataclass
class AsrResult:
    """Result of a single video ASR transcription attempt."""

    video_id: str
    status: str  # "ok", "skipped", "error", "unavailable"
    transcript_path: Path | None = None
    language: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers — formatting
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def _fmt_eta(avg_seconds: float, remaining: int) -> str:
    """Estimate remaining time."""
    total = avg_seconds * remaining
    return _fmt_duration(total)


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    """Simple text progress bar: [████████░░░░░░░░] 45%"""
    if total == 0:
        return "[" + "░" * width + "]   0%"
    pct = done / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:5.1%}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_asr(config: AsrConfig) -> dict[str, Any]:
    """
    Main entry point for the ASR command.

    Reads video_index.csv, transcribes eligible videos, writes .txt files,
    and updates video_index.csv. Returns a summary dict.
    """
    channel_dir = config.channel_dir.resolve()
    meta_dir = channel_dir / META_FOLDER
    video_index_path = meta_dir / "video_index.csv"

    if not video_index_path.exists():
        raise FileNotFoundError(
            f"video_index.csv not found in: {meta_dir}\n"
            "Run 'yt-channel-transcriber run <url>' first."
        )

    rows = _read_video_index(video_index_path)
    existing_transcripts = _scan_existing_transcripts(channel_dir)

    # Determine which statuses to process
    target_statuses: set[str] = {STATUS_NO_SUBTITLES}
    if config.include_failed:
        target_statuses.add(STATUS_FAILED)

    eligible_rows = [
        row for row in rows
        if (row.get("status") or "").strip() in target_statuses
    ]

    if config.limit > 0:
        eligible_rows = eligible_rows[: config.limit]

    total_eligible = len(eligible_rows)

    # --- Header ---
    print()
    print("=" * 65)
    print("  ASR Transcription")
    print("=" * 65)
    print(f"  Videos to process : {total_eligible}")
    print(f"  Model             : {config.model}")
    print(f"  Device            : {config.device} ({config.compute_type})")
    print(f"  Language           : {config.language}")
    print(f"  Beam size          : {config.beam_size}")
    print("=" * 65)
    print()

    if total_eligible == 0:
        print("Nothing to do — all videos already have transcripts.")
        return _build_report(config, rows, [], started_at=now_utc_iso())

    # Load model once (expensive operation)
    model = _load_model(config)

    started_at = now_utc_iso()
    results: list[AsrResult] = []
    row_by_id = {row["video_id"]: row for row in rows if row.get("video_id")}

    # Timing for ETA
    processing_times: list[float] = []
    ok_count = 0
    skip_count = 0
    err_count = 0
    wall_start = time.time()

    for index, row in enumerate(eligible_rows, start=1):
        video_id = row.get("video_id", "").strip()
        title = row.get("title", "").strip() or video_id
        url = row.get("url", "").strip()
        channel = row.get("assigned_playlist_title", "").strip()

        if not video_id or not url:
            print(f"  ⚠ [{index}/{total_eligible}] Skipping — missing video_id or url")
            results.append(AsrResult(video_id=video_id, status="skipped", error="Missing video_id or url"))
            skip_count += 1
            continue

        # Determine target path (same folder as indexed)
        folder_rel = row.get("folder", "").strip()
        if folder_rel:
            target_dir = channel_dir / folder_rel
        else:
            target_dir = channel_dir / "Sin_Playlist"

        transcript_name = build_transcript_filename(video_id=video_id, title=title)
        transcript_path = target_dir / transcript_name

        # Skip if already transcribed (unless force)
        if not config.force and video_id in existing_transcripts:
            skip_count += 1
            results.append(AsrResult(video_id=video_id, status="skipped", transcript_path=existing_transcripts[video_id]))
            if config.verbose:
                print(f"  ⏭ [{index}/{total_eligible}] {video_id} — skipped (exists)")
            continue

        # --- Process this video ---
        short_title = (title[:45] + "...") if len(title) > 45 else title

        # Status line: header
        print(f"  ▶ [{index}/{total_eligible}] {short_title}")
        print(f"    ID: {video_id}")

        t0 = time.time()

        # Step 1: Download audio
        print(f"    ↓ Downloading audio...", end="", flush=True)
        dl_start = time.time()
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                audio_path = _download_audio(url=url, tmp_dir=Path(tmp_dir), verbose=config.verbose)
            except Exception as exc:
                dl_time = time.time() - dl_start
                error_msg = str(exc)
                print(f" FAILED ({_fmt_duration(dl_time)})")
                print(f"    ✗ Download error: {error_msg[:120]}")
                result = AsrResult(video_id=video_id, status="error", error=f"Download failed: {error_msg}")
                results.append(result)
                err_count += 1
                if video_id in row_by_id:
                    row_by_id[video_id]["error"] = result.error
                # Save CSV incrementally
                _write_video_index(video_index_path, rows)
                continue

            dl_time = time.time() - dl_start
            print(f" OK ({_fmt_duration(dl_time)})")

            # Step 2: Transcribe
            print(f"    🎙 Transcribing...", end="", flush=True)
            tr_start = time.time()
            try:
                transcript_text, detected_lang = _transcribe_audio(
                    audio_path=audio_path,
                    model=model,
                    config=config,
                )
            except Exception as exc:
                tr_time = time.time() - tr_start
                error_msg = str(exc)
                print(f" FAILED ({_fmt_duration(tr_time)})")
                print(f"    ✗ Transcription error: {error_msg[:120]}")
                result = AsrResult(video_id=video_id, status="error", error=f"Transcription failed: {error_msg}")
                results.append(result)
                err_count += 1
                if video_id in row_by_id:
                    row_by_id[video_id]["error"] = result.error
                # Save CSV incrementally
                _write_video_index(video_index_path, rows)
                continue

            tr_time = time.time() - tr_start
            print(f" OK ({_fmt_duration(tr_time)})")

        # Step 3: Validate
        total_time = time.time() - t0

        if not transcript_text.strip():
            print(f"    ✗ Empty transcript (silent video?)")
            result = AsrResult(video_id=video_id, status="error", error="ASR produced empty transcript")
            results.append(result)
            err_count += 1
            if video_id in row_by_id:
                row_by_id[video_id]["error"] = result.error
            _write_video_index(video_index_path, rows)
            continue

        # Step 4: Write transcript file
        ensure_dir(transcript_path.parent)
        _write_asr_transcript(
            path=transcript_path,
            title=title,
            video_id=video_id,
            url=url,
            channel=channel,
            language=detected_lang or config.language,
            transcript_text=transcript_text,
        )

        # Update in-memory row
        lang_used = detected_lang or config.language
        if video_id in row_by_id:
            row_by_id[video_id]["status"] = STATUS_OK
            row_by_id[video_id]["language"] = lang_used
            row_by_id[video_id]["source"] = SOURCE_ASR
            row_by_id[video_id]["error"] = ""
            row_by_id[video_id]["transcript_path"] = _relative_path_str(transcript_path, channel_dir)

        existing_transcripts[video_id] = transcript_path
        result = AsrResult(
            video_id=video_id,
            status=STATUS_OK,
            transcript_path=transcript_path,
            language=lang_used,
        )
        results.append(result)
        ok_count += 1
        processing_times.append(total_time)

        # Success line + stats
        remaining = total_eligible - index
        avg_time = sum(processing_times) / len(processing_times)
        eta = _fmt_eta(avg_time, remaining)

        print(f"    ✓ Done in {_fmt_duration(total_time)} | lang={lang_used}")
        print(f"    {_progress_bar(index, total_eligible)}  OK:{ok_count}  Err:{err_count}  Skip:{skip_count}  ETA: {eta}")
        print()

        # Save CSV incrementally for crash safety
        _write_video_index(video_index_path, rows)

        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)

    # Final summary
    report = _build_report(config, rows, results, started_at=started_at)
    _print_summary(report, wall_start)
    return report


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------

def _download_audio(url: str, tmp_dir: Path, verbose: bool = False) -> Path:
    """
    Download best audio from a YouTube URL to tmp_dir using yt-dlp.
    Returns the path to the downloaded audio file.
    Converts to WAV via ffmpeg postprocessor for maximum Whisper compatibility.
    """
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install yt-dlp"
        ) from exc

    output_template = str(tmp_dir / "audio.%(ext)s")

    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "ignoreerrors": False,
    }
    if verbose:
        ydl_opts["quiet"] = False
        ydl_opts["no_warnings"] = False
        ydl_opts["noprogress"] = False

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("yt-dlp returned empty metadata — video may be private or unavailable")

    # Find the downloaded WAV file
    wav_candidates = list(tmp_dir.glob("*.wav"))
    if wav_candidates:
        return wav_candidates[0]

    # Fallback: any audio file
    for ext in ("m4a", "webm", "opus", "mp3", "ogg", "flac"):
        candidates = list(tmp_dir.glob(f"*.{ext}"))
        if candidates:
            return candidates[0]

    raise RuntimeError(f"No audio file found in {tmp_dir} after download")


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def _load_model(config: AsrConfig) -> Any:
    """
    Load the faster-whisper WhisperModel. Falls back to CPU if CUDA unavailable.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed.\n"
            "Run: pip install faster-whisper\n"
            "For GPU support also install: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        ) from exc

    device = config.device
    compute_type = config.compute_type

    # Auto-detect CUDA availability
    if device == "cuda":
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU (transcription will be slow)")
                device = "cpu"
                compute_type = "int8"
        except ImportError:
            print("Warning: torch not installed, falling back to CPU")
            device = "cpu"
            compute_type = "int8"

    print(f"Loading Whisper model '{config.model}' on {device} ({compute_type})...")
    model = WhisperModel(config.model, device=device, compute_type=compute_type)
    print("Model loaded.")
    return model


def _transcribe_audio(audio_path: Path, model: Any, config: AsrConfig) -> tuple[str, str]:
    """
    Transcribe an audio file using faster-whisper.
    Returns (transcript_text, detected_language).
    """
    segments, info = model.transcribe(
        str(audio_path),
        language=config.language if config.language else None,
        beam_size=config.beam_size,
        vad_filter=True,          # Voice Activity Detection — reduces hallucinations on silence
        vad_parameters={
            "min_silence_duration_ms": 500,
        },
        word_timestamps=False,    # Not needed for plain text output
    )

    # Collect all segment texts (generator — must be consumed)
    text_parts: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            text_parts.append(text)

    transcript_text = "\n".join(text_parts)
    detected_language = getattr(info, "language", "") or config.language
    return transcript_text, detected_language


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _write_asr_transcript(
    path: Path,
    title: str,
    video_id: str,
    url: str,
    channel: str,
    language: str,
    transcript_text: str,
) -> None:
    """
    Write a transcript .txt file in the same format as transcriber._write_transcript_file.
    """
    lines = [
        f"Title: {title}",
        f"Video ID: {video_id}",
        f"Video URL: {url}",
        f"Channel: {channel}",
        f"Subtitle language: {language}",
        f"Subtitle source: {SOURCE_ASR}",
        f"Generated at: {now_utc_iso()}",
        "",
        "-----",
        "",
        transcript_text,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_video_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_video_index(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIDEO_INDEX_HEADERS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in VIDEO_INDEX_HEADERS})


def _scan_existing_transcripts(channel_dir: Path) -> dict[str, Path]:
    existing: dict[str, Path] = {}
    for file_path in channel_dir.rglob("*.txt"):
        if META_FOLDER in file_path.parts:
            continue
        video_id = extract_video_id_from_stem(file_path.stem)
        if not video_id:
            continue
        if video_id not in existing:
            existing[video_id] = file_path
    return existing


def _relative_path_str(path: Path | None, base: Path) -> str:
    if not path:
        return ""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _build_report(
    config: AsrConfig,
    rows: list[dict[str, str]],
    results: list[AsrResult],
    started_at: str,
) -> dict[str, Any]:
    ok_count = sum(1 for r in results if r.status == STATUS_OK)
    skipped_count = sum(1 for r in results if r.status == "skipped")
    error_count = sum(1 for r in results if r.status == "error")
    return {
        "started_at": started_at,
        "finished_at": now_utc_iso(),
        "channel_dir": str(config.channel_dir.resolve()),
        "model": config.model,
        "device": config.device,
        "compute_type": config.compute_type,
        "language": config.language,
        "total_processed": len(results),
        "ok": ok_count,
        "skipped": skipped_count,
        "errors": error_count,
    }


def _print_summary(report: dict[str, Any], wall_start: float | None = None) -> None:
    elapsed = ""
    if wall_start:
        elapsed = f" in {_fmt_duration(time.time() - wall_start)}"

    print()
    print("=" * 65)
    print(f"  ASR Complete{elapsed}")
    print("=" * 65)
    print(f"  Model      : {report.get('model')} ({report.get('device')}, {report.get('compute_type')})")
    print(f"  Language   : {report.get('language')}")
    print(f"  Processed  : {report.get('total_processed', 0)}")
    print(f"  ✓ OK       : {report.get('ok', 0)}")
    print(f"  ⏭ Skipped  : {report.get('skipped', 0)}")
    print(f"  ✗ Errors   : {report.get('errors', 0)}")
    print("=" * 65)
