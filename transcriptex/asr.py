"""
asr.py — ASR fallback transcription using faster-whisper (GPU/CPU).

Reads the channel's video_index.csv, downloads audio for videos with
status='no_subtitles' (or 'failed' if requested), transcribes them locally
with faster-whisper, and writes .txt files in the same format as the main
transcriber. Updates video_index.csv with the new status.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Queue
from typing import Any

# FIX: Disable symlinks on Windows to avoid [WinError 1314] if not Admin/DevMode
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# FIX: Reconfigure stdout/stderr to UTF-8 at module level so that
# Unicode progress characters (▶, ✓) print correctly on Windows
# consoles (default cp1252) even when run_asr() is called directly
# (bypassing the cli.main() reconfigure).
for _stream in (sys.stdout, sys.stderr):
    try:
        if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() != "utf-8":
            _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from transcriptex.fs_utils import (
    build_transcript_filename,
    ensure_dir,
    extract_video_id_from_stem,
    now_utc_iso,
    sanitize_name,
)
from transcriptex.yt_dlp_utils import (
    PermanentVideoError,
    YouTubeRateLimitError,
    apply_youtube_ejs_opts,
    apply_youtube_throttle_opts,
    audio_format_selector,
    detect_js_runtime,
    ensure_yt_dlp_ejs,
    is_private_video_error,
    is_rate_limit_error,
    is_permanent_download_error,
)

META_FOLDER = "_meta"
STATUS_OK = "ok"
STATUS_NO_SUBTITLES = "no_subtitles"
STATUS_FAILED = "failed"
STATUS_EXISTING = "existing"
# Valid audio that contains no transcribable speech (music, silent slideshows,
# trailers). Terminal: never retried, not an error — just nothing to transcribe.
STATUS_NO_SPEECH = "no_speech"
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
    sleep_seconds: float = 2.0
    include_failed: bool = False
    verbose: bool = False
    cookies: str = ""
    quiet: bool = False
    print_summary: bool = True
    retries: int = 2  # transcription retries per video (OOM / transient GPU errors)
    rate_limit_wait_seconds: float = 60.0  # base backoff when YouTube rate-limits (doubles each retry)
    max_rate_limit_waits: int = 3  # backoffs before aborting the run cleanly
    yt_sleep_interval: float = 0.0  # extra yt-dlp sleep between HTTP requests (0 = fastest)
    download_workers: int = 4  # parallel download threads prefetching audio for the GPU


# Hard cap on the rate-limit backoff so an unattended run never idles for an
# hour — it keeps probing YouTube and resumes as soon as the limit clears.
RATE_LIMIT_MAX_BACKOFF = 900.0  # 15 minutes


@dataclass
class AsrResult:
    """Result of a single video ASR transcription attempt."""

    video_id: str
    status: str  # "ok", "skipped", "error", "unavailable"
    transcript_path: Path | None = None
    language: str = ""
    error: str = ""


@dataclass
class _PreparedVideo:
    """Downloaded (or failed/skipped) video ready for GPU transcription."""

    index: int
    total: int
    row: dict[str, str]
    audio_path: Path | None = None
    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    error_msg: str = ""
    skipped: bool = False
    skip_reason: str = ""


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
            "Run 'transcriptex run <url>' first."
        )

    rows = _read_video_index(video_index_path)
    existing_transcripts = _scan_existing_transcripts(channel_dir)

    # Reconcile disk files with the index (crash-safe resume after partial runs).
    synced = _sync_existing_transcripts_to_index(
        rows=rows,
        existing_transcripts=existing_transcripts,
        channel_dir=channel_dir,
        force=config.force,
    )
    if synced:
        _write_video_index(video_index_path, rows)

    # Determine which statuses to process
    target_statuses: set[str] = {STATUS_NO_SUBTITLES}
    if config.include_failed:
        target_statuses.add(STATUS_FAILED)

    eligible_rows = [
        row for row in rows
        if (row.get("status") or "").strip() in target_statuses
    ]

    # Process subtitle-less videos before optional failed/private retries.
    eligible_rows.sort(
        key=lambda r: (0 if (r.get("status") or "").strip() == STATUS_NO_SUBTITLES else 1)
    )

    # Skip videos already known to be permanently inaccessible (private, etc.).
    if not config.force:
        eligible_rows = [
            row for row in eligible_rows
            if not _is_known_permanent_failure(row)
        ]

    # Skip videos that already have a transcript on disk (unless --force).
    if not config.force:
        eligible_rows = [
            row for row in eligible_rows
            if (row.get("video_id") or "").strip() not in existing_transcripts
        ]

    if config.limit > 0:
        eligible_rows = eligible_rows[: config.limit]

    total_eligible = len(eligible_rows)
    quiet = config.quiet

    # --- Auto-detect GPU VRAM and optimize settings ---
    config = _auto_optimize_config(config, quiet=quiet)

    # --- Header ---
    if not quiet:
        print()
        print("=" * 65)
        print("  ASR Transcription")
        print("=" * 65)
        print(f"  Videos to process : {total_eligible}")
        print(f"  Model             : {config.model}")
        print(f"  Device            : {config.device} ({config.compute_type})")
        print(f"  Language           : {config.language}")
        print(f"  Beam size          : {config.beam_size}")
        print(f"  Download workers   : {config.download_workers}")
        print("=" * 65)
        print()

    if total_eligible == 0:
        if not quiet:
            print("Nothing to do — all videos already have transcripts.")
        return _build_report(config, rows, [], started_at=now_utc_iso())

    # Load model once (expensive operation). model_holder IS the engine dict
    # ({"model", "batched", "device"}) so OOM fallback can swap it in place.
    model_holder: dict[str, Any] = _load_model(config, quiet=quiet)

    started_at = now_utc_iso()
    results: list[AsrResult] = []
    row_by_id = {row["video_id"]: row for row in rows if row.get("video_id")}

    # Timing for ETA
    processing_times: list[float] = []
    ok_count = 0
    skip_count = 0
    err_count = 0
    no_speech_count = 0
    wall_start = time.time()
    interrupted = False
    rate_limited = False

    pipeline_state = {
        "ok_count": 0,
        "skip_count": 0,
        "err_count": 0,
        "no_speech_count": 0,
        "processing_times": processing_times,
        "rate_limited": False,
    }
    state_lock = threading.Lock()

    try:
        _run_asr_pipeline(
            eligible_rows=eligible_rows,
            total_eligible=total_eligible,
            config=config,
            channel_dir=channel_dir,
            meta_dir=meta_dir,
            video_index_path=video_index_path,
            rows=rows,
            row_by_id=row_by_id,
            existing_transcripts=existing_transcripts,
            model_holder=model_holder,
            results=results,
            started_at=started_at,
            quiet=quiet,
            state=pipeline_state,
            state_lock=state_lock,
        )
    except KeyboardInterrupt:
        interrupted = True
        if not quiet:
            print("\n  Interrupted — saving progress to video_index.csv ...")
    finally:
        _write_video_index(video_index_path, rows)

    ok_count = pipeline_state["ok_count"]
    skip_count = pipeline_state["skip_count"]
    err_count = pipeline_state["err_count"]
    no_speech_count = pipeline_state["no_speech_count"]
    rate_limited = pipeline_state["rate_limited"]

    # Final summary
    report = _build_report(config, rows, results, started_at=started_at)
    report["interrupted"] = interrupted
    report["rate_limited"] = rate_limited
    if config.print_summary:
        _print_summary(report, wall_start)
    return report


# ---------------------------------------------------------------------------
# Pipelined download → GPU transcription
# ---------------------------------------------------------------------------

class _PipelineSentinel:
    """Marks end of the download → GPU ready queue."""


_PIPELINE_END = _PipelineSentinel()


def _run_asr_pipeline(
    *,
    eligible_rows: list[dict[str, str]],
    total_eligible: int,
    config: AsrConfig,
    channel_dir: Path,
    meta_dir: Path,
    video_index_path: Path,
    rows: list[dict[str, str]],
    row_by_id: dict[str, dict[str, str]],
    existing_transcripts: dict[str, Path],
    model_holder: dict[str, Any],
    results: list[AsrResult],
    started_at: str,
    quiet: bool,
    state: dict[str, Any],
    state_lock: threading.Lock,
) -> None:
    """
    Decoupled producer/consumer pipeline.

    Download workers fill ``ready_queue`` as soon as each audio file is ready.
    The GPU thread consumes from the queue without waiting for in-flight
    downloads — so if transcription finishes in 15s but the next download
    completed at 10s, the GPU picks it up immediately (zero idle gap).
    """
    workers = max(1, config.download_workers)
    ready_queue: Queue[_PreparedVideo | _PipelineSentinel] = Queue(
        maxsize=workers * 3,
    )
    rate_limit_lock = threading.Lock()
    gpu_lock = threading.Lock()
    videos_since_vram_cleanup = 0

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asr-dl") as executor:
        coordinator = threading.Thread(
            target=_download_coordinator,
            kwargs={
                "eligible_rows": eligible_rows,
                "total_eligible": total_eligible,
                "config": config,
                "executor": executor,
                "ready_queue": ready_queue,
                "workers": workers,
                "quiet": quiet,
                "rate_limit_lock": rate_limit_lock,
                "state": state,
                "state_lock": state_lock,
            },
            name="asr-dl-coordinator",
            daemon=True,
        )
        coordinator.start()

        while True:
            prepared = ready_queue.get()
            if prepared is _PIPELINE_END:
                break

            _process_prepared_video(
                prepared=prepared,
                config=config,
                channel_dir=channel_dir,
                meta_dir=meta_dir,
                video_index_path=video_index_path,
                rows=rows,
                row_by_id=row_by_id,
                existing_transcripts=existing_transcripts,
                model_holder=model_holder,
                results=results,
                started_at=started_at,
                quiet=quiet,
                state=state,
                state_lock=state_lock,
                gpu_lock=gpu_lock,
            )
            if prepared.tmp_dir is not None:
                prepared.tmp_dir.cleanup()

            videos_since_vram_cleanup += 1
            if videos_since_vram_cleanup >= 20:
                _release_gpu_memory(config.device)
                videos_since_vram_cleanup = 0

        coordinator.join(timeout=5.0)


def _download_coordinator(
    *,
    eligible_rows: list[dict[str, str]],
    total_eligible: int,
    config: AsrConfig,
    executor: ThreadPoolExecutor,
    ready_queue: Queue[_PreparedVideo | _PipelineSentinel],
    workers: int,
    quiet: bool,
    rate_limit_lock: threading.Lock,
    state: dict[str, Any],
    state_lock: threading.Lock,
) -> None:
    """Keep ``workers`` downloads in flight; push each result into ``ready_queue``."""
    row_iter = iter(enumerate(eligible_rows, start=1))
    in_flight: dict[Future[_PreparedVideo], None] = {}

    def submit_next() -> bool:
        try:
            index, row = next(row_iter)
        except StopIteration:
            return False
        fut = executor.submit(
            _download_prepared,
            row=row,
            index=index,
            total=total_eligible,
            config=config,
            quiet=quiet,
            rate_limit_lock=rate_limit_lock,
            state=state,
            state_lock=state_lock,
        )
        in_flight[fut] = None
        # Stagger download starts (rate-limit protection) without blocking the GPU.
        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds / workers)
        return True

    for _ in range(min(workers, total_eligible)):
        submit_next()

    while in_flight:
        done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
        for fut in done:
            in_flight.pop(fut)
            prepared = fut.result()
            ready_queue.put(prepared)
            submit_next()

    ready_queue.put(_PIPELINE_END)


def _download_prepared(
    *,
    row: dict[str, str],
    index: int,
    total: int,
    config: AsrConfig,
    quiet: bool,
    rate_limit_lock: threading.Lock,
    state: dict[str, Any],
    state_lock: threading.Lock,
) -> _PreparedVideo:
    """Download audio for one video (runs in a worker thread)."""
    video_id = row.get("video_id", "").strip()
    title = row.get("title", "").strip() or video_id
    url = row.get("url", "").strip()

    if not video_id or not url:
        return _PreparedVideo(
            index=index, total=total, row=row,
            skipped=True, skip_reason="Missing video_id or url",
        )

    short_title = (title[:45] + "...") if len(title) > 45 else title
    if not quiet:
        with state_lock:
            print(f"  ▶ [{index}/{total}] {short_title}")
            print(f"    ID: {video_id}")
            print(f"    ↓ [{index}/{total}] Downloading...", end="", flush=True)

    rate_limit_waits = 0
    while True:
        tmp = tempfile.TemporaryDirectory()
        try:
            dl_start = time.time()
            audio_path = _download_audio(
                url=url,
                tmp_dir=Path(tmp.name),
                verbose=config.verbose,
                cookies=config.cookies,
                max_retries=2,
                sleep_base=2.0,
            )
            if not quiet:
                with state_lock:
                    print(f" OK ({_fmt_duration(time.time() - dl_start)})")
            return _PreparedVideo(
                index=index, total=total, row=row,
                audio_path=audio_path, tmp_dir=tmp,
            )
        except Exception as exc:
            tmp.cleanup()
            if is_rate_limit_error(exc):
                rate_limit_waits += 1
                with state_lock:
                    state["rate_limited"] = True
                exponent = min(rate_limit_waits - 1, config.max_rate_limit_waits)
                backoff = min(
                    config.rate_limit_wait_seconds * (2 ** exponent),
                    RATE_LIMIT_MAX_BACKOFF,
                )
                if not quiet:
                    with state_lock:
                        print(" RATE-LIMITED")
                        print(
                            f"    ⏸ YouTube rate limit (retry #{rate_limit_waits}). "
                            f"Backing off {_fmt_duration(backoff)} then retrying..."
                        )
                with rate_limit_lock:
                    time.sleep(backoff)
                continue
            if not quiet:
                with state_lock:
                    print(" FAILED")
            return _PreparedVideo(
                index=index, total=total, row=row,
                error_msg=str(exc),
            )


def _process_prepared_video(
    *,
    prepared: _PreparedVideo,
    config: AsrConfig,
    channel_dir: Path,
    meta_dir: Path,
    video_index_path: Path,
    rows: list[dict[str, str]],
    row_by_id: dict[str, dict[str, str]],
    existing_transcripts: dict[str, Path],
    model_holder: dict[str, Any],
    results: list[AsrResult],
    started_at: str,
    quiet: bool,
    state: dict[str, Any],
    state_lock: threading.Lock,
    gpu_lock: threading.Lock,
) -> None:
    """Transcribe a prefetched video and update index (GPU thread — main loop)."""
    row = prepared.row
    video_id = row.get("video_id", "").strip()
    title = row.get("title", "").strip() or video_id
    url = row.get("url", "").strip()
    channel = row.get("assigned_playlist_title", "").strip()
    index = prepared.index
    total = prepared.total

    if prepared.skipped:
        if not quiet:
            with state_lock:
                print(f"  ⚠ [{index}/{total}] Skipping — {prepared.skip_reason}")
        with state_lock:
            results.append(AsrResult(video_id=video_id, status="skipped", error=prepared.skip_reason))
            state["skip_count"] += 1
        return

    folder_rel = row.get("folder", "").strip()
    target_dir = channel_dir / folder_rel if folder_rel else channel_dir / "Sin_Playlist"
    transcript_path = target_dir / build_transcript_filename(video_id=video_id, title=title)

    t0 = time.time()

    if prepared.error_msg:
        if not quiet:
            with state_lock:
                phase = "Transcription" if "Transcription failed" in prepared.error_msg else "Download"
                print(f"    ✗ {phase} error: {prepared.error_msg[:120]}")
        with state_lock:
            if video_id in row_by_id:
                if "private video" in prepared.error_msg.lower():
                    row_by_id[video_id]["status"] = STATUS_FAILED
                row_by_id[video_id]["error"] = prepared.error_msg[:500]
            results.append(AsrResult(video_id=video_id, status="error", error=prepared.error_msg))
            state["err_count"] += 1
            _write_video_index(video_index_path, rows)
        return

    transcript_text = ""
    detected_lang = ""
    try:
        if not quiet:
            with state_lock:
                print(f"    🎙 [{index}/{total}] Transcribing...", end="", flush=True)
        tr_start = time.time()
        with gpu_lock:
            transcript_text, detected_lang = _transcribe_audio_with_retry(
                audio_path=prepared.audio_path,  # type: ignore[arg-type]
                model_holder=model_holder,
                config=config,
                quiet=True,
            )
        if not quiet:
            with state_lock:
                print(f" OK ({_fmt_duration(time.time() - tr_start)})")
    except Exception as exc:
        error_msg = str(exc)
        if not quiet:
            with state_lock:
                print(f" FAILED")
                print(f"    ✗ Transcription error: {error_msg[:120]}")
        with state_lock:
            if video_id in row_by_id:
                row_by_id[video_id]["error"] = error_msg[:500]
            results.append(AsrResult(video_id=video_id, status="error", error=error_msg))
            state["err_count"] += 1
            _write_video_index(video_index_path, rows)
        return

    total_time = time.time() - t0

    if not transcript_text.strip():
        if not quiet:
            with state_lock:
                print(f"    ⏭ No speech detected (music/silent) — marked no_speech")
        with state_lock:
            results.append(AsrResult(
                video_id=video_id, status=STATUS_NO_SPEECH,
                error="No speech detected (music/silence)",
            ))
            state["no_speech_count"] += 1
            if video_id in row_by_id:
                row_by_id[video_id]["status"] = STATUS_NO_SPEECH
                row_by_id[video_id]["error"] = "No speech detected (music/silence)"
            _write_video_index(video_index_path, rows)
        return

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

    lang_used = detected_lang or config.language
    with state_lock:
        if video_id in row_by_id:
            row_by_id[video_id]["status"] = STATUS_OK
            row_by_id[video_id]["language"] = lang_used
            row_by_id[video_id]["source"] = SOURCE_ASR
            row_by_id[video_id]["error"] = ""
            row_by_id[video_id]["transcript_path"] = _relative_path_str(transcript_path, channel_dir)

        existing_transcripts[video_id] = transcript_path
        results.append(AsrResult(
            video_id=video_id,
            status=STATUS_OK,
            transcript_path=transcript_path,
            language=lang_used,
        ))
        state["ok_count"] += 1
        state["processing_times"].append(total_time)

        ok = state["ok_count"]
        no_speech = state["no_speech_count"]
        err = state["err_count"]
        skip = state["skip_count"]
        times: list[float] = state["processing_times"]

        _write_video_index(video_index_path, rows)
        _write_asr_progress(
            meta_dir,
            started_at=started_at,
            index=index,
            total=total,
            ok=ok,
            errors=err,
            skipped=skip,
            last_video_id=video_id,
        )

    if not quiet:
        remaining = total - index
        avg_time = sum(times) / len(times) if times else 0.0
        eta = _fmt_eta(avg_time, remaining)
        with state_lock:
            print(f"    ✓ Done in {_fmt_duration(total_time)} | lang={lang_used}")
            print(f"    {_progress_bar(index, total)}  OK:{ok}  NoSpc:{no_speech}  Err:{err}  Skip:{skip}  ETA: {eta}")
            print()


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------

def _auto_optimize_config(config: AsrConfig, quiet: bool = False) -> AsrConfig:
    """
    Detect GPU VRAM and optimize compute_type / batch_size automatically.

    For known cards (e.g. RTX 4060 with 8GB):
      - float16 is safe for large-v3-turbo (~4GB)
      - int8_float16 saves VRAM if needed
      - Reduce batch_size if VRAM < 8GB
    """
    if config.device != "cuda":
        return config

    try:
        from transcriptex.dependency_manager import cuda_available

        if not cuda_available():
            return config

        from transcriptex.dependency_manager import nvidia_gpu_info
        gpu_info = nvidia_gpu_info()
        if gpu_info is None:
            return config  # Unknown VRAM must not be interpreted as a zero-memory GPU.
        device_name, total_vram_mb = gpu_info

        if not quiet:
            if total_vram_mb > 0:
                print(f"  GPU: {device_name} ({total_vram_mb:.0f} MB VRAM)")
            else:
                print(f"  GPU: {device_name}")

        # Make a mutable copy (dataclass replace)
        new_config = AsrConfig(
            channel_dir=config.channel_dir,
            model=config.model,
            device=config.device,
            compute_type=config.compute_type,
            language=config.language,
            beam_size=config.beam_size,
            batch_size=config.batch_size,
            force=config.force,
            limit=config.limit,
            sleep_seconds=config.sleep_seconds,
            include_failed=config.include_failed,
            verbose=config.verbose,
            cookies=config.cookies,
            quiet=config.quiet,
            print_summary=config.print_summary,
            retries=config.retries,
            rate_limit_wait_seconds=config.rate_limit_wait_seconds,
            max_rate_limit_waits=config.max_rate_limit_waits,
            yt_sleep_interval=config.yt_sleep_interval,
            download_workers=config.download_workers,
        )

        # Optimize compute_type based on VRAM
        if total_vram_mb < 6144:  # < 6GB
            if new_config.compute_type in ("float16", "float32"):
                if not quiet:
                    print(f"  VRAM < 6GB — switching to int8 for stability")
                new_config.compute_type = "int8"
            if new_config.batch_size > 4:
                new_config.batch_size = 4
        elif total_vram_mb < 8192:  # 6-8GB (e.g. RTX 4060)
            if new_config.compute_type == "float32":
                if not quiet:
                    print(f"  VRAM ~8GB — switching from float32 to float16")
                new_config.compute_type = "float16"
            # large-v3-turbo is light (~1.5GB) so batch 16 fits comfortably.
            if new_config.batch_size > 16:
                new_config.batch_size = 16
        # > 8GB: keep user settings

        return new_config
    except Exception:
        # If anything goes wrong in detection, just return original config
        return config


def _is_known_permanent_failure(row: dict[str, str]) -> bool:
    """Skip rows that already failed for reasons that won't heal with time."""
    error = (row.get("error") or "").lower()
    if "private video" in error:
        return True
    if "has been removed" in error:
        return True
    if "copyright" in error and "blocked" in error:
        return True
    return False


def _download_audio(
    url: str,
    tmp_dir: Path,
    verbose: bool = False,
    cookies: str = "",
    max_retries: int = 3,
    sleep_base: float = 2.0,
    yt_sleep_interval: float = 0.0,
) -> Path:
    """
    Download the smallest usable audio stream from a YouTube URL.

    No WAV conversion: faster-whisper decodes compressed audio (webm/m4a/opus)
    directly via ffmpeg and resamples to 16 kHz mono internally, so converting
    to a multi-hundred-MB WAV first only wastes disk and time.

    Raises:
        YouTubeRateLimitError  — session temporarily blocked (retry the video later)
        PermanentVideoError    — video is private/removed/blocked (skip forever)
    """
    from transcriptex.yt_dlp_utils import (
        PermanentVideoError,
        YouTubeRateLimitError,
        ensure_yt_dlp_ejs,
    )
    ensure_yt_dlp_ejs()
    import yt_dlp  # type: ignore

    if not detect_js_runtime():
        raise RuntimeError(
            "No JavaScript runtime (node, deno, or bun) found on PATH. "
            "YouTube audio downloads require Node.js (or Deno) since yt-dlp 2025.11. "
            "Install Node.js from https://nodejs.org/ and re-run."
        )

    output_template = str(tmp_dir / "audio.%(ext)s")
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        ydl_opts: dict[str, Any] = apply_youtube_ejs_opts({
            "quiet": True if not verbose else False,
            "no_warnings": True if not verbose else False,
            "noprogress": True if not verbose else False,
            # Smallest audio that still decodes cleanly — speech transcription
            # does not benefit from high-bitrate audio (Whisper downsamples to 16k).
            "format": "worstaudio[abr>=24]/bestaudio/best",
            "outtmpl": output_template,
            "ignoreerrors": False,
            "retries": 2,
            "fragment_retries": 2,
        })
        if yt_sleep_interval > 0:
            ydl_opts = apply_youtube_throttle_opts(
                ydl_opts,
                sleep_interval=yt_sleep_interval,
                sleep_interval_requests=max(0.5, yt_sleep_interval / 2),
            )
        if cookies:
            ydl_opts["cookiefile"] = cookies

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    raise RuntimeError("yt-dlp returned empty metadata")

            for ext in ("m4a", "webm", "opus", "mp3", "ogg", "flac", "wav", "mp4"):
                candidates = list(tmp_dir.glob(f"*.{ext}"))
                if candidates:
                    return candidates[0]
            # Last resort: any file written
            any_file = [p for p in tmp_dir.iterdir() if p.is_file()]
            if any_file:
                return any_file[0]
            raise RuntimeError(f"No audio file found in {tmp_dir} after download")

        except Exception as exc:
            last_error = exc
            if is_rate_limit_error(exc):
                raise YouTubeRateLimitError(str(exc)) from exc
            if is_private_video_error(exc) or is_permanent_download_error(exc):
                raise PermanentVideoError(str(exc)) from exc

            if attempt < max_retries:
                sleep_time = sleep_base * (2 ** (attempt - 1))
                if verbose:
                    print(f"    Download failed (attempt {attempt}/{max_retries}): {exc}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                break

    raise last_error if last_error else RuntimeError("Download failed after all retries")


# ---------------------------------------------------------------------------
# Resume / progress helpers
# ---------------------------------------------------------------------------

def _sync_existing_transcripts_to_index(
    rows: list[dict[str, str]],
    existing_transcripts: dict[str, Path],
    channel_dir: Path,
    force: bool,
) -> int:
    """
    Mark indexed rows as ok when a transcript file already exists on disk.

    Returns the number of rows updated. Skipped when ``force`` is True.
    """
    if force:
        return 0

    updated = 0
    for row in rows:
        video_id = (row.get("video_id") or "").strip()
        if not video_id or video_id not in existing_transcripts:
            continue
        status = (row.get("status") or "").strip()
        if status in (STATUS_OK, STATUS_EXISTING):
            continue

        path = existing_transcripts[video_id]
        row["status"] = STATUS_OK
        row["transcript_path"] = _relative_path_str(path, channel_dir)
        if not (row.get("source") or "").strip():
            row["source"] = SOURCE_ASR
        row["error"] = ""
        updated += 1
    return updated


def _write_asr_progress(
    meta_dir: Path,
    *,
    started_at: str,
    index: int,
    total: int,
    ok: int,
    errors: int,
    skipped: int,
    last_video_id: str,
) -> None:
    """Write a lightweight progress snapshot for overnight / agent monitoring."""
    payload = {
        "updated_at": now_utc_iso(),
        "started_at": started_at,
        "processed_in_run": index,
        "total_in_run": total,
        "ok": ok,
        "errors": errors,
        "skipped": skipped,
        "last_video_id": last_video_id,
    }
    try:
        (meta_dir / "asr_progress.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _release_gpu_memory(device: str) -> None:
    """Collect released objects; do not install another inference engine to clean CT2."""
    import gc
    gc.collect()
    torch = sys.modules.get('torch')
    if device == 'cuda' and torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg and "memory" in msg


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def _load_model(config: AsrConfig, quiet: bool = False) -> Any:
    """
    Load the faster-whisper WhisperModel. Falls back to CPU if CUDA unavailable.
    """
    from transcriptex.dependency_manager import ensure_faster_whisper
    ensure_faster_whisper()
    from faster_whisper import WhisperModel  # type: ignore

    device = config.device
    compute_type = config.compute_type

    # Auto-detect CUDA availability
    if device == "cuda":
        from transcriptex.dependency_manager import cuda_available

        if not cuda_available():
            if not quiet:
                print("Warning: CUDA not available, falling back to CPU (transcription will be slow)")
            device = "cpu"
            compute_type = "int8"

    if not quiet:
        print(f"Loading Whisper model '{config.model}' on {device} ({compute_type})...")
    model = WhisperModel(config.model, device=device, compute_type=compute_type)

    # Batched inference is ~3-4x faster for long files. Only worth it on GPU.
    batched = None
    if device == "cuda" and config.batch_size > 1:
        try:
            from faster_whisper import BatchedInferencePipeline  # type: ignore
            batched = BatchedInferencePipeline(model=model)
        except Exception:
            batched = None

    if not quiet:
        mode = f"batched x{config.batch_size}" if batched is not None else "sequential"
        print(f"Model loaded ({mode}).")
    return {"model": model, "batched": batched, "device": device}


def _transcribe_audio(audio_path: Path, engine: dict[str, Any], config: AsrConfig) -> tuple[str, str]:
    """
    Transcribe an audio file using faster-whisper (batched when available).
    Returns (transcript_text, detected_language).
    """
    language = config.language if config.language else None
    vad_parameters = {"min_silence_duration_ms": 500}

    batched = engine.get("batched")
    if batched is not None:
        segments, info = batched.transcribe(
            str(audio_path),
            language=language,
            beam_size=config.beam_size,
            vad_filter=True,
            vad_parameters=vad_parameters,
            batch_size=config.batch_size,
        )
    else:
        segments, info = engine["model"].transcribe(
            str(audio_path),
            language=language,
            beam_size=config.beam_size,
            vad_filter=True,
            vad_parameters=vad_parameters,
            word_timestamps=False,
        )

    text_parts: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            text_parts.append(text)

    transcript_text = "\n".join(text_parts)
    detected_language = getattr(info, "language", "") or config.language
    return transcript_text, detected_language


def _transcribe_audio_with_retry(
    audio_path: Path,
    model_holder: dict[str, Any],
    config: AsrConfig,
    quiet: bool = False,
) -> tuple[str, str]:
    """
    Transcribe with retries. On CUDA OOM, release VRAM and reload the model
    once with a lighter compute_type before giving up.
    """
    max_attempts = max(1, config.retries)
    last_error: Exception | None = None
    oom_fallback_used = False

    for attempt in range(1, max_attempts + 1):
        try:
            return _transcribe_audio(audio_path, model_holder, config)
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break

            if _is_oom_error(exc) and config.device == "cuda" and not oom_fallback_used:
                oom_fallback_used = True
                if not quiet:
                    print(
                        f"\n    ⚠ GPU OOM — reloading model with int8_float16 ...",
                        flush=True,
                    )
                model_holder.clear()
                _release_gpu_memory(config.device)
                lighter = replace(
                    config,
                    compute_type="int8_float16",
                    batch_size=max(1, config.batch_size // 2),
                )
                config = lighter
                model_holder.update(_load_model(lighter, quiet=quiet))
                continue

            if not quiet:
                print(
                    f"\n    ⚠ Transcription failed (attempt {attempt}/{max_attempts}): "
                    f"{str(exc)[:80]} — retrying...",
                    flush=True,
                )
            _release_gpu_memory(config.device)
            time.sleep(min(2.0 * attempt, 10.0))

    raise last_error if last_error else RuntimeError("Transcription failed")


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
    no_speech_count = sum(1 for r in results if r.status == STATUS_NO_SPEECH)
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
        "no_speech": no_speech_count,
    }


def _print_summary(report: dict[str, Any], wall_start: float | None = None) -> None:
    elapsed = ""
    if wall_start:
        elapsed = f" in {_fmt_duration(time.time() - wall_start)}"

    print()
    print("=" * 65)
    label = "ASR Complete"
    if report.get("interrupted"):
        label = "ASR Interrupted (progress saved)"
    print(f"  {label}{elapsed}")
    print("=" * 65)
    print(f"  Model      : {report.get('model')} ({report.get('device')}, {report.get('compute_type')})")
    print(f"  Language   : {report.get('language')}")
    print(f"  Processed  : {report.get('total_processed', 0)}")
    print(f"  ✓ OK       : {report.get('ok', 0)}")
    print(f"  ⏭ Skipped  : {report.get('skipped', 0)}")
    print(f"  ♪ No speech : {report.get('no_speech', 0)}")
    print(f"  ✗ Errors   : {report.get('errors', 0)}")
    if report.get("rate_limited"):
        print("  ⚠ Hit YouTube rate limits during the run (auto-retried).")
    print("=" * 65)
