"""
transcriptex.cli — Command-line interface for the transcriptEX toolkit.

This module is the single entry point for every subcommand. It defines
the argument parser, the command dispatcher and a thin wrapper around
each subcommand's business logic. The CLI is designed to be friendly to
both humans and AI agents:

- Every subcommand has a multi-line ``description`` and an ``epilog`` with
  examples and documented exit codes.
- All reporting subcommands accept ``--json`` to print a single,
  machine-readable JSON object on stdout instead of the human summary.
- All subcommands accept ``--quiet`` to suppress the human summary.
- The top-level help shows every subcommand, a quick-start recipe,
  environment variables, and a summary of exit codes — all in one place.
- Exit codes are consistent across subcommands:
    0  success
    1  fatal error (bad arguments, missing files, network failure)
    2  completed with warnings (some items failed, no matches, etc.)
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, nullcontext
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

from transcriptex import __version__
from transcriptex.audit import AuditConfig, audit_channel_output
from transcriptex.fs_utils import normalize_list_arg
from transcriptex.scan import ScanConfig, scan_channel
from transcriptex.search import SearchConfig, parse_comma_int_list, search_channel_output
from transcriptex.transcriber import DEFAULT_TABS, ChannelTranscriber, TranscriberConfig

# ---------------------------------------------------------------------------
# Subcommand constants
# ---------------------------------------------------------------------------

RUN_COMMAND = "run"
AUDIT_COMMAND = "audit"
SEARCH_COMMAND = "search"
ASR_COMMAND = "asr"
TRANSCRIBE_COMMAND = "transcribe"
REALTIME_COMMAND = "realtime"
SCAN_COMMAND = "scan"
COOKIES_COMMAND = "cookies"

KNOWN_SUBCOMMANDS = {
    RUN_COMMAND,
    AUDIT_COMMAND,
    SEARCH_COMMAND,
    ASR_COMMAND,
    TRANSCRIBE_COMMAND,
    REALTIME_COMMAND,
    SCAN_COMMAND,
    COOKIES_COMMAND,
}

# Subcommands that produce a structured report and accept --json / --quiet.
REPORTING_SUBCOMMANDS = {
    RUN_COMMAND,
    AUDIT_COMMAND,
    SEARCH_COMMAND,
    ASR_COMMAND,
    SCAN_COMMAND,
    TRANSCRIBE_COMMAND,
}

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_WARNINGS = 2

# ---------------------------------------------------------------------------
# Shared help text fragments
# ---------------------------------------------------------------------------

COOKIES_HELP = (
    "Path to a cookies.txt file (Netscape format) for authenticated "
    "indexing/downloading of members-only, age-restricted, or private "
    "videos. If not provided, transcriptex will use stored cookies from "
    "'transcriptex cookies'. Empty = no cookies."
)

QUIET_HELP = (
    "Suppress the human-readable summary printed at the end. "
    "Diagnostics go to stderr with --json; stdout contains only the JSON report."
)

JSON_HELP = (
    "Print a machine-readable JSON report to stdout instead of the "
    "human-readable summary. Exit codes remain identical. "
    "Useful for piping into jq, scripts, or AI agents."
)

# Top-level description
TOP_DESCRIPTION = (
    "transcriptex — YouTube transcript CLI (no API key required).\n"
    "\n"
    "Export transcripts from YouTube channels, transcribe local media, capture\n"
    "live audio, audit transcript collections, and search across transcripts —\n"
    "all from a single tool that uses yt-dlp and faster-whisper under the hood."
)

TOP_EPILOG = (
    "Quick start:\n"
    "  transcriptex cookies\n"
    "  transcriptex run \"https://www.youtube.com/@channel\" -o ./output\n"
    "  transcriptex scan \"output/My Channel\" --no-interactive\n"
    "  transcriptex search \"output/My Channel\" \"keyword\"\n"
    "  transcriptex asr \"output/My Channel\" --device cuda --language es --include-failed\n"
    "  transcriptex transcribe video.mp4 --format srt\n"
    "\n"
    "Subcommands:\n"
    "  run         Export subtitles from every video in a YouTube channel.\n"
    "  scan        Cross-reference a channel's videos with transcripts on disk.\n"
    "  audit       Compare a channel's index against actual transcript files.\n"
    "  search      Search transcripts by keyword, phrase, regex, or video ID.\n"
    "  asr         Transcribe 'no_subtitles' videos locally with Whisper (GPU).\n"
    "  transcribe  Transcribe a single local file or YouTube URL with Whisper.\n"
    "  realtime    Live transcription of system audio (Windows WASAPI loopback).\n"
    "  cookies     Set up or manage YouTube cookies for authenticated downloads.\n"
    "\n"
    "Environment variables:\n"
    "  PYTHONIOENCODING=utf-8     Force UTF-8 stdout (recommended on Windows).\n"
    "  HF_HUB_DISABLE_SYMLINKS=1  Auto-set on import; avoids symlink errors.\n"
    "\n"
    "Exit codes (consistent across every subcommand):\n"
    "  0  success — every item processed without errors\n"
    "  1  fatal error — bad arguments, missing files, network failure\n"
    "  2  completed with warnings — some items failed, no matches, etc.\n"
    "\n"
    "See README.md for the full command reference and architecture overview."
)

# Per-subcommand epilogs
EPILOG_RUN = (
    "Examples:\n"
    "  transcriptex run \"https://www.youtube.com/@mkbhd\" -o ./output\n"
    "  transcriptex run \"https://www.youtube.com/@mkbhd\" --lang-priority en,es --retries 3\n"
    "  transcriptex run \"https://www.youtube.com/@mkbhd\" --cookies cookies.txt\n"
    "  transcriptex run \"https://www.youtube.com/channel/UCxxxxx\" --force\n"
    "\n"
    "Exit codes:\n"
    "  0  every video processed successfully\n"
    "  1  fatal error\n"
    "  2  some videos failed (see video_index.csv error column)"
)

EPILOG_AUDIT = (
    "Examples:\n"
    "  transcriptex audit \"output/My Channel\"\n"
    "  transcriptex audit \"output/My Channel\" --max-rows 25\n"
    "  transcriptex audit \"output/My Channel\" --rename-legacy --no-write-files\n"
    "\n"
    "Exit codes:\n"
    "  0  index matches the transcripts on disk\n"
    "  1  fatal error (no _meta/video_index.csv found)\n"
    "  2  at least one indexed video is missing its transcript"
)

EPILOG_SEARCH = (
    "Examples:\n"
    "  transcriptex search \"output/My Channel\" \"inteligencia artificial\"\n"
    "  transcriptex search \"output/My Channel\" \"quijote\" --fields title,transcript --match-all\n"
    "  transcriptex search \"output/My Channel\" \"cervantes\" --export-top 10 --export-dir ./results\n"
    "  transcriptex search \"output/My Channel\" \"regex.*pattern\" --regex --top 5\n"
    "\n"
    "Exit codes:\n"
    "  0  at least one match found\n"
    "  1  fatal error\n"
    "  2  zero matches"
)

EPILOG_ASR = (
    "Examples:\n"
    "  transcriptex asr \"output/My Channel\" --device cuda --language es\n"
    "  transcriptex asr \"output/My Channel\" --download-workers 3   # prefetch while GPU works\n"
    "  transcriptex asr \"output/My Channel\" --batch-size 16        # faster on GPU\n"
    "  transcriptex asr \"output/My Channel\" --include-failed --cookies cookies.txt\n"
    "\n"
    "Speed & rate limits:\n"
    "  - Batched GPU inference + compressed audio (no WAV) = much faster.\n"
    "  - Public videos download WITHOUT cookies on purpose: account cookies\n"
    "    make YouTube rate-limit/ban far more aggressively. Use --cookies or\n"
    "    --include-failed only for private/members content.\n"
    "  - On rate limit the run backs off, then stops cleanly. Re-run the same\n"
    "    command later to resume (already-done videos are skipped).\n"
    "\n"
    "Exit codes:\n"
    "  0  every targeted video processed successfully\n"
    "  1  fatal error (no _meta/video_index.csv found)\n"
    "  2  one or more videos failed, or run stopped/interrupted (progress saved)"
)

EPILOG_TRANSCRIBE = (
    "Examples:\n"
    "  transcriptex transcribe video.mp4\n"
    "  transcriptex transcribe video.mp4 --format srt --language es\n"
    "  transcriptex transcribe video.mp4 --device cpu --model medium\n"
    "  transcriptex transcribe \"https://www.youtube.com/watch?v=abc\" --cookies cookies.txt\n"
    "\n"
    "Exit codes:\n"
    "  0  all requested transcripts written successfully\n"
    "  1  fatal error for a single input\n"
    "  2  one or more inputs failed in a multi-input run"
)

EPILOG_REALTIME = (
    "Examples:\n"
    "  transcriptex realtime\n"
    "  transcriptex realtime --model medium --language es --chunk-duration 3.0\n"
    "  transcriptex realtime --list-devices\n"
    "  transcriptex realtime --process zen.exe --language en\n"
    "\n"
    "Notes:\n"
    "  - Linux: PulseAudio/PipeWire (parec); Windows: WASAPI.\n"
    "  - Linux requires --output; partial previews are not confirmed text.\n"
    "  - Linux finalizes a full-recording transcript on Ctrl+C by default.\n"
    "  - Press Ctrl+C to stop."
)

EPILOG_SCAN = (
    "Examples:\n"
    "  transcriptex scan \"https://www.youtube.com/@mkbhd\" --no-interactive -o ./output\n"
    "  transcriptex scan \"output/My Channel\" --no-interactive\n"
    "  transcriptex scan \"https://www.youtube.com/@mkbhd\" -o ./output\n"
    "  transcriptex scan \"https://www.youtube.com/@mkbhd\" --cookies cookies.txt\n"
    "\n"
    "Modes:\n"
    "  URL mode   discovers the channel, fetches the uploads playlist (UU...)\n"
    "             and every named playlist, then cross-references with disk.\n"
    "  folder     reuses an existing _meta/video_index.csv (no network call).\n"
    "\n"
    "Exit codes:\n"
    "  0  every video already has a transcript on disk\n"
    "  1  fatal error\n"
    "  2  at least one video is pending (no transcript on disk)"
)


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriptex",
        description=TOP_DESCRIPTION,
        epilog=TOP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"transcriptex {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="SUBCOMMAND")

    # --- run ---------------------------------------------------------------
    run_parser = subparsers.add_parser(
        RUN_COMMAND,
        help="Export subtitles from every video in a YouTube channel.",
        description=(
            "Export transcripts from all videos in a YouTube channel, organized by "
            "playlists, without using the official YouTube API. Reads the channel's "
            "uploads playlist (UU...) for every video and per-playlist endpoints in "
            "parallel, then writes one .txt file per video into the matching playlist "
            "folder. Videos in multiple playlists are physically copied into each."
        ),
        epilog=EPILOG_RUN,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "channel_url",
        help=(
            "YouTube channel URL. Accepts @handle (https://www.youtube.com/@mkbhd), "
            "/channel/UCxxxxx, or /c/Name or /user/Name forms."
        ),
    )
    run_parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output root directory; the channel folder is created inside it (default: ./output).",
    )
    run_parser.add_argument(
        "--lang-priority",
        default="es,en",
        help=(
            "Comma-separated ISO 639-1 language preference in priority order "
            "(default: es,en). Examples: 'en', 'es,en,fr', 'pt-BR,en'. "
            "Manual tracks are tried first, then auto-generated."
        ),
    )
    run_parser.add_argument(
        "--tabs",
        default=",".join(DEFAULT_TABS),
        help=(
            "Comma-separated channel tabs to use as a fallback when the uploads "
            "playlist is unavailable. Available: videos, shorts, streams, "
            "playlists (default: " + ",".join(DEFAULT_TABS) + ")."
        ),
    )
    run_parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry attempts for metadata/subtitle retrieval (default: 2).",
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Network timeout in seconds (default: 30).",
    )
    run_parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between videos — useful to avoid rate-limits (default: 0).",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate transcript files even if they already exist on disk.",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable yt-dlp verbose output (very chatty).",
    )
    run_parser.add_argument("--cookies", default="", help=COOKIES_HELP)
    _add_report_flags(run_parser)

    # --- audit -------------------------------------------------------------
    audit_parser = subparsers.add_parser(
        AUDIT_COMMAND,
        help="Compare a channel's index against actual transcript files.",
        description=(
            "Audit a previously-exported channel output folder. Compares the "
            "_meta/video_index.csv against transcript .txt files on disk, reports "
            "videos indexed without a transcript, transcript files not in the index "
            "(orphans), and duplicate transcript IDs. Optionally renames legacy "
            "files named only by VIDEO_ID to 'Title [VIDEO_ID].txt'."
        ),
        epilog=EPILOG_AUDIT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    audit_parser.add_argument(
        "channel_dir",
        help="Path to channel output folder (e.g. output/My Channel). Must contain _meta/video_index.csv.",
    )
    audit_parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="Max missing rows to print in the console summary (default: 10).",
    )
    audit_parser.add_argument(
        "--no-write-files",
        action="store_true",
        help="Do not write audit_report.json / *.csv files into _meta.",
    )
    audit_parser.add_argument(
        "--rename-legacy",
        action="store_true",
        help="Rename legacy transcript files named only by VIDEO_ID into 'Title [VIDEO_ID].txt'.",
    )
    _add_report_flags(audit_parser)

    # --- search ------------------------------------------------------------
    search_parser = subparsers.add_parser(
        SEARCH_COMMAND,
        help="Search transcripts by keyword, phrase, regex, or video ID.",
        description=(
            "Search across the metadata and text of every transcript in a channel "
            "output folder. Results are ranked with phrase/token weights per field "
            "(video_id highest, transcript lowest) and can optionally be exported to "
            "a folder and/or zipped."
        ),
        epilog=EPILOG_SEARCH,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    search_parser.add_argument(
        "channel_dir",
        help="Path to channel output folder (e.g. output/My Channel).",
    )
    search_parser.add_argument(
        "query",
        help="Search query — keyword, phrase, regex (with --regex), or exact video ID.",
    )
    search_parser.add_argument(
        "--fields",
        default="title,playlist,video_id,transcript",
        help=(
            "Comma-separated fields to search across. Available: title, playlist, "
            "video_id, transcript (default: title,playlist,video_id,transcript)."
        ),
    )
    search_parser.add_argument(
        "--status",
        default="",
        help=(
            "Comma-separated status filter. Available: ok, existing, no_subtitles, "
            "no_text, failed. Empty = all statuses."
        ),
    )
    search_parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Maximum results shown on the console (default: 25).",
    )
    search_parser.add_argument(
        "--match-all",
        action="store_true",
        help="Require every query token to match (AND logic, default: OR).",
    )
    search_parser.add_argument(
        "--regex",
        action="store_true",
        help="Treat the query as a Python regular expression.",
    )
    search_parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Case-sensitive matching (default: case-insensitive).",
    )
    search_parser.add_argument(
        "--snippet-len",
        type=int,
        default=140,
        help="Character length of transcript match snippets (default: 140).",
    )
    search_parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Write a CSV of the top results into _meta/.",
    )
    search_parser.add_argument(
        "--export-dir",
        default="",
        help="Directory to copy matched transcript files into (default: _meta/_exports/search_<ts>).",
    )
    search_parser.add_argument(
        "--export-top",
        type=int,
        default=0,
        help="Export the top N results (default: 0 = no export).",
    )
    search_parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export every result shown on the console.",
    )
    search_parser.add_argument(
        "--export-ids",
        default="",
        help="Export by specific video IDs (comma-separated).",
    )
    search_parser.add_argument(
        "--export-indices",
        default="",
        help="Export by result indices shown on the console (comma-separated, 1-based).",
    )
    search_parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a ZIP archive of the exported files.",
    )
    _add_report_flags(search_parser)

    # --- asr ---------------------------------------------------------------
    asr_parser = subparsers.add_parser(
        ASR_COMMAND,
        help="Transcribe 'no_subtitles' videos locally with Whisper (GPU).",
        description=(
            "Read _meta/video_index.csv, download audio for every video with "
            "status='no_subtitles' (or 'failed' with --include-failed), transcribe "
            "them locally with faster-whisper, and write .txt files. The CSV is "
            "updated incrementally after each video (crash-safe and resumable). "
            "Re-running the same command skips completed transcripts automatically. "
            "Writes _meta/asr_progress.json during long batches."
        ),
        epilog=EPILOG_ASR,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    asr_parser.add_argument(
        "channel_dir",
        help="Path to channel output folder (e.g. output/My Channel). Must contain _meta/video_index.csv.",
    )
    asr_parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help=(
            "Whisper model name. Available: tiny, base, small, medium, large-v2, "
            "large-v3, large-v3-turbo (default: large-v3-turbo). Larger = better "
            "quality, slower."
        ),
    )
    asr_parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device (default: cuda). Falls back to cpu/int8 if CUDA unavailable.",
    )
    asr_parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "int8_float16", "int8", "float32"],
        help="Model quantization (default: float16). Use int8_float16 to save VRAM.",
    )
    asr_parser.add_argument(
        "--language",
        default="es",
        help="Language code for transcription (default: es). Use '' for auto-detect.",
    )
    asr_parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help=(
            "Beam search size. 1 (default) is fastest and works great with "
            "large-v3-turbo; raise to 5 only if you need max quality."
        ),
    )
    asr_parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batched-inference chunk size on GPU — higher = faster (default: 16).",
    )
    asr_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even if a transcript file already exists.",
    )
    asr_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of videos to process (default: 0 = no limit).",
    )
    asr_parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help=(
            "Seconds to wait between videos. Small values keep it fast; "
            "raise to 5-10 only if you hit rate limits (default: 2)."
        ),
    )
    asr_parser.add_argument(
        "--rate-limit-wait",
        type=float,
        default=60.0,
        help=(
            "Base backoff (seconds) when YouTube rate-limits the session. "
            "Doubles on each retry, then the run aborts cleanly so you can "
            "resume later (default: 60)."
        ),
    )
    asr_parser.add_argument(
        "--max-rate-limit-waits",
        type=int,
        default=3,
        help="Backoff retries before aborting the run on persistent rate limits (default: 3).",
    )
    asr_parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help=(
            "Parallel download threads prefetching audio into a buffer queue "
            "while the GPU transcribes (default: 4). Raise to 6 if GPU still "
            "has idle gaps; lower to 2 if YouTube rate-limits."
        ),
    )
    asr_parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help=(
            "Retry attempts per video when transcription fails transiently "
            "(GPU OOM, etc.). On OOM, reloads the model with int8_float16 once "
            "(default: 2)."
        ),
    )
    asr_parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also retry videos with status 'failed' (not just 'no_subtitles').",
    )
    asr_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose yt-dlp output during audio download.",
    )
    asr_parser.add_argument("--cookies", default="", help=COOKIES_HELP)
    _add_report_flags(asr_parser)

    # --- transcribe --------------------------------------------------------
    transcribe_parser = subparsers.add_parser(
        TRANSCRIBE_COMMAND,
        help="Transcribe media files or video URLs (subtitles first, local ASR fallback).",
        description=(
            "Transcribe local audio/video files (MP4, MKV, MP3, WAV, etc.) or "
            "video URLs. URLs prefer existing subtitles by default; otherwise local "
            "faster-whisper performs ASR. Files are read directly. Audio is downloaded "
            "and deleted after transcription."
        ),
        epilog=EPILOG_TRANSCRIBE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    transcribe_parser.add_argument(
        "input_path", nargs="+",
        help="One or more local media files or video URLs; reuses one ASR model.",
    )
    transcribe_parser.add_argument(
        "-o", "--output",
        default="",
        help=(
            "Output directory. Default: same folder as a local input file, or the "
            "current directory for a YouTube URL."
        ),
    )
    transcribe_parser.add_argument(
        "--format",
        default="txt",
        choices=["txt", "srt", "vtt", "json"],
        help=(
            "Output format (default: txt). "
            "txt = clean plain text; srt = numbered subtitles with timestamps; "
            "vtt = WebVTT subtitles; json = full segment data as JSON."
        ),
    )
    transcribe_parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help=(
            "Whisper model name. Available: tiny, base, small, medium, large-v2, "
            "large-v3, large-v3-turbo (default: large-v3-turbo)."
        ),
    )
    transcribe_parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device (default: auto — uses GPU if available, else CPU).",
    )
    transcribe_parser.add_argument(
        "--compute-type",
        default="auto",
        help=(
            "Quantization: auto, float16, int8_float16, int8, float32 "
            "(default: auto — float16 for GPU, int8 for CPU)."
        ),
    )
    transcribe_parser.add_argument(
        "--language",
        default="",
        help="Language code, e.g. 'es' or 'en' (default: auto-detect).",
    )
    transcribe_parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam search size (default: 1). Use 5 to explore more candidates, at greater cost.",
    )
    transcribe_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    transcribe_parser.add_argument(
        "--cookies",
        default="",
        help=COOKIES_HELP + " (only used for YouTube URL input)",
    )
    transcribe_parser.add_argument("--source", choices=["auto", "subtitles", "asr"], default="auto",
        help="URL source: auto prefers existing subtitles, subtitles never runs ASR, asr forces recognition.")
    transcribe_parser.add_argument("--lang-priority", default="",
        help="Subtitle language preferences, comma-separated (default: --language or es,en).")
    transcribe_parser.add_argument("--batch-size", type=int, default=0,
        help="ASR batch size: 0 selects 8 on CUDA / 1 on CPU; 1 disables batching. Reduced on GPU OOM.")
    transcribe_parser.add_argument("--no-vad", action="store_true",
        help="Disable silence filtering and batching for file ASR.")
    _add_report_flags(transcribe_parser)

    # --- realtime ----------------------------------------------------------
    realtime_parser = subparsers.add_parser(
        REALTIME_COMMAND,
        help="Live system/microphone transcription on Linux or Windows.",
        description=(
            "Capture audio playing on your PC (browsers, media players, meetings, "
            "etc.) via WASAPI Loopback and transcribe it live to the console. "
            "Linux uses PulseAudio/PipeWire (parec); Windows uses pyaudiowpatch."
        ),
        epilog=EPILOG_REALTIME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    realtime_parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Whisper model name (default: large-v3-turbo).",
    )
    realtime_parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device (default: auto — CUDA when available).",
    )
    realtime_parser.add_argument(
        "--compute-type",
        default="auto",
        help="Model quantization (default: auto — float16 on CUDA / int8 on CPU).",
    )
    realtime_parser.add_argument(
        "--language",
        default="",
        help="Language code (default: auto-detect).",
    )
    realtime_parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam search size (default: 1 — fastest for live use).",
    )
    realtime_parser.add_argument(
        "--chunk-duration",
        type=float,
        default=3.0,
        help="Linux partial-preview interval / Windows buffer duration (default: 3.0 s).",
    )
    realtime_parser.add_argument(
        "--process",
        default="",
        help=(
            "Capture audio only from a specific process by name (e.g. 'zen.exe'). "
            "Requires Windows 10/11 Application Loopback."
        ),
    )
    realtime_parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List PulseAudio/PipeWire sources on Linux or WASAPI devices on Windows.",
    )
    realtime_parser.add_argument(
        "--device-index",
        type=int,
        default=-1,
        help="Select audio device by index (use --list-devices to see available devices).",
    )

    realtime_parser.add_argument("--source", default="@DEFAULT_MONITOR@", help="Linux PulseAudio source name.")
    realtime_parser.add_argument("--microphone", default="", help="Optional Linux microphone source.")
    realtime_parser.add_argument("--monitor-stream", type=int, default=-1, help="Linux sink-input index to isolate application audio.")
    realtime_parser.add_argument("--output", default="", help="Linux UTF-8 TXT output, appended and flushed continuously.")

    realtime_parser.add_argument("--max-segment-duration", type=float, default=15.0,
        help="Linux maximum speech segment before an overlapping cut (default: 15 s).")
    realtime_parser.add_argument("--silence-duration", type=float, default=0.6,
        help="Linux pause needed to confirm a segment (default: 0.6 s).")
    realtime_parser.add_argument("--silence-threshold", type=float, default=0.002,
        help="Linux RMS threshold for pause detection (default: 0.002).")
    realtime_parser.add_argument("--overlap-duration", type=float, default=1.0,
        help="Linux audio retained across forced cuts (default: 1 s).")
    realtime_parser.add_argument("--no-preview", action="store_true",
        help="Linux: only print confirmed segments, reducing GPU work.")
    realtime_parser.add_argument("--no-finalize", action="store_true",
        help="Linux: skip full-recording transcription on stop; temporary audio is not retained.")
    realtime_parser.add_argument("--record-audio", action="store_true",
        help="Linux: retain per-source WAV files beside the TXT (otherwise temporary audio is removed).")
    realtime_parser.add_argument("--queue-seconds", type=float, default=30.0,
        help="Linux maximum queued audio per source; fail explicitly on overflow (default: 30 s).")

    # --- scan --------------------------------------------------------------
    scan_parser = subparsers.add_parser(
        SCAN_COMMAND,
        help="Cross-reference a channel's videos with transcripts on disk.",
        description=(
            "Scan a YouTube channel (or a previously-exported channel folder) and "
            "report transcript coverage. In URL mode, discovers the channel via the "
            "uploads playlist (UU...) and every named playlist, then cross-references "
            "with transcript files on disk. In folder mode, reuses the existing "
            "_meta/video_index.csv without making any network call. Writes "
            "_meta/channel_status.json with a machine-readable snapshot."
        ),
        epilog=EPILOG_SCAN,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scan_parser.add_argument(
        "channel_url_or_dir",
        help="YouTube channel URL or local path to a channel folder.",
    )
    scan_parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output root directory for URL mode (default: ./output).",
    )
    scan_parser.add_argument(
        "--cookies",
        default="",
        help=COOKIES_HELP,
    )
    scan_parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Print the summary and exit (no terminal menu). Use in CI / non-TTY environments.",
    )
    scan_parser.add_argument(
        "--page-size",
        type=int,
        default=20,
        help="Number of videos per page in the interactive browser (default: 20).",
    )
    _add_report_flags(scan_parser)

    # --- cookies -----------------------------------------------------------
    cookies_parser = subparsers.add_parser(
        COOKIES_COMMAND,
        help="Set up or manage YouTube cookies for authenticated downloads.",
        description=(
            "Extract YouTube cookies from your browser using yt-dlp's built-in "
            "cookie extraction. Cookies are saved persistently and used automatically "
            "by all subcommands that download from YouTube."
        ),
        epilog=(
            "Examples:\n"
            "  transcriptex cookies\n"
            "  transcriptex cookies --status\n"
            "  transcriptex cookies --remove\n"
            "\n"
            "Exit codes:\n"
            "  0  cookies configured successfully\n"
            "  1  fatal error (no browsers found, extraction failed)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cookies_parser.add_argument(
        "--status",
        action="store_true",
        help="Show current cookie configuration and exit.",
    )
    cookies_parser.add_argument(
        "--remove",
        action="store_true",
        help="Delete stored cookies.",
    )

    return parser


def _add_report_flags(parser: argparse.ArgumentParser) -> None:
    """Attach --quiet and --json flags to a reporting subcommand."""
    parser.add_argument("--quiet", action="store_true", help=QUIET_HELP)
    parser.add_argument("--json", action="store_true", help=JSON_HELP)


def _resolve_cookies(explicit: str) -> str:
    """
    Return the cookies file path to use.

    If the user passed an explicit ``--cookies`` path, use that.
    Otherwise fall back to stored cookies from ``transcriptex cookies``.
    """
    explicit = (explicit or "").strip()
    if explicit:
        return explicit
    from transcriptex.cookies import get_cookies_file

    return get_cookies_file()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    # Help, subtitle retrieval and searches must not initialize/install GPU runtimes.
    if (getattr(sys.stdout, 'encoding', None) or '').lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass

    argv = list(argv if argv is not None else sys.argv[1:])
    argv = _normalize_argv_for_backward_compat(argv)

    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or RUN_COMMAND

    handler = _DISPATCH.get(command, _run_export)
    json_mode = bool(getattr(args, 'json', False))
    args._report_stream = sys.stdout
    # Capture Python diagnostics from nested modules without buffering channel runs.
    # The final JSON is explicitly written to the original stdout.
    with redirect_stdout(sys.stderr) if json_mode else nullcontext():
        try:
            return handler(args)
        except KeyboardInterrupt as exc:
            print("\nInterrupted by user.", file=sys.stderr)
            return _emit_error_and_return(args, exc, command=command, exit_code=130)
        except Exception as exc:
            print(f"Fatal error: {exc}", file=sys.stderr)
            return _emit_error_and_return(args, exc, command=command)


_DISPATCH: dict[str, Callable[[argparse.Namespace], int]] = {}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _run_export(args: argparse.Namespace) -> int:
    tabs = normalize_list_arg(args.tabs) or list(DEFAULT_TABS)
    lang_priority = normalize_list_arg(args.lang_priority) or ["es", "en"]

    config = TranscriberConfig(
        channel_url=args.channel_url,
        output_dir=Path(args.output).resolve(),
        lang_priority=lang_priority,
        include_tabs=tabs,
        retries=max(0, args.retries),
        force=bool(args.force),
        verbose=bool(args.verbose),
        timeout=max(5, args.timeout),
        sleep_seconds=max(0.0, args.sleep),
        cookies=_resolve_cookies(str(getattr(args, "cookies", "") or "")),
    )

    transcriber = ChannelTranscriber(config)
    try:
        report = transcriber.run()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return _emit_error_and_return(args, exc, command=RUN_COMMAND)

    return _emit_report(
        args,
        command=RUN_COMMAND,
        report=report,
        human_formatter=lambda: _print_run_summary(report),
        exit_code=EXIT_WARNINGS if report.get("failed", 0) else EXIT_OK,
    )


def _run_audit(args: argparse.Namespace) -> int:
    config = AuditConfig(
        channel_dir=Path(args.channel_dir),
        write_files=not bool(args.no_write_files),
        max_rows_in_console=max(1, int(args.max_rows)),
        rename_legacy_files=bool(args.rename_legacy),
        print_summary=_should_print_summary(args),
    )
    try:
        report = audit_channel_output(config)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return _emit_error_and_return(args, exc, command=AUDIT_COMMAND)

    missing = int(report.get("indexed_without_transcript_count", 0))
    return _emit_report(
        args,
        command=AUDIT_COMMAND,
        report=report,
        human_formatter=None,  # audit prints its own summary
        exit_code=EXIT_WARNINGS if missing > 0 else EXIT_OK,
    )


def _run_search(args: argparse.Namespace) -> int:
    fields = normalize_list_arg(args.fields)
    statuses = normalize_list_arg(args.status)
    export_ids = normalize_list_arg(args.export_ids)
    export_indices = parse_comma_int_list(args.export_indices)
    export_dir = Path(args.export_dir).resolve() if args.export_dir else None

    config = SearchConfig(
        channel_dir=Path(args.channel_dir),
        query=args.query,
        fields=fields,
        statuses=statuses,
        top=max(1, int(args.top)),
        match_all_tokens=bool(args.match_all),
        regex=bool(args.regex),
        case_sensitive=bool(args.case_sensitive),
        snippet_len=max(60, int(args.snippet_len)),
        save_csv=bool(args.save_csv),
        print_results=_should_print_summary(args),
        export_dir=export_dir,
        export_top=max(0, int(args.export_top)),
        export_all=bool(args.export_all),
        export_ids=export_ids,
        export_indices=export_indices,
        zip_export=bool(args.zip),
        print_summary=_should_print_summary(args),
    )
    try:
        report = search_channel_output(config)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return _emit_error_and_return(args, exc, command=SEARCH_COMMAND)

    return _emit_report(
        args,
        command=SEARCH_COMMAND,
        report=report,
        human_formatter=None,  # search prints its own summary
        exit_code=EXIT_OK if int(report.get("total_matches", 0)) > 0 else EXIT_WARNINGS,
    )


def _run_asr(args: argparse.Namespace) -> int:
    from transcriptex.asr import AsrConfig, run_asr

    # Only attach cookies when truly needed. Sending account cookies for bulk
    # public downloads ties every request to your Google account, which makes
    # YouTube rate-limit (and potentially ban) far more aggressively. Private
    # videos (--include-failed) or an explicit --cookies path still use them.
    explicit_cookies = str(getattr(args, "cookies", "") or "").strip()
    if explicit_cookies or bool(args.include_failed):
        cookies = _resolve_cookies(explicit_cookies)
    else:
        cookies = ""

    config = AsrConfig(
        channel_dir=Path(args.channel_dir),
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=max(1, int(args.beam_size)),
        batch_size=max(1, int(args.batch_size)),
        force=bool(args.force),
        limit=max(0, int(args.limit)),
        sleep_seconds=max(0.0, float(args.sleep)),
        include_failed=bool(args.include_failed),
        verbose=bool(args.verbose),
        cookies=cookies,
        quiet=bool(getattr(args, "quiet", False)),
        print_summary=_should_print_summary(args),
        retries=max(1, int(getattr(args, "retries", 2))),
        rate_limit_wait_seconds=max(10.0, float(getattr(args, "rate_limit_wait", 60.0))),
        max_rate_limit_waits=max(1, int(getattr(args, "max_rate_limit_waits", 3))),
        yt_sleep_interval=0.0,
        download_workers=max(1, int(getattr(args, "download_workers", 4))),
    )
    try:
        report = run_asr(config)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return _emit_error_and_return(args, exc, command=ASR_COMMAND)

    interrupted = bool(report.get("interrupted"))
    has_errors = report.get("errors", 0) > 0
    exit_code = EXIT_WARNINGS if (has_errors or interrupted) else EXIT_OK

    return _emit_report(
        args,
        command=ASR_COMMAND,
        report=report,
        human_formatter=None,  # asr prints its own summary
        exit_code=exit_code,
    )


def _run_transcribe_single(args: argparse.Namespace) -> int:
    from transcriptex.transcribe_single import TranscribeSingleConfig, run_transcribe_single
    from transcriptex.asr_engine import ASREngine

    inputs = args.input_path if isinstance(args.input_path, list) else [args.input_path]
    inputs = list(dict.fromkeys(inputs))
    if args.batch_size < 0:
        raise ValueError('--batch-size must be nonnegative')
    engine = ASREngine(args.model, args.device, args.compute_type)
    cookies = _resolve_cookies(str(getattr(args, "cookies", "") or ""))
    results = []
    used_outputs = set()
    for input_path in inputs:
        is_url = input_path.startswith(('http://', 'https://'))
        output_dir = Path(args.output).resolve() if args.output else (Path.cwd() if is_url else Path(input_path).resolve().parent)
        stem = ''
        if not is_url:
            stem = Path(input_path).stem
            if (output_dir, stem) in used_outputs:
                stem += '-' + hashlib.sha256(str(Path(input_path).resolve()).encode()).hexdigest()[:8]
            used_outputs.add((output_dir, stem))
        config = TranscribeSingleConfig(
            input_path=input_path, output_dir=output_dir, output_format=args.format,
            model=args.model, device=args.device, compute_type=args.compute_type,
            language=args.language, beam_size=max(1, int(args.beam_size)),
            verbose=bool(args.verbose), cookies=cookies,
            quiet=bool(getattr(args, 'quiet', False)), print_summary=_should_print_summary(args),
            source=args.source, lang_priority=args.lang_priority, batch_size=args.batch_size,
            vad_filter=not args.no_vad, output_stem=stem,
        )
        try:
            report = run_transcribe_single(config, engine=engine)
        except Exception as exc:
            print(f"Failed: {input_path}: {exc}", file=sys.stderr)
            report = {'success': False, 'input': input_path, 'error': str(exc), 'error_type': type(exc).__name__}
        results.append(report)
    failures = sum(not r.get('success') for r in results)
    if len(results) == 1:
        report = results[0]
        exit_code = EXIT_FATAL if failures else EXIT_OK
    else:
        report = {'results': results, 'processed_ok': len(results)-failures, 'failed': failures}
        exit_code = EXIT_WARNINGS if failures else EXIT_OK
    return _emit_report(args, command=TRANSCRIBE_COMMAND, report=report,
        human_formatter=None, exit_code=exit_code)


def _run_realtime(args: argparse.Namespace) -> int:
    from transcriptex.realtime import RealtimeConfig, run_realtime

    config = RealtimeConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=max(1, int(args.beam_size)),
        chunk_duration=float(args.chunk_duration),
        process=args.process,
        list_devices=args.list_devices,
        device_index=args.device_index,
        source=args.source, microphone=args.microphone,
        monitor_stream=args.monitor_stream, output=args.output,
        max_segment_duration=args.max_segment_duration, silence_duration=args.silence_duration,
        silence_threshold=args.silence_threshold, overlap_duration=args.overlap_duration,
        preview=not args.no_preview, finalize=not args.no_finalize,
        record_audio=args.record_audio, queue_seconds=args.queue_seconds,
    )
    try:
        report = run_realtime(config)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return _emit_error_and_return(args, exc, command=REALTIME_COMMAND)

    return EXIT_OK if report.get("success") else EXIT_FATAL


def _run_scan(args: argparse.Namespace) -> int:
    config = ScanConfig(
        channel_url_or_dir=str(args.channel_url_or_dir),
        output_dir=Path(args.output).resolve(),
        cookies=_resolve_cookies(str(getattr(args, "cookies", "") or "")),
        no_interactive=bool(args.no_interactive),
        page_size=max(1, int(args.page_size)),
        print_summary=_should_print_summary(args),
    )
    try:
        report = scan_channel(config)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return _emit_error_and_return(args, exc, command=SCAN_COMMAND)

    return _emit_report(
        args,
        command=SCAN_COMMAND,
        report=report,
        human_formatter=None,  # scan prints its own summary
        exit_code=EXIT_WARNINGS if report.get("pending_count", 0) > 0 else EXIT_OK,
    )


def _run_cookies(args: argparse.Namespace) -> int:
    from transcriptex.cookies import (
        remove_stored_cookies,
        run_interactive_setup,
        show_cookie_status,
    )

    if getattr(args, "status", False):
        show_cookie_status()
        return EXIT_OK

    if getattr(args, "remove", False):
        if remove_stored_cookies():
            print("  Stored cookies removed.")
        else:
            print("  No stored cookies found.")
        return EXIT_OK

    try:
        run_interactive_setup()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return EXIT_FATAL

    return EXIT_OK


# Populate dispatch table at module import.
_DISPATCH.update({
    RUN_COMMAND: _run_export,
    AUDIT_COMMAND: _run_audit,
    SEARCH_COMMAND: _run_search,
    ASR_COMMAND: _run_asr,
    TRANSCRIBE_COMMAND: _run_transcribe_single,
    REALTIME_COMMAND: _run_realtime,
    SCAN_COMMAND: _run_scan,
    COOKIES_COMMAND: _run_cookies,
})


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _should_print_summary(args: argparse.Namespace) -> bool:
    """
    Return True if the human-readable summary should be printed.

    Suppressed when --quiet or --json is passed.
    """
    if bool(getattr(args, "quiet", False)):
        return False
    if bool(getattr(args, "json", False)):
        return False
    return True


def _emit_report(
    args: argparse.Namespace,
    command: str,
    report: dict[str, Any],
    human_formatter: Callable[[], None] | None,
    exit_code: int,
) -> int:
    """
    Print either the JSON report (--json) or the human summary, then return
    the exit code. Honors --quiet by skipping both.
    """
    if bool(getattr(args, "json", False)):
        # Wrap with command + meta for machine consumers
        payload = {
            "command": command,
            "success": exit_code == EXIT_OK,
            "exit_code": exit_code,
            "report": report,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), file=getattr(args, "_report_stream", sys.stdout))
        return exit_code

    if not _should_print_summary(args):
        return exit_code

    if human_formatter is not None:
        human_formatter()
    return exit_code


def _emit_error_and_return(
    args: argparse.Namespace,
    exc: BaseException,
    command: str,
    exit_code: int = EXIT_FATAL,
) -> int:
    """Emit a JSON error report if --json, otherwise return the fatal exit code."""
    if bool(getattr(args, "json", False)):
        payload = {
            "command": command,
            "success": False,
            "exit_code": exit_code,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False), file=getattr(args, "_report_stream", sys.stdout))
    return exit_code


def _print_run_summary(report: dict[str, Any]) -> None:
    print("")
    print("Run summary:")
    print(f"  Channel: {report.get('channel_title', '')}")
    print(f"  Videos indexed: {report.get('total_videos', 0)}")
    print(f"  OK: {report.get('processed_ok', 0)}")
    print(f"  Existing: {report.get('already_existing', 0)}")
    print(f"  No subtitles: {report.get('no_subtitles', 0)}")
    print(f"  Empty text: {report.get('no_text', 0)}")
    print(f"  Failed: {report.get('failed', 0)}")
    print(f"  Output: {report.get('output_dir', '')}")


# ---------------------------------------------------------------------------
# Backward compat
# ---------------------------------------------------------------------------

def _normalize_argv_for_backward_compat(argv: list[str]) -> list[str]:
    """
    If the first argument is not a known subcommand (and not a flag),
    assume the user meant `run`. This preserves the historical behaviour
    `transcriptex https://www.youtube.com/@foo` (treated as `run`).
    """
    if not argv:
        return ["-h"]
    first = argv[0]
    if first in KNOWN_SUBCOMMANDS:
        return argv
    if first.startswith("-"):
        return argv
    return [RUN_COMMAND, *argv]
