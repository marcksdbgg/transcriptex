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

KNOWN_SUBCOMMANDS = {
    RUN_COMMAND,
    AUDIT_COMMAND,
    SEARCH_COMMAND,
    ASR_COMMAND,
    TRANSCRIBE_COMMAND,
    REALTIME_COMMAND,
    SCAN_COMMAND,
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
    "videos. Empty = no cookies."
)

QUIET_HELP = (
    "Suppress the human-readable summary printed at the end. "
    "Errors and progress output are still shown unless combined with --json."
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
    "  transcriptex run \"https://www.youtube.com/@channel\" -o ./output\n"
    "  transcriptex scan \"output/My Channel\" --no-interactive\n"
    "  transcriptex search \"output/My Channel\" \"keyword\"\n"
    "  transcriptex asr \"output/My Channel\" --device cuda\n"
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
    "  transcriptex asr \"output/My Channel\"\n"
    "  transcriptex asr \"output/My Channel\" --model medium --device cuda --language es\n"
    "  transcriptex asr \"output/My Channel\" --include-failed --limit 50\n"
    "  transcriptex asr \"output/My Channel\" --cookies cookies.txt\n"
    "\n"
    "Exit codes:\n"
    "  0  every targeted video processed successfully\n"
    "  1  fatal error (no _meta/video_index.csv found)\n"
    "  2  one or more videos failed (see video_index.csv error column)"
)

EPILOG_TRANSCRIBE = (
    "Examples:\n"
    "  transcriptex transcribe video.mp4\n"
    "  transcriptex transcribe video.mp4 --format srt --language es\n"
    "  transcriptex transcribe video.mp4 --device cpu --model medium\n"
    "  transcriptex transcribe \"https://www.youtube.com/watch?v=abc\" --cookies cookies.txt\n"
    "\n"
    "Exit codes:\n"
    "  0  transcript written successfully\n"
    "  1  fatal error (download failed, transcription failed, etc.)"
)

EPILOG_REALTIME = (
    "Examples:\n"
    "  transcriptex realtime\n"
    "  transcriptex realtime --model medium --language es --chunk-duration 3.0\n"
    "  transcriptex realtime --list-devices\n"
    "  transcriptex realtime --process zen.exe --language en\n"
    "\n"
    "Notes:\n"
    "  - Windows-only (requires pyaudiowpatch and numpy).\n"
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
            "updated incrementally after each video (crash-safe and resumable)."
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
        default=5,
        help="Beam search size — higher = better quality, slower (default: 5).",
    )
    asr_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Audio chunk batch size (default: 8).",
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
        default=0.0,
        help="Sleep seconds between videos (default: 0).",
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
        help="Transcribe a single local file or YouTube URL with Whisper.",
        description=(
            "Transcribe any local audio/video file (MP4, MKV, MP3, WAV, etc.) or a "
            "YouTube URL using local faster-whisper. For local files, the file is "
            "read directly — no download. For URLs, audio is downloaded temporarily "
            "and deleted after transcription."
        ),
        epilog=EPILOG_TRANSCRIBE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    transcribe_parser.add_argument(
        "input_path",
        help="Local file path or YouTube URL (http(s)://...).",
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
        default=5,
        help="Beam search size (default: 5). Higher = better quality, slower.",
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
    _add_report_flags(transcribe_parser)

    # --- realtime ----------------------------------------------------------
    realtime_parser = subparsers.add_parser(
        REALTIME_COMMAND,
        help="Live transcription of system audio (Windows WASAPI loopback).",
        description=(
            "Capture audio playing on your PC (browsers, media players, meetings, "
            "etc.) via WASAPI Loopback and transcribe it live to the console. "
            "Windows-only — requires pyaudiowpatch and numpy."
        ),
        epilog=EPILOG_REALTIME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    realtime_parser.add_argument(
        "--model",
        default="medium",
        help="Whisper model name (default: medium — smaller = lower latency).",
    )
    realtime_parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device (default: cuda).",
    )
    realtime_parser.add_argument(
        "--compute-type",
        default="float16",
        help="Model quantization (default: float16).",
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
        help="Audio buffer window duration in seconds (default: 3.0).",
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
        help="List available WASAPI loopback audio devices and exit.",
    )
    realtime_parser.add_argument(
        "--device-index",
        type=int,
        default=-1,
        help="Select audio device by index (use --list-devices to see available devices).",
    )

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

    return parser


def _add_report_flags(parser: argparse.ArgumentParser) -> None:
    """Attach --quiet and --json flags to a reporting subcommand."""
    parser.add_argument("--quiet", action="store_true", help=QUIET_HELP)
    parser.add_argument("--json", action="store_true", help=JSON_HELP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    argv = list(argv if argv is not None else sys.argv[1:])
    argv = _normalize_argv_for_backward_compat(argv)

    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or RUN_COMMAND

    handler = _DISPATCH.get(command)
    if handler is None:
        return _run_export(args)
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130


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
        cookies=str(getattr(args, "cookies", "") or ""),
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
        cookies=str(getattr(args, "cookies", "") or ""),
        quiet=bool(getattr(args, "quiet", False)),
        print_summary=_should_print_summary(args),
    )
    try:
        report = run_asr(config)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return _emit_error_and_return(args, exc, command=ASR_COMMAND)

    return _emit_report(
        args,
        command=ASR_COMMAND,
        report=report,
        human_formatter=None,  # asr prints its own summary
        exit_code=EXIT_WARNINGS if report.get("errors", 0) > 0 else EXIT_OK,
    )


def _run_transcribe_single(args: argparse.Namespace) -> int:
    from transcriptex.transcribe_single import TranscribeSingleConfig, run_transcribe_single

    input_path = args.input_path
    is_url = input_path.startswith("http://") or input_path.startswith("https://")

    if args.output:
        output_dir = Path(args.output).resolve()
    elif not is_url:
        output_dir = Path(input_path).resolve().parent
    else:
        output_dir = Path(".").resolve()

    device = args.device
    compute_type = args.compute_type
    if device == "auto" or compute_type == "auto":
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False
        if device == "auto":
            device = "cuda" if gpu_available else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if (device == "cuda") else "int8"

    config = TranscribeSingleConfig(
        input_path=input_path,
        output_dir=output_dir,
        output_format=args.format,
        model=args.model,
        device=device,
        compute_type=compute_type,
        language=args.language,
        beam_size=max(1, int(args.beam_size)),
        verbose=bool(args.verbose),
        cookies=str(getattr(args, "cookies", "") or ""),
        quiet=bool(getattr(args, "quiet", False)),
        print_summary=_should_print_summary(args),
    )
    try:
        report = run_transcribe_single(config)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return _emit_error_and_return(args, exc, command=TRANSCRIBE_COMMAND)

    return _emit_report(
        args,
        command=TRANSCRIBE_COMMAND,
        report=report,
        human_formatter=None,  # transcribe prints its own summary
        exit_code=EXIT_OK if report.get("success") else EXIT_FATAL,
    )


def _run_realtime(args: argparse.Namespace) -> int:
    from transcriptex.realtime import RealtimeConfig, run_realtime

    config = RealtimeConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=max(1, int(args.beam_size)),
        chunk_duration=max(0.5, float(args.chunk_duration)),
        process=args.process,
        list_devices=args.list_devices,
        device_index=args.device_index,
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
        cookies=str(getattr(args, "cookies", "") or ""),
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


# Populate dispatch table at module import.
_DISPATCH.update({
    RUN_COMMAND: _run_export,
    AUDIT_COMMAND: _run_audit,
    SEARCH_COMMAND: _run_search,
    ASR_COMMAND: _run_asr,
    TRANSCRIBE_COMMAND: _run_transcribe_single,
    REALTIME_COMMAND: _run_realtime,
    SCAN_COMMAND: _run_scan,
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
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
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
) -> int:
    """Emit a JSON error report if --json, otherwise return the fatal exit code."""
    if bool(getattr(args, "json", False)):
        payload = {
            "command": command,
            "success": False,
            "exit_code": EXIT_FATAL,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return EXIT_FATAL


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
