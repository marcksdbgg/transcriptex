from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcriptex.audit import AuditConfig, audit_channel_output
from transcriptex.fs_utils import normalize_list_arg
from transcriptex.search import SearchConfig, parse_comma_int_list, search_channel_output
from transcriptex.transcriber import DEFAULT_TABS, ChannelTranscriber, TranscriberConfig

RUN_COMMAND = "run"
AUDIT_COMMAND = "audit"
SEARCH_COMMAND = "search"
ASR_COMMAND = "asr"
TRANSCRIBE_COMMAND = "transcribe"
REALTIME_COMMAND = "realtime"
KNOWN_SUBCOMMANDS = {RUN_COMMAND, AUDIT_COMMAND, SEARCH_COMMAND, ASR_COMMAND, TRANSCRIBE_COMMAND, REALTIME_COMMAND}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriptex",
        description="YouTube channel transcript CLI (run + audit + search).",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        RUN_COMMAND,
        help="Run transcript export from a YouTube channel URL",
        description=(
            "Export transcripts from all videos in a YouTube channel, organized by playlists, "
            "without using the official YouTube API."
        ),
    )
    run_parser.add_argument("channel_url", help="YouTube channel URL (example: https://www.youtube.com/@channel)")
    run_parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Output root directory (default: ./output)",
    )
    run_parser.add_argument(
        "--lang-priority",
        default="es,en",
        help="Comma-separated subtitle language preference (default: es,en)",
    )
    run_parser.add_argument(
        "--tabs",
        default=",".join(DEFAULT_TABS),
        help=f"Comma-separated channel tabs to index (default: {','.join(DEFAULT_TABS)})",
    )
    run_parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry attempts for metadata/subtitle retrieval (default: 2)",
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Network timeout in seconds (default: 30)",
    )
    run_parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep seconds between videos (default: 0)",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate transcript files even if they already exist",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable yt-dlp verbose output",
    )

    audit_parser = subparsers.add_parser(
        AUDIT_COMMAND,
        help="Audit a channel output folder and compare indexed videos vs transcripts",
    )
    audit_parser.add_argument(
        "channel_dir",
        help="Path to channel output folder (example: output/My Channel)",
    )
    audit_parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="Max missing rows to print in console summary (default: 10)",
    )
    audit_parser.add_argument(
        "--no-write-files",
        action="store_true",
        help="Do not write audit files into _meta",
    )
    audit_parser.add_argument(
        "--rename-legacy",
        action="store_true",
        help="Rename legacy transcript files named only by VIDEO_ID into title [VIDEO_ID].txt",
    )

    search_parser = subparsers.add_parser(
        SEARCH_COMMAND,
        help="Search videos/transcripts in an existing channel output folder",
    )
    search_parser.add_argument(
        "channel_dir",
        help="Path to channel output folder (example: output/My Channel)",
    )
    search_parser.add_argument(
        "query",
        help="Search query (keyword, phrase, or video ID)",
    )
    search_parser.add_argument(
        "--fields",
        default="title,playlist,video_id,transcript",
        help="Comma-separated fields: title,playlist,video_id,transcript",
    )
    search_parser.add_argument(
        "--status",
        default="",
        help="Comma-separated status filter (ok,existing,no_subtitles,no_text,failed)",
    )
    search_parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Max results shown (default: 25)",
    )
    search_parser.add_argument(
        "--match-all",
        action="store_true",
        help="Require all query tokens to match",
    )
    search_parser.add_argument(
        "--regex",
        action="store_true",
        help="Treat query as regex",
    )
    search_parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Case-sensitive search",
    )
    search_parser.add_argument(
        "--snippet-len",
        type=int,
        default=140,
        help="Snippet length for transcript matches (default: 140)",
    )
    search_parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Write CSV of top results into _meta",
    )
    search_parser.add_argument(
        "--export-dir",
        default="",
        help="Directory for exporting matched transcript files",
    )
    search_parser.add_argument(
        "--export-top",
        type=int,
        default=0,
        help="Export top N results",
    )
    search_parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all shown results",
    )
    search_parser.add_argument(
        "--export-ids",
        default="",
        help="Export by video IDs (comma-separated)",
    )
    search_parser.add_argument(
        "--export-indices",
        default="",
        help="Export by result indices shown in console (comma-separated, 1-based)",
    )
    search_parser.add_argument(
        "--zip",
        action="store_true",
        help="Create ZIP for exported files",
    )
    asr_parser = subparsers.add_parser(
        ASR_COMMAND,
        help="Transcribe videos without subtitles using local ASR (faster-whisper + GPU)",
        description=(
            "Download audio for videos with 'no_subtitles' status and transcribe them locally "
            "using faster-whisper. Requires: pip install faster-whisper. "
            "GPU strongly recommended (RTX 4060 or better)."
        ),
    )
    asr_parser.add_argument(
        "channel_dir",
        help="Path to channel output folder (example: output/My Channel)",
    )
    asr_parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Whisper model name (default: large-v3-turbo). Options: tiny, base, small, medium, large-v2, large-v3, large-v3-turbo",
    )
    asr_parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device (default: cuda). Falls back to cpu if CUDA unavailable.",
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
        help="Beam search size — higher = better quality, slower (default: 5)",
    )
    asr_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Audio chunk batch size (default: 8)",
    )
    asr_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even if a transcript file already exists",
    )
    asr_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of videos to process (default: 0 = no limit)",
    )
    asr_parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between videos (default: 0)",
    )
    asr_parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also retry videos with status 'failed' (e.g. metadata errors)",
    )
    asr_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose yt-dlp output during audio download",
    )

    transcribe_parser = subparsers.add_parser(
        TRANSCRIBE_COMMAND,
        help="Transcribe a single local audio/video file or YouTube URL",
        description="Download audio if URL, then transcribe using local faster-whisper.",
    )
    transcribe_parser.add_argument("input_path", help="Local file path or YouTube URL")
    transcribe_parser.add_argument("-o", "--output", default="", help="Output directory (default: same folder as input file; current dir for YouTube URLs)")
    transcribe_parser.add_argument(
        "--format",
        default="txt",
        choices=["txt", "srt", "vtt", "json"],
        help=(
            "Output format (default: txt).\n"
            "  txt  — clean plain text, words joined, paragraph breaks on long silences (DEFAULT)\n"
            "  srt  — numbered subtitles with timestamps (00:00:00,000 --> 00:00:00,000)\n"
            "  vtt  — WebVTT subtitles (compatible with HTML5 <track>, .vtt files)\n"
            "  json — full segment data as JSON array [{start, end, text}, ...]"
        ),
    )
    transcribe_parser.add_argument("--model", default="large-v3-turbo", help="Whisper model: tiny, base, small, medium, large-v2, large-v3, large-v3-turbo (default: large-v3-turbo)")
    transcribe_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device: auto, cuda, cpu (default: auto — uses GPU if available, else CPU)")
    transcribe_parser.add_argument("--compute-type", default="auto", help="Quantization: auto, float16, int8_float16, int8, float32 (default: auto — float16 for GPU, int8 for CPU)")
    transcribe_parser.add_argument("--language", default="", help="Language code, e.g. 'es' or 'en' (default: auto-detect)")
    transcribe_parser.add_argument("--beam-size", type=int, default=5, help="Beam search size, higher = better quality but slower (default: 5)")
    transcribe_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    realtime_parser = subparsers.add_parser(
        REALTIME_COMMAND,
        help="Live realtime transcription of system audio (Loopback WASAPI)",
        description="Captures live audio playing on your PC (e.g., from a browser) and transcribes it to the console.",
    )
    realtime_parser.add_argument("--model", default="medium", help="Whisper model name (default: medium, for speed)")
    realtime_parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device (default: cuda)")
    realtime_parser.add_argument("--compute-type", default="float16", help="Model quantization (default: float16)")
    realtime_parser.add_argument("--language", default="", help="Language code (default: auto)")
    realtime_parser.add_argument("--beam-size", type=int, default=1, help="Beam search size (default: 1 for latency)")
    realtime_parser.add_argument("--chunk-duration", type=float, default=3.0, help="Audio buffer block duration in seconds (default: 3.0)")

    return parser


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

    if command == AUDIT_COMMAND:
        return _run_audit(args)
    if command == SEARCH_COMMAND:
        return _run_search(args)
    if command == ASR_COMMAND:
        return _run_asr(args)
    if command == TRANSCRIBE_COMMAND:
        return _run_transcribe_single(args)
    if command == REALTIME_COMMAND:
        return _run_realtime(args)
    return _run_export(args)


def _run_export(args: argparse.Namespace) -> int:
    tabs = normalize_list_arg(args.tabs)
    if not tabs:
        tabs = list(DEFAULT_TABS)

    lang_priority = normalize_list_arg(args.lang_priority)
    if not lang_priority:
        lang_priority = ["es", "en"]

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
    )

    transcriber = ChannelTranscriber(config)
    try:
        report = transcriber.run()
    except Exception as exc:
        print(f"Fatal error: {exc}")
        return 1

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

    return 2 if report.get("failed", 0) else 0


def _run_audit(args: argparse.Namespace) -> int:
    config = AuditConfig(
        channel_dir=Path(args.channel_dir),
        write_files=not bool(args.no_write_files),
        max_rows_in_console=max(1, int(args.max_rows)),
        rename_legacy_files=bool(args.rename_legacy),
    )
    try:
        report = audit_channel_output(config)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        return 1

    missing = int(report.get("indexed_without_transcript_count", 0))
    return 2 if missing > 0 else 0


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
        export_dir=export_dir,
        export_top=max(0, int(args.export_top)),
        export_all=bool(args.export_all),
        export_ids=export_ids,
        export_indices=export_indices,
        zip_export=bool(args.zip),
    )
    try:
        report = search_channel_output(config)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        return 1

    return 0 if int(report.get("total_matches", 0)) > 0 else 2


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
    )
    try:
        report = run_asr(config)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        return 1

    return 2 if report.get("errors", 0) > 0 else 0


def _run_transcribe_single(args: argparse.Namespace) -> int:
    from transcriptex.transcribe_single import TranscribeSingleConfig, run_transcribe_single

    input_path = args.input_path
    is_url = input_path.startswith("http://") or input_path.startswith("https://")

    # Resolve output directory:
    # - If explicitly provided: use it.
    # - If local file and no output given: save next to the input file.
    # - If URL and no output given: use current directory.
    if args.output:
        output_dir = Path(args.output).resolve()
    elif not is_url:
        output_dir = Path(input_path).resolve().parent
    else:
        output_dir = Path(".").resolve()

    # Resolve device/compute-type: "auto" → detect GPU, fallback CPU
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
    )
    try:
        report = run_transcribe_single(config)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        return 1

    return 0 if report.get("success") else 1


def _run_realtime(args: argparse.Namespace) -> int:
    from transcriptex.realtime import RealtimeConfig, run_realtime

    config = RealtimeConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        beam_size=max(1, int(args.beam_size)),
        chunk_duration=max(0.5, float(args.chunk_duration)),
    )
    try:
        report = run_realtime(config)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        return 1

    return 0 if report.get("success") else 1


def _normalize_argv_for_backward_compat(argv: list[str]) -> list[str]:
    if not argv:
        return ["-h"]
    first = argv[0]
    if first in KNOWN_SUBCOMMANDS:
        return argv
    if first in {"-h", "--help"}:
        return argv
    return [RUN_COMMAND, *argv]
