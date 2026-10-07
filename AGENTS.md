# AGENTS.md

## Setup

- Python 3.10+. Venv at `.venv/`. Use `.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on Linux.
- `ffmpeg` must be on PATH for the channel `asr` pipeline; single-file ASR uses PyAV.
- Install: `pip install -e .` (editable) or `pip install -r requirements.txt`.

## Commands

```bash
# Run tests (unittest, not pytest)
.venv\Scripts\python.exe -m unittest discover -s tests -v

# Run a single test module
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_url_utils.py" -v

# Run the CLI
.venv\Scripts\python.exe -m transcriptex <command> [args]

# Validate no old name references remain
.venv\Scripts\python.exe validate_refactor.py
```

No linter, formatter, typechecker, or CI is configured.

## Windows console encoding

`asr.py` configures stdout/stderr as UTF-8 when supported, including direct
`run_asr()` calls. Preserve this protection for Unicode progress messages.
The latest regression suite was run on Linux; Windows capture still needs
validation on a Windows machine.

## Auto-Installation of Heavy Dependencies

`transcriptex/dependency_manager.py` provides `ensure_*` helpers that install missing
packages **into the current interpreter** (usually `.venv/`) via `subprocess` + `pip`.

- `faster-whisper` is installed on-demand for ASR. PyTorch is not required by the CTranslate2 inference path.
- GPU detection uses CTranslate2 and VRAM sizing uses `nvidia-smi`, without requiring or installing PyTorch.
- `pyaudiowpatch` and `numpy` are installed on-demand for `realtime`.
- `yt-dlp` and `yt-dlp-ejs` are core dependencies; `dependency_manager` guards them too.
- YouTube downloads (ASR audio, `run` metadata) require **Node.js, Deno, or Bun on PATH** for yt-dlp's EJS challenge solver (since yt-dlp 2025.11). Without it, downloads fail with *"Requested format is not available"*.

This means the CLI no longer crashes with *"faster-whisper is not installed"* or
*"torch not installed"*; it prompts for installation and continues.

## Architecture

Single package `transcriptex/` with 8 CLI subcommands dispatched from `cli.py`:

| Command | Module | Purpose |
|---|---|---|
| `run` | `transcriber.py` | Channel subtitle export (no audio download) |
| `transcribe` | `transcribe_single.py` + `subtitle_segments.py` | One/multiple files or URLs; subtitles first, batched ASR fallback, shared model |
| `asr` | `asr.py` + `asr_engine.py` | Batch Whisper fallback for videos without subtitles |
| `realtime` | `realtime.py` + `realtime_linux.py` + `live_decoder.py` | Windows WASAPI / Linux PipeWire capture; bounded pause-aware decoding and final full-audio pass |
| `audit` | `audit.py` | Compare `video_index.csv` vs files on disk |
| `search` | `search.py` | Search across transcripts and metadata |
| `scan` | `scan.py` | Coverage discovery and index reconciliation |
| `cookies` | `cookies.py` | Explicit interactive cookie setup/status |

- `models.py`: shared dataclasses (`VideoRecord`, `PlaylistRecord`, `ChannelInfo`, `SubtitleTrack`, `TranscriptResult`).
- Heavy inference and capture imports (`faster_whisper`, `numpy`, `pyaudiowpatch`) are **lazy** — only loaded when needed. PyTorch and sounddevice are not required.
- `asr.py` saves `video_index.csv` incrementally after each video — crash-safe and resumable.
- `cli.py` has backward compat: if first arg is not a known subcommand, it assumes `run`.
- `dependency_manager.py`: auto-installs missing heavy deps (`faster-whisper`, `pyaudiowpatch`, `numpy`, `yt-dlp`) into the running interpreter on first use.

## Conventions

- `os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"` is set in `asr.py` to avoid Windows symlink permission errors. Preserve this if refactoring.
- Transcript files follow the pattern `Title [VIDEOID].txt` with a metadata header block.
- `video_index.csv` in `_meta/` is the central index — all pipelines read/write it.
- The old package name was `yt_channel_transcriber` / `yt-channel`. Use `validate_refactor.py` to verify no references remain after renames.

## Automation and live invariants

- Reporting commands keep stdout as one JSON document with `--json`; diagnostics
  go to stderr. Preserve this for nested modules and error paths.
- `transcribe --format json` writes segments, while `--json` prints a report.
  Multi-input calls reuse one lazy ASREngine and return `report.results`.
- URL `--source auto` prefers captions; `subtitles` must never invoke ASR.
- ASREngine file batching defaults to 8 on CUDA / sequential on CPU, and reduces
  batch size on OOM before any export is written. CPU and no-VAD paths stay sequential.
- Linux previews are provisional; only confirmed events enter the live TXT.
  Forced cuts retain overlap and filter previously committed word timings.
- SIGINT must stop capture subprocesses, drain queued frames and flush speech.
  Finalization uses temporary per-source audio; only `--record-audio` retains WAVs.
- Keep microphone and application sources isolated. Never recover a vanished
  `--monitor-stream` by switching to all system playback. Source labels are not diarization.
- Relevant regression tests: `test_transcribe_automation.py`, `test_asr_engine.py`,
  `test_live_decoder.py`, `test_realtime_linux.py`, `test_capture_sources.py`; hardware/network benchmarks are separate.

- Validate audio source names before opening parec; PipeWire can otherwise route an unknown target to the default microphone. Disable node fallback/reconnect and verify the actual capture route. Isolated PipeWire sink-input capture uses `target.object`, not a regular source index.
