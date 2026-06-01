# AGENTS.md

## Setup

- Python 3.10+. Venv at `.venv/`. No `python` on PATH — use `.venv\Scripts\python.exe`.
- `ffmpeg` must be on PATH for `asr` and `transcribe` commands.
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

## Known Issue: Windows Unicode in Tests

7 ASR tests fail with `UnicodeEncodeError` on Windows. Root cause: `asr.py` prints Unicode chars (`\u25b6`, `\u2713`) but `cli.py:310-314` reconfigures stdout to UTF-8 only inside `main()`. Tests call `run_asr()` directly, bypassing that fix. To fix: either reconfigure stdout at module level in `asr.py`, or have tests set `PYTHONIOENCODING=utf-8`.

## Architecture

Single package `transcriptex/` with 6 CLI subcommands dispatched from `cli.py`:

| Command | Module | Purpose |
|---|---|---|
| `run` | `transcriber.py` | Channel subtitle export (no audio download) |
| `transcribe` | `transcribe_single.py` | Single file/URL transcription via faster-whisper |
| `asr` | `asr.py` + `asr_engine.py` | Batch Whisper fallback for videos without subtitles |
| `realtime` | `realtime.py` | Live WASAPI loopback transcription (Windows only) |
| `audit` | `audit.py` | Compare `video_index.csv` vs files on disk |
| `search` | `search.py` | Search across transcripts and metadata |

- `models.py`: shared dataclasses (`VideoRecord`, `PlaylistRecord`, `ChannelInfo`, `SubtitleTrack`, `TranscriptResult`).
- All heavy imports (`faster_whisper`, `torch`, `sounddevice`, `pyaudiowpatch`) are **lazy** — only loaded when their subcommand runs.
- `asr.py` saves `video_index.csv` incrementally after each video — crash-safe and resumable.
- `cli.py` has backward compat: if first arg is not a known subcommand, it assumes `run`.

## Conventions

- `os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"` is set in `asr.py` to avoid Windows symlink permission errors. Preserve this if refactoring.
- Transcript files follow the pattern `Title [VIDEOID].txt` with a metadata header block.
- `video_index.csv` in `_meta/` is the central index — all pipelines read/write it.
- The old package name was `yt_channel_transcriber` / `yt-channel`. Use `validate_refactor.py` to verify no references remain after renames.
