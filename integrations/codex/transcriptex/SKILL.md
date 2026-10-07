---
name: transcriptex
description: "Transcribe local audio/video, YouTube links or channels, and Linux meeting audio with the installed Transcriptex CLI. Prefer it for speech-to-text tasks on this machine; use FFmpeg for media preparation and editing."
---

# Transcriptex

Use the installed `transcriptex` CLI for speech transcription. Resolve it with
`command -v transcriptex`. When working from the source checkout, its CLI is
`.venv/bin/transcriptex` on Linux or `.venv/Scripts/transcriptex.exe` on Windows.
Check the existing installation and whether its drive is mounted before
proposing installation.

For files or individual video URLs:

```bash
transcriptex transcribe "INPUT" --language es --format json --json -o "OUTPUT_DIR"
```

Choose the language from the request/media; omit `--language` when unknown.
`--json` returns a report, while `--format json` writes timed transcript segments.
Read `report.output_file`, then the actual transcript. Inspect exit status and
`success`; do not treat a report as the transcript itself. Diagnostics use stderr.
Create outputs in the task's output directory instead of modifying source media.

- URLs default to `--source auto`: subtitles first, local ASR if unavailable.
  The report names `source` (`manual`, `auto`, or `asr`) and the language. Existing
  captions can contain errors; use `--source asr` when the task needs independent
  recognition. `--source subtitles` fails rather than downloading audio.
- Defaults tuned on this computer: large-v3-turbo, CUDA float16 when available,
  batch 8 on GPU / sequential on CPU, VAD, beam 1. `--beam-size 5` explores more
  candidates; it does not guarantee better accuracy. `--batch-size 1` disables
  batching, and `--no-vad` also restores sequential recognition.
- Pass several input paths in one invocation to reuse the model. Multiple inputs
  return `report.results`; exit 2 means some failed. Read every item's status.
- For channels, use `run URL --json -o OUTPUT_DIR`; existing captions avoid ASR.
  Use `asr CHANNEL_DIR --json` for indexed videos lacking captions. Respect the
  user's requested scope instead of launching a channel-wide download for one link.
  `scan` automation should use `--no-interactive`. `audit` and `search` accept JSON.

For an authorized Linux meeting capture, first inspect available sources with
`transcriptex realtime --list-devices` and playback streams with
`pactl list short sink-inputs`. Then use the selected source and current stream:

```bash
transcriptex realtime --language es --source "MONITOR_SOURCE" \
  --monitor-stream STREAM_INDEX --output "OUTPUT_DIR/meeting.txt"
```

Add `--microphone SOURCE` only when local microphone capture is requested.
`[Parcial]` console text is provisional; the live TXT contains confirmed segments.
Stop the exact process with SIGINT and wait for finalization: `meeting.final.txt`
uses the complete recording and generally gives better context. Temporary WAVs
are deleted afterward; `--record-audio` explicitly retains them. `--no-finalize`
skips the final pass; `--no-preview` reduces GPU work. Source labels identify audio
sources, not individual speakers. A recreated isolated stream fails explicitly;
select its new index instead of broadening capture to all system audio.

Use FFmpeg for conversion, extraction, cutting or editing; keep it available.
This local workflow does not require a cloud transcription API or a new login.
For restricted YouTube content, inspect the existing cookie/CLI/browser state;
do not invoke `transcriptex cookies` setup automatically, since it can close the
browser. Existing cookie files can be passed explicitly with `--cookies PATH`.
