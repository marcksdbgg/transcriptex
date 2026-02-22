# transcriptEX (CLI MVP)

CLI tool to export transcripts from all videos in a YouTube channel, organized by playlist folders, without using the official YouTube Data API.

## What this tool does

- Accepts a YouTube channel URL.
- Indexes videos from channel tabs (`videos`, `shorts`, `streams`).
- Indexes playlists and maps videos to playlists.
- Exports exactly one transcript file per `video_id` (no duplicates) from YouTube's auto/manual subtitles.
- Stores videos without playlists in `Sin_Playlist` and uses human-readable names: `Video title [VIDEO_ID].txt`.
- Includes an `asr` command to transcribe videos without CC/subtitles using local GPU transcription (`faster-whisper`).
- Includes a `search` command to find and export transcripts by keywords/phrases.
- Includes an `audit` command to analyze existing transcripts and find anomalies (missing, orphaned, etc).
- Writes detailed reports in the `_meta` folder (`playlists.csv`, `video_index.csv`, `run_report.json`, etc).

## Requirements

- Python 3.10+
- `yt-dlp` (installed via requirements)

Install dependencies:

```bash
pip install -r requirements.txt
```

For local ASR capabilities (optional but recommended), you also need to install `faster-whisper`. For GPU acceleration, PyTorch for CUDA is required:

```bash
pip install faster-whisper
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 1. Run Transcriber (YouTube Subtitles)

Run directly as a module:

```bash
python -m transcriptex "https://www.youtube.com/@example_channel"
```

Or install in editable mode and use the command:

```bash
pip install -e .
transcriptex "https://www.youtube.com/@example_channel"
```

### Useful options

```bash
transcriptex "https://www.youtube.com/@example_channel" \
  --output ./exports \
  --lang-priority es,en \
  --tabs videos,shorts,streams \
  --retries 3 \
  --timeout 45 \
  --sleep 0.2
```

- `--force`: regenerate transcripts even if the file already exists.
- `--verbose`: show yt-dlp verbose logs.

## 2. ASR Transcription (Local Whisper)

If a video has no manual/auto subtitle tracks on YouTube, it will be marked as `no_subtitles` in `video_index.csv`. You can transcribe these videos locally using `faster-whisper` and your GPU. 

The ASR process features:
- **Sequential reliable design**: No threading/tqdm UI freezing. Clean, line-by-line CLI progress.
- **Immediate crash safety**: Your CSV is saved incrementally after every video, allowing you to stop and resume natively.
- **Optimized Model**: Default model is `large-v3-turbo` which offers a great balance between accuracy and significantly faster speed compared to standard `large`.

```bash
transcriptex asr "output/My Channel"
```

### Useful ASR options

```bash
transcriptex asr "output/My Channel" \
  --model large-v3-turbo \
  --device cuda \
  --compute-type float16 \
  --language es \
  --beam-size 5
```

- `--force`: Force re-transcription of videos that already have a transcript file.
- `--limit N`: Process at most N videos in one run.
- `--include-failed`: Also attempt to transcribe videos marked as `failed` in the index, not just `no_subtitles`.
- `--verbose`: Show detailed debug logs.

## 3. Audit an existing output folder

Compare indexed videos vs transcript files and export audit artifacts (missing or orphan files):

```bash
transcriptex audit "output/My Channel"
```

To rename old legacy files like `VIDEOID.txt` to `Video title [VIDEOID].txt`:

```bash
transcriptex audit "output/My Channel" --rename-legacy
```

## 4. Search transcripts and export matches

Search by video title, playlist, video ID, or transcript content:

```bash
transcriptex search "output/My Channel" "fiesta del chivo"
```

Filter fields and require all words to match:

```bash
transcriptex search "output/My Channel" "trujillo urania" \
  --fields title,transcript \
  --match-all \
  --top 30
```

Export selected results:

```bash
# Export top 10 matches
transcriptex search "output/My Channel" "cervantes" --export-top 10

# Export specific results by shown index (1-based)
transcriptex search "output/My Channel" "cervantes" \
  --top 20 \
  --export-indices 1,3,7

# Export specific video IDs and create a structured zip archive
transcriptex search "output/My Channel" "quijote" \
  --export-ids abc123XYZ_9,def456UVW_8 \
  --zip
```

## Output Layout

```text
output/
  Channel_Name/
    Playlist_A [PLxxxx]/
      Video title [VIDEOID].txt
    Playlist_B [PLyyyy]/
      Video title [VIDEOID].txt
    Sin_Playlist/
      Video title [VIDEOID].txt
    _meta/
      playlists.csv
      video_index.csv
      run_report.json
      audit_report.json
      missing_transcripts.csv
      orphan_transcripts.csv
```

## Notes

- The primary `run` command does not download video/audio files, it only fetches subtitle tracks natively.
- The `asr` command *does* temporarily download the audio track to generate the transcript locally, immediately cleaning it up afterwards to preserve disk space.
- Restricted videos may fail depending on region, availability, or authentication requirements.
