# Implementation Plan: scan Command and Core Improvements

This plan details the addition of a new `scan` command to transcriptEX, along with the resolution of critical issues (thread pooling, absolute video mapping via the uploads playlist, cookie auth, and multi-playlist transcription copies).

## User Review Required

> [!IMPORTANT]
> The `scan` command provides an interactive CLI interface. If run in non-interactive environments (like CI/CD), the `--no-interactive` flag must be passed to prevent the terminal from blocking.

> [!NOTE]
> Writing transcripts to multiple playlist folders will duplicate the `.txt` files on disk for videos that reside in multiple playlists. This is chosen instead of symbolic links to ensure 100% compatibility across all Windows user account configurations (since symlinks often require developer mode or Administrator privileges).

---

## Proposed Changes

### 1. YouTube Client & Base Options

#### [MODIFY] [youtube_client.py](file:///g:/Mark/Codes/YT_transcriber/transcriptex/youtube_client.py)
* Add `cookies` (default `""`) to `YouTubeClient.__init__` and store it as `self.cookies`.
* Update `_base_opts` to include `"cookiefile": self.cookies` if `self.cookies` is set.
* Implement `extract_uploads_playlist_videos(self, channel_id: str)`:
  * Check that `channel_id` is a valid string starting with `UC`.
  * Construct the uploads playlist ID by replacing the second character `C` with `U` (e.g., `UCxxxx` -> `UUxxxx`).
  * Query `https://www.youtube.com/playlist?list=UUxxxx` using `extract_playlist_videos` and return the list of `VideoRecord`s.

### 2. Transcriber Core Optimizations

#### [MODIFY] [transcriber.py](file:///g:/Mark/Codes/YT_transcriber/transcriptex/transcriber.py)
* Update `TranscriberConfig` to include a `cookies` field (type `str`, default `""`).
* Pass `config.cookies` into `YouTubeClient`.
* In `_collect_channel_tab_videos`:
  * If the channel ID is available, try `self.client.extract_uploads_playlist_videos(channel_id)` first as the primary method to get all uploads.
  * If that fails or returns nothing, fallback to the current tab-based extraction (`videos`, `shorts`, `streams`).
* In `_collect_playlists_and_membership`:
  * Import `ThreadPoolExecutor` from `concurrent.futures`.
  * Parallelize fetching playlist videos for all playlists. Use a maximum of 4 workers to speed up indexing without hitting rate limits.
* In `_write_transcript_file` / `run`:
  * If a video is assigned to multiple playlists (meaning `playlist_membership[video_id]` has multiple entries), write the transcript file to **all** folders corresponding to those playlists, instead of only the first one.

### 3. Scan Command Implementation

#### [NEW] [scan.py](file:///g:/Mark/Codes/YT_transcriber/transcriptex/scan.py)
* Define a `ScanConfig` dataclass containing:
  * `channel_url_or_dir`: str
  * `output_dir`: Path
  * `cookies`: str
  * `no_interactive`: bool
* Implement `scan_channel(config: ScanConfig)`:
  * Determine if `channel_url_or_dir` is a URL or a directory.
  * If a URL:
    * Normalize and discover the channel.
    * Establish the channel output directory (e.g. `output/Channel_Name`).
    * Fetch all videos using `UU...` uploads playlist (or tab fallbacks) + playlists.
  * If a directory:
    * Search for `_meta/video_index.csv` inside it. If missing, fail with an error.
    * Parse the CSV to read all videos and playlists.
  * Scan the physical directory for existing transcripts (`*.txt` files matching `Title [VIDEOID].txt`).
  * Cross-reference the lists to count:
    * Total videos
    * Transcribed videos (already on disk)
    * Pending videos (indexed but missing transcript)
  * Write or update `_meta/channel_status.json` containing:
    * `channel_title`, `channel_id`, `channel_url`, `last_updated`, `total_videos`, `total_playlists`, `transcribed_count`, `pending_count`, `status_breakdown`, `source_breakdown`.
  * If `no_interactive` is `True`:
    * Print a clean summary table to stdout and exit.
  * If `no_interactive` is `False`:
    * Enter an interactive loop in the terminal:
      1. View summary statistics.
      2. Browse all videos (interactive pagination: e.g., 20 items per page, support `n`/`p`/`q` controls).
      3. List pending videos.
      4. List transcribed videos.
      5. Search videos by ID or text query (filters title, ID, playlist, and status).
      6. Exit.

### 4. CLI Subcommand Integration

#### [MODIFY] [cli.py](file:///g:/Mark/Codes/YT_transcriber/transcriptex/cli.py)
* Register the new `scan` subcommand in `build_parser()`:
  ```python
  scan_parser = subparsers.add_parser("scan", help="Scan a channel to map transcripts and view statuses")
  scan_parser.add_argument("channel_url_or_dir", help="YouTube channel URL or local path to channel folder")
  scan_parser.add_argument("-o", "--output", default="output", help="Output root directory")
  scan_parser.add_argument("--cookies", default="", help="Path to a cookies file")
  scan_parser.add_argument("--no-interactive", action="store_true", help="Print summary and exit without interactive menu")
  ```
* Register the `scan` subcommand in `KNOWN_SUBCOMMANDS`.
* Add the `--cookies` argument to `run` and `asr` subcommands.
* Add `_run_scan(args)` handler mapping command arguments to `ScanConfig` and invoking `scan_channel(config)`.

---

## Verification Plan

### Automated Tests
* Add tests in a new file `tests/test_scan.py` to:
  * Verify `scan_channel` works with a mock directory and mock CSV.
  * Verify `channel_status.json` is generated correctly.
  * Verify `UU...` playlist construction and fallbacks.
* Run the tests:
  ```bash
  .venv\Scripts\python.exe -m unittest discover -s tests -v
  ```

### Manual Verification
* Test the `scan` command against a small test channel and verify:
  * The interactive menu is functional.
  * Pagination works.
  * Search behaves correctly.
  * Running the command with `--no-interactive` behaves correctly.
* Run a scan for `"https://www.youtube.com/@JesúsGMaestro"` and verify:
  * Total counts match (~1.9k videos).
  * Check the generated `_meta/channel_status.json` file.
