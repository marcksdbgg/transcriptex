"""
scan.py — Channel scan command.

Cross-references the YouTube channel inventory (URL or local folder) with
transcript files on disk, and writes/updates `_meta/channel_status.json`
with the resulting snapshot.

Modes:
- URL mode:    discover the channel, fetch every video via the uploads
               playlist (UUxxxx...) and per-playlist endpoints, then scan
               the resolved output folder for transcripts.
- Folder mode: re-use the existing `_meta/video_index.csv` of a previous
               `run` to compute the status snapshot without any network
               access.

If `no_interactive` is False, a terminal menu is presented for browsing,
filtering and searching. If True, a clean summary table is printed and
the function returns immediately.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transcriptex.fs_utils import (
    extract_video_id_from_stem,
    now_utc_iso,
    sanitize_name,
)
from transcriptex.models import ChannelInfo, PlaylistRecord, VideoRecord
from transcriptex.url_utils import normalize_channel_url
from transcriptex.youtube_client import YouTubeClient

META_FOLDER = "_meta"
CHANNEL_STATUS_FILE = "channel_status.json"
VIDEO_INDEX_FILE = "video_index.csv"
PLAYLISTS_FILE = "playlists.csv"
PAGE_SIZE = 20

# Statuses we treat as "transcribed" when cross-referencing.
_TRANSCRIBED_STATUSES = {"ok", "existing"}


@dataclass
class ScanConfig:
    channel_url_or_dir: str
    output_dir: Path = Path("output")
    cookies: str = ""
    no_interactive: bool = False
    page_size: int = PAGE_SIZE
    print_summary: bool = True


@dataclass
class ScanEntry:
    video_id: str
    title: str
    url: str
    status: str
    language: str
    source: str
    assigned_playlist_id: str
    assigned_playlist_title: str
    all_playlist_ids: list[str] = field(default_factory=list)
    all_playlist_titles: list[str] = field(default_factory=list)
    has_transcript: bool = False
    transcript_path: str = ""


@dataclass
class ScanResult:
    channel_title: str
    channel_id: str
    channel_url: str
    channel_dir: Path
    total_videos: int
    total_playlists: int
    transcribed_count: int
    pending_count: int
    status_breakdown: dict[str, int]
    source_breakdown: dict[str, int]
    playlist_breakdown: dict[str, int]
    entries: list[ScanEntry]
    generated_at: str


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def scan_channel(config: ScanConfig) -> dict[str, Any]:
    """
    Run the scan command. Returns a JSON-serializable summary dict, and
    writes `_meta/channel_status.json` inside the resolved channel dir.
    """
    raw = (config.channel_url_or_dir or "").strip()
    if not raw:
        raise ValueError("channel_url_or_dir is required")

    if _looks_like_url(raw):
        result = _scan_from_url(config=config, raw=raw)
    else:
        result = _scan_from_dir(config=config, raw=raw)

    _write_channel_status(result)

    if config.no_interactive:
        if config.print_summary:
            _print_non_interactive_summary(result)
    else:
        _run_interactive_loop(result, page_size=max(1, int(config.page_size)))

    return _result_to_dict(result)


# ---------------------------------------------------------------------------
# URL mode
# ---------------------------------------------------------------------------

def _scan_from_url(config: ScanConfig, raw: str) -> ScanResult:
    channel_base_url = normalize_channel_url(raw)
    client = YouTubeClient(cookies=config.cookies)

    channel_info = client.discover_channel(channel_base_url)
    channel_folder_name = sanitize_name(
        channel_info.title or channel_info.handle or "Channel",
        max_len=120,
        fallback="Channel",
    )
    channel_dir = (config.output_dir / channel_folder_name).resolve()

    uploads_records: list[VideoRecord] = []
    if channel_info.channel_id:
        uploads_records = client.extract_uploads_playlist_videos(
            channel_id=channel_info.channel_id,
            fallback_channel=channel_info.title,
        )

    videos_by_id: dict[str, VideoRecord] = {}
    for record in uploads_records:
        _upsert_video(videos_by_id, record)

    playlist_records: list[PlaylistRecord] = []
    try:
        playlists = client.extract_playlists(channel_base_url)
    except Exception as exc:
        print(f"Warning: failed to list playlists: {exc}")
        playlists = []

    for playlist in playlists:
        try:
            records = client.extract_playlist_videos(
                playlist_url=playlist.url,
                fallback_channel=channel_info.title,
            )
        except Exception as exc:
            print(f"Warning: failed to read playlist '{playlist.title}': {exc}")
            records = []
        playlist.videos = records
        playlist_records.append(playlist)
        for record in records:
            _upsert_video(videos_by_id, record)

    transcript_map, id_to_files = _scan_transcript_files(channel_dir)

    # Persist discovered videos so that downstream commands (run/asr) can
    # use them without re-indexing. We deliberately do NOT transcribe or
    # download any media — this is a scan, not a run.
    if videos_by_id:
        _persist_scan_index(
            channel_dir=channel_dir,
            channel_info=channel_info,
            channel_base_url=channel_base_url,
            videos_by_id=videos_by_id,
            playlists=playlist_records,
        )

    entries = _build_entries_from_videos(
        videos_by_id=videos_by_id,
        playlists=playlist_records,
        transcript_map=transcript_map,
    )

    return _assemble_result(
        channel_info=channel_info,
        channel_base_url=channel_base_url,
        channel_dir=channel_dir,
        entries=entries,
        total_playlists=len(playlist_records),
    )


# ---------------------------------------------------------------------------
# Folder mode
# ---------------------------------------------------------------------------

def _scan_from_dir(config: ScanConfig, raw: str) -> ScanResult:
    channel_dir = Path(raw).expanduser().resolve()
    if not channel_dir.exists():
        raise FileNotFoundError(f"Channel folder not found: {channel_dir}")
    if not channel_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {channel_dir}")

    meta_dir = channel_dir / META_FOLDER
    video_index_path = meta_dir / VIDEO_INDEX_FILE
    if not video_index_path.exists():
        raise FileNotFoundError(
            f"Missing file: {video_index_path}. Run 'transcriptex run <url>' first."
        )

    rows = _read_video_index(video_index_path)
    transcript_map, _ = _scan_transcript_files(channel_dir)
    playlists = _read_playlists_csv(meta_dir / PLAYLISTS_FILE)

    entries = _build_entries_from_rows(rows=rows, transcript_map=transcript_map)

    channel_title = ""
    channel_id = ""
    channel_url = ""
    if rows:
        first = rows[0]
        channel_title = ""
        channel_id = ""
        channel_url = first.get("url", "").rsplit("/", 1)[0] if first.get("url") else ""

    # Try to recover the channel title from playlists.csv (no header for it
    # in the current schema), so fall back to folder name.
    if not channel_title:
        channel_title = channel_dir.name

    return _assemble_result(
        channel_info=ChannelInfo(
            base_url=channel_url,
            title=channel_title,
            channel_id=channel_id,
        ),
        channel_base_url=channel_url,
        channel_dir=channel_dir,
        entries=entries,
        total_playlists=len(playlists),
    )


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------

def _assemble_result(
    channel_info: ChannelInfo,
    channel_base_url: str,
    channel_dir: Path,
    entries: list[ScanEntry],
    total_playlists: int,
) -> ScanResult:
    transcribed = sum(1 for entry in entries if entry.has_transcript)
    pending = len(entries) - transcribed

    status_breakdown: Counter[str] = Counter()
    source_breakdown: Counter[str] = Counter()
    playlist_breakdown: Counter[str] = Counter()
    for entry in entries:
        status_breakdown[entry.status or "unknown"] += 1
        if entry.source:
            source_breakdown[entry.source] += 1
        else:
            source_breakdown["(none)"] += 1
        bucket = entry.assigned_playlist_title or "(ungrouped)"
        playlist_breakdown[bucket] += 1

    return ScanResult(
        channel_title=channel_info.title or channel_dir.name,
        channel_id=channel_info.channel_id or "",
        channel_url=channel_base_url or channel_info.base_url or "",
        channel_dir=channel_dir,
        total_videos=len(entries),
        total_playlists=total_playlists,
        transcribed_count=transcribed,
        pending_count=pending,
        status_breakdown=dict(status_breakdown),
        source_breakdown=dict(source_breakdown),
        playlist_breakdown=dict(playlist_breakdown),
        entries=entries,
        generated_at=now_utc_iso(),
    )


def _result_to_dict(result: ScanResult) -> dict[str, Any]:
    return {
        "channel_title": result.channel_title,
        "channel_id": result.channel_id,
        "channel_url": result.channel_url,
        "channel_dir": str(result.channel_dir),
        "last_updated": result.generated_at,
        "total_videos": result.total_videos,
        "total_playlists": result.total_playlists,
        "transcribed_count": result.transcribed_count,
        "pending_count": result.pending_count,
        "status_breakdown": result.status_breakdown,
        "source_breakdown": result.source_breakdown,
        "playlist_breakdown": result.playlist_breakdown,
    }


# ---------------------------------------------------------------------------
# Build ScanEntry lists
# ---------------------------------------------------------------------------

def _build_entries_from_videos(
    videos_by_id: dict[str, VideoRecord],
    playlists: list[PlaylistRecord],
    transcript_map: dict[str, Path],
) -> list[ScanEntry]:
    playlist_by_id = {pl.playlist_id: pl for pl in playlists}
    membership: dict[str, list[str]] = defaultdict(list)
    for playlist in playlists:
        for video in playlist.videos:
            if playlist.playlist_id not in membership[video.video_id]:
                membership[video.video_id].append(playlist.playlist_id)

    entries: list[ScanEntry] = []
    for video_id, video in videos_by_id.items():
        ids = membership.get(video_id, [])
        assigned_id = ids[0] if ids else ""
        assigned_playlist = playlist_by_id.get(assigned_id)
        assigned_title = assigned_playlist.title if assigned_playlist else ""
        titles = [
            playlist_by_id[pid].title
            for pid in ids
            if pid in playlist_by_id
        ]
        transcript_path = transcript_map.get(video_id)
        entries.append(
            ScanEntry(
                video_id=video_id,
                title=video.title or "",
                url=video.url or "",
                status="indexed",
                language="",
                source="",
                assigned_playlist_id=assigned_id,
                assigned_playlist_title=assigned_title,
                all_playlist_ids=ids,
                all_playlist_titles=titles,
                has_transcript=transcript_path is not None,
                transcript_path=str(transcript_path) if transcript_path else "",
            )
        )

    entries.sort(key=lambda entry: ((entry.title or "").lower(), entry.video_id))
    return entries


def _build_entries_from_rows(
    rows: list[dict[str, str]],
    transcript_map: dict[str, Path],
) -> list[ScanEntry]:
    entries: list[ScanEntry] = []
    for row in rows:
        video_id = (row.get("video_id") or "").strip()
        if not video_id:
            continue
        all_ids = _split_pipe(row.get("all_playlist_ids"))
        all_titles = _split_pipe(row.get("all_playlist_titles"))
        transcript_path = transcript_map.get(video_id)
        entries.append(
            ScanEntry(
                video_id=video_id,
                title=(row.get("title") or "").strip(),
                url=(row.get("url") or "").strip(),
                status=(row.get("status") or "unknown").strip().lower(),
                language=(row.get("language") or "").strip(),
                source=(row.get("source") or "").strip(),
                assigned_playlist_id=(row.get("assigned_playlist_id") or "").strip(),
                assigned_playlist_title=(row.get("assigned_playlist_title") or "").strip(),
                all_playlist_ids=all_ids,
                all_playlist_titles=all_titles,
                has_transcript=transcript_path is not None,
                transcript_path=str(transcript_path) if transcript_path else "",
            )
        )
    return entries


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _scan_transcript_files(channel_dir: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    transcript_map: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = defaultdict(list)
    if not channel_dir.exists():
        return transcript_map, duplicates
    for path in channel_dir.rglob("*.txt"):
        if META_FOLDER in path.parts:
            continue
        video_id = extract_video_id_from_stem(path.stem)
        if not video_id:
            continue
        if video_id not in transcript_map:
            transcript_map[video_id] = path
        else:
            duplicates[video_id].append(path)
    return transcript_map, duplicates


def _persist_scan_index(
    channel_dir: Path,
    channel_info: ChannelInfo,
    channel_base_url: str,
    videos_by_id: dict[str, VideoRecord],
    playlists: list[PlaylistRecord],
) -> None:
    """
    Persist a lightweight index of the discovered videos into the channel
    folder. We do not write the `run_report.json` because no transcripts
    were produced.
    """
    if not videos_by_id:
        return
    meta_dir = channel_dir / META_FOLDER
    meta_dir.mkdir(parents=True, exist_ok=True)

    membership: dict[str, list[str]] = defaultdict(list)
    for playlist in playlists:
        for video in playlist.videos:
            if playlist.playlist_id not in membership[video.video_id]:
                membership[video.video_id].append(playlist.playlist_id)

    playlist_by_id = {pl.playlist_id: pl for pl in playlists}

    rows: list[dict[str, str]] = []
    for video_id, video in videos_by_id.items():
        ids = membership.get(video_id, [])
        assigned_id = ids[0] if ids else ""
        assigned_playlist = playlist_by_id.get(assigned_id)
        rows.append(
            {
                "video_id": video_id,
                "title": video.title,
                "url": video.url,
                "assigned_playlist_id": assigned_id,
                "assigned_playlist_title": assigned_playlist.title if assigned_playlist else "",
                "all_playlist_ids": "|".join(ids),
                "all_playlist_titles": "|".join(
                    playlist_by_id[pid].title for pid in ids if pid in playlist_by_id
                ),
                "folder": "",
                "transcript_path": "",
                "status": "indexed",
                "language": "",
                "source": "",
                "error": "",
            }
        )

    headers = [
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
    video_index_path = meta_dir / VIDEO_INDEX_FILE
    with video_index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if playlists:
        playlists_path = meta_dir / PLAYLISTS_FILE
        with playlists_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["playlist_id", "title", "url", "video_count", "folder"],
            )
            writer.writeheader()
            for playlist in playlists:
                writer.writerow(
                    {
                        "playlist_id": playlist.playlist_id,
                        "title": playlist.title,
                        "url": playlist.url,
                        "video_count": str(len(playlist.videos)),
                        "folder": "",
                    }
                )

    (meta_dir / "scan_info.json").write_text(
        json.dumps(
            {
                "channel_title": channel_info.title,
                "channel_id": channel_info.channel_id,
                "channel_url": channel_base_url,
                "scanned_at": now_utc_iso(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_channel_status(result: ScanResult) -> Path:
    meta_dir = result.channel_dir / META_FOLDER
    meta_dir.mkdir(parents=True, exist_ok=True)
    status_path = meta_dir / CHANNEL_STATUS_FILE
    payload = _result_to_dict(result)
    payload["transcribed_videos"] = result.transcribed_count
    payload["pending_videos"] = result.pending_count
    status_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return status_path


def _read_video_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_playlists_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Output / interaction
# ---------------------------------------------------------------------------

def _print_non_interactive_summary(result: ScanResult) -> None:
    print("")
    print("=" * 70)
    print("  Channel scan summary")
    print("=" * 70)
    print(f"  Channel         : {result.channel_title}")
    if result.channel_id:
        print(f"  Channel ID      : {result.channel_id}")
    if result.channel_url:
        print(f"  Channel URL     : {result.channel_url}")
    print(f"  Channel folder  : {result.channel_dir}")
    print(f"  Last updated    : {result.generated_at}")
    print(f"  Total videos    : {result.total_videos}")
    print(f"  Total playlists : {result.total_playlists}")
    print(f"  Transcribed     : {result.transcribed_count}")
    print(f"  Pending         : {result.pending_count}")
    print("")
    print("  Status breakdown:")
    for status, count in sorted(result.status_breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {status:<16s} {count}")
    print("")
    print("  Source breakdown:")
    for source, count in sorted(result.source_breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {source:<16s} {count}")
    print("")
    print(f"  Status JSON     : {result.channel_dir / META_FOLDER / CHANNEL_STATUS_FILE}")
    print("=" * 70)


def _run_interactive_loop(result: ScanResult, page_size: int) -> None:
    if not sys.stdin.isatty():
        # Belt and braces: never block in non-TTY environments.
        return

    actions: dict[str, Any] = {
        "1": ("View summary statistics", lambda: _show_summary(result)),
        "2": ("Browse all videos", lambda: _browse_videos(result, page_size, filter_fn=None)),
        "3": ("List pending videos", lambda: _browse_videos(
            result,
            page_size,
            filter_fn=lambda entry: not entry.has_transcript,
        )),
        "4": ("List transcribed videos", lambda: _browse_videos(
            result,
            page_size,
            filter_fn=lambda entry: entry.has_transcript,
        )),
        "5": ("Search videos by ID or text", lambda: _search_videos(result, page_size)),
        "6": ("Exit", lambda: (_exit_loop())),
    }

    while True:
        print("")
        print("-" * 70)
        print(f"  Channel: {result.channel_title}  ({result.channel_dir})")
        print("-" * 70)
        for key in sorted(actions.keys()):
            label, _ = actions[key]
            print(f"  [{key}] {label}")
        print("-" * 70)
        try:
            choice = input("Choose an action: ").strip()
        except EOFError:
            print("")
            return
        if not choice:
            continue
        action = actions.get(choice)
        if action is None:
            print(f"  Unknown option: {choice!r}")
            continue
        label, handler = action
        handler()
        if choice == "6":
            return


def _exit_loop() -> None:
    print("Exiting scan.")


def _show_summary(result: ScanResult) -> None:
    _print_non_interactive_summary(result)


def _browse_videos(
    result: ScanResult,
    page_size: int,
    filter_fn: Any | None,
) -> None:
    entries = [e for e in result.entries if filter_fn is None or filter_fn(e)]
    if not entries:
        print("")
        print("  No videos match this filter.")
        return

    total = len(entries)
    page = 0
    total_pages = max(1, (total + page_size - 1) // page_size)

    while True:
        start = page * page_size
        end = min(total, start + page_size)
        page_entries = entries[start:end]
        print("")
        print(f"  Videos {start + 1}-{end} of {total} (page {page + 1}/{total_pages})")
        print(f"  {'#':>4}  {'Video ID':<13}  {'Status':<14}  {'Playlist':<32}  Title")
        print(f"  {'-'*4}  {'-'*13}  {'-'*14}  {'-'*32}  -----")
        for index, entry in enumerate(page_entries, start=start + 1):
            playlist = entry.assigned_playlist_title or "(ungrouped)"
            if len(playlist) > 30:
                playlist = playlist[:29] + "…"
            title = entry.title or ""
            if len(title) > 60:
                title = title[:59] + "…"
            print(
                f"  {index:>4}  {entry.video_id:<13}  {entry.status:<14}  {playlist:<32}  {title}"
            )

        print("")
        print("  Controls: [n]ext, [p]rev, [q]uit")
        try:
            cmd = input("  > ").strip().lower()
        except EOFError:
            return
        if cmd in {"q", "quit", "exit"}:
            return
        if cmd in {"n", "next"}:
            if page + 1 < total_pages:
                page += 1
            continue
        if cmd in {"p", "prev"}:
            if page > 0:
                page -= 1
            continue
        if cmd == "":
            continue
        print(f"  Unknown command: {cmd!r}")


def _search_videos(result: ScanResult, page_size: int) -> None:
    try:
        query = input("Search query (id, title, playlist, status): ").strip()
    except EOFError:
        return
    if not query:
        print("  Empty query — cancelled.")
        return

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    matches = [
        entry
        for entry in result.entries
        if (
            pattern.search(entry.video_id)
            or pattern.search(entry.title)
            or pattern.search(entry.assigned_playlist_title)
            or pattern.search(entry.status)
            or any(pattern.search(t) for t in entry.all_playlist_titles)
        )
    ]
    if not matches:
        print(f"  No matches for {query!r}.")
        return

    print(f"  {len(matches)} match(es) for {query!r}:")
    sub_result = ScanResult(
        channel_title=result.channel_title,
        channel_id=result.channel_id,
        channel_url=result.channel_url,
        channel_dir=result.channel_dir,
        total_videos=len(matches),
        total_playlists=result.total_playlists,
        transcribed_count=sum(1 for e in matches if e.has_transcript),
        pending_count=sum(1 for e in matches if not e.has_transcript),
        status_breakdown=Counter(),
        source_breakdown=Counter(),
        playlist_breakdown=Counter(),
        entries=matches,
        generated_at=result.generated_at,
    )
    _browse_videos(sub_result, page_size, filter_fn=None)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def _looks_like_url(value: str) -> bool:
    value = value.strip()
    if value.startswith(("http://", "https://", "www.")):
        return True
    if "://" in value:
        return True
    lowered = value.lower()
    return "youtube.com" in lowered or "youtu.be" in lowered


def _upsert_video(videos_by_id: dict[str, VideoRecord], incoming: VideoRecord) -> None:
    current = videos_by_id.get(incoming.video_id)
    if not current:
        videos_by_id[incoming.video_id] = incoming
        return
    if not current.title and incoming.title:
        current.title = incoming.title
    if not current.url and incoming.url:
        current.url = incoming.url
    if not current.channel and incoming.channel:
        current.channel = incoming.channel
    if not current.channel_id and incoming.channel_id:
        current.channel_id = incoming.channel_id


def _split_pipe(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def is_transcribed_status(status: str) -> bool:
    return (status or "").strip().lower() in _TRANSCRIBED_STATUSES
