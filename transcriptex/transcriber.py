from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transcriptex.fs_utils import (
    build_transcript_filename,
    ensure_dir,
    extract_video_id_from_stem,
    now_utc_iso,
    sanitize_name,
)
from transcriptex.models import PlaylistRecord, SubtitleTrack, TranscriptResult, VideoRecord
from transcriptex.subtitle_parser import parse_subtitle_bytes
from transcriptex.url_utils import normalize_channel_url, video_url_from_id
from transcriptex.youtube_client import YouTubeClient, infer_subtitle_ext

DEFAULT_TABS = ("videos", "shorts", "streams")
UNGROUPED_FOLDER = "Sin_Playlist"
META_FOLDER = "_meta"
SUBTITLE_EXT_PRIORITY = ["vtt", "srv3", "srv2", "srv1", "ttml", "xml", "srt", "json3", "json"]
STATUS_OK = "ok"
STATUS_EXISTING = "existing"
STATUS_NO_SUBTITLES = "no_subtitles"
STATUS_NO_TEXT = "no_text"
STATUS_FAILED = "failed"


@dataclass
class TranscriberConfig:
    channel_url: str
    output_dir: Path
    lang_priority: list[str]
    include_tabs: list[str]
    retries: int = 2
    force: bool = False
    verbose: bool = False
    timeout: int = 30
    sleep_seconds: float = 0.0


class ChannelTranscriber:
    def __init__(self, config: TranscriberConfig) -> None:
        self.config = config
        self.client = YouTubeClient(
            retries=self.config.retries,
            timeout=self.config.timeout,
            verbose=self.config.verbose,
        )

    def run(self) -> dict[str, Any]:
        started_at = now_utc_iso()
        channel_base_url = normalize_channel_url(self.config.channel_url)
        channel_info = self.client.discover_channel(channel_base_url)
        channel_folder_name = sanitize_name(channel_info.title, max_len=120, fallback="Channel")

        channel_dir = ensure_dir(self.config.output_dir / channel_folder_name)
        meta_dir = ensure_dir(channel_dir / META_FOLDER)
        ungrouped_dir = ensure_dir(channel_dir / UNGROUPED_FOLDER)
        existing_transcripts = self._scan_existing_transcripts(channel_dir)

        videos_by_id: dict[str, VideoRecord] = {}
        playlist_membership: dict[str, list[str]] = defaultdict(list)

        print(f"Channel: {channel_info.title}")
        print(f"Output: {channel_dir}")

        self._collect_channel_tab_videos(
            videos_by_id=videos_by_id,
            channel_base_url=channel_base_url,
            fallback_channel=channel_info.title,
        )
        playlists = self._collect_playlists_and_membership(
            channel_base_url=channel_base_url,
            videos_by_id=videos_by_id,
            playlist_membership=playlist_membership,
            fallback_channel=channel_info.title,
            channel_dir=channel_dir,
        )

        if not videos_by_id:
            raise RuntimeError("No videos found for this channel URL")

        playlist_by_id = {playlist.playlist_id: playlist for playlist in playlists}
        playlist_folders = {
            playlist.playlist_id: channel_dir
            / f"{sanitize_name(playlist.title, max_len=90, fallback='Playlist')} [{playlist.playlist_id}]"
            for playlist in playlists
        }

        for folder in playlist_folders.values():
            ensure_dir(folder)

        ordered_video_ids = sorted(
            videos_by_id.keys(),
            key=lambda video_id: ((videos_by_id[video_id].title or "").lower(), video_id),
        )
        total_videos = len(ordered_video_ids)
        print(f"Videos indexed: {total_videos}")
        print(f"Playlists indexed: {len(playlists)}")

        video_rows: list[dict[str, str]] = []
        status_counter = {
            STATUS_OK: 0,
            STATUS_EXISTING: 0,
            STATUS_NO_SUBTITLES: 0,
            STATUS_NO_TEXT: 0,
            STATUS_FAILED: 0,
        }

        for index, video_id in enumerate(ordered_video_ids, start=1):
            video = videos_by_id[video_id]
            memberships = playlist_membership.get(video_id, [])
            assigned_playlist_id = memberships[0] if memberships else ""
            assigned_playlist = playlist_by_id.get(assigned_playlist_id)
            target_dir = playlist_folders.get(assigned_playlist_id, ungrouped_dir)
            transcript_name = build_transcript_filename(video_id=video_id, title=video.title or video_id)
            transcript_path = target_dir / transcript_name

            if not self.config.force and video_id in existing_transcripts:
                result = TranscriptResult(
                    status=STATUS_EXISTING,
                    transcript_path=existing_transcripts[video_id],
                )
            else:
                result = self._transcribe_video(video=video, transcript_path=transcript_path)
                if result.status == STATUS_OK and result.transcript_path:
                    existing_transcripts[video_id] = result.transcript_path

            status_counter[result.status] = status_counter.get(result.status, 0) + 1
            playlist_titles = [playlist_by_id[playlist_id].title for playlist_id in memberships if playlist_id in playlist_by_id]
            actual_folder = result.transcript_path.parent if result.transcript_path else target_dir

            video_rows.append(
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "url": video.url,
                    "assigned_playlist_id": assigned_playlist_id,
                    "assigned_playlist_title": assigned_playlist.title if assigned_playlist else "",
                    "all_playlist_ids": "|".join(memberships),
                    "all_playlist_titles": "|".join(playlist_titles),
                    "folder": self._relative_path_str(actual_folder, channel_dir),
                    "transcript_path": self._relative_path_str(result.transcript_path, channel_dir)
                    if result.transcript_path
                    else "",
                    "status": result.status,
                    "language": result.language,
                    "source": result.source,
                    "error": result.error,
                }
            )

            print(f"[{index}/{total_videos}] {video.video_id} -> {result.status}")
            if self.config.sleep_seconds > 0:
                time.sleep(self.config.sleep_seconds)

        self._write_playlists_csv(
            path=meta_dir / "playlists.csv",
            playlists=playlists,
            playlist_folders=playlist_folders,
            channel_dir=channel_dir,
        )
        self._write_video_index_csv(meta_dir / "video_index.csv", video_rows)

        report = {
            "started_at": started_at,
            "finished_at": now_utc_iso(),
            "channel_title": channel_info.title,
            "channel_id": channel_info.channel_id,
            "channel_url": channel_base_url,
            "output_dir": str(channel_dir),
            "total_videos": total_videos,
            "playlists": len(playlists),
            "processed_ok": status_counter.get(STATUS_OK, 0),
            "already_existing": status_counter.get(STATUS_EXISTING, 0),
            "no_subtitles": status_counter.get(STATUS_NO_SUBTITLES, 0),
            "no_text": status_counter.get(STATUS_NO_TEXT, 0),
            "failed": status_counter.get(STATUS_FAILED, 0),
        }
        (meta_dir / "run_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return report

    def _collect_channel_tab_videos(
        self,
        videos_by_id: dict[str, VideoRecord],
        channel_base_url: str,
        fallback_channel: str,
    ) -> None:
        tabs = self.config.include_tabs or list(DEFAULT_TABS)
        indexed_before = len(videos_by_id)
        for tab in tabs:
            tab_value = tab.strip()
            if not tab_value:
                continue
            try:
                records = self.client.extract_tab_videos(
                    channel_base_url=channel_base_url,
                    tab=tab_value,
                    fallback_channel=fallback_channel,
                )
            except Exception as exc:
                print(f"Warning: failed to read tab '{tab_value}': {exc}")
                continue

            for record in records:
                self._upsert_video(videos_by_id, record)
            print(f"Tab '{tab_value}': {len(records)} videos indexed")

        if len(videos_by_id) > indexed_before:
            return

        try:
            records = self.client.extract_tab_videos(
                channel_base_url=channel_base_url,
                tab="",
                fallback_channel=fallback_channel,
            )
            for record in records:
                self._upsert_video(videos_by_id, record)
            print(f"Fallback base channel URL: {len(records)} videos indexed")
        except Exception as exc:
            print(f"Warning: base channel fallback failed: {exc}")

    def _collect_playlists_and_membership(
        self,
        channel_base_url: str,
        videos_by_id: dict[str, VideoRecord],
        playlist_membership: dict[str, list[str]],
        fallback_channel: str,
        channel_dir: Path,
    ) -> list[PlaylistRecord]:
        try:
            playlists = self.client.extract_playlists(channel_base_url)
        except Exception as exc:
            print(f"Warning: failed to list playlists: {exc}")
            return []

        for playlist in playlists:
            try:
                records = self.client.extract_playlist_videos(
                    playlist_url=playlist.url,
                    fallback_channel=fallback_channel,
                )
            except Exception as exc:
                print(f"Warning: failed to read playlist '{playlist.title}': {exc}")
                records = []

            playlist.videos = records
            for record in records:
                self._upsert_video(videos_by_id, record)
                if playlist.playlist_id not in playlist_membership[record.video_id]:
                    playlist_membership[record.video_id].append(playlist.playlist_id)

            folder = channel_dir / f"{sanitize_name(playlist.title, max_len=90, fallback='Playlist')} [{playlist.playlist_id}]"
            print(f"Playlist '{playlist.title}': {len(records)} videos -> {folder.name}")
        return playlists

    def _transcribe_video(self, video: VideoRecord, transcript_path: Path) -> TranscriptResult:
        last_error = ""
        attempts = max(1, self.config.retries + 1)
        for _ in range(attempts):
            try:
                info = self.client.extract_video_info(video.url or video_url_from_id(video.video_id))
                if not info:
                    raise RuntimeError("yt-dlp returned empty video metadata")

                video.title = video.title or str(info.get("title") or "")
                video.channel = video.channel or str(info.get("channel") or info.get("uploader") or "")
                canonical_url = str(info.get("webpage_url") or info.get("original_url") or "")
                if canonical_url:
                    video.url = canonical_url

                track = select_subtitle_track(info, self.config.lang_priority)
                if not track:
                    return TranscriptResult(
                        status=STATUS_NO_SUBTITLES,
                        error="No subtitle tracks found",
                    )

                subtitle_bytes = self.client.fetch_binary(track.url)
                transcript_text = parse_subtitle_bytes(subtitle_bytes, ext_hint=track.ext).strip()
                if not transcript_text:
                    return TranscriptResult(
                        status=STATUS_NO_TEXT,
                        language=track.language,
                        source=track.source,
                        error="Subtitle track is empty after parsing",
                    )

                self._write_transcript_file(
                    path=transcript_path,
                    video=video,
                    track=track,
                    transcript_text=transcript_text,
                )
                return TranscriptResult(
                    status=STATUS_OK,
                    transcript_path=transcript_path,
                    language=track.language,
                    source=track.source,
                )
            except Exception as exc:
                last_error = str(exc)
                continue

        return TranscriptResult(status=STATUS_FAILED, error=last_error or "Unknown error")

    def _write_transcript_file(
        self,
        path: Path,
        video: VideoRecord,
        track: SubtitleTrack,
        transcript_text: str,
    ) -> None:
        ensure_dir(path.parent)
        lines = [
            f"Title: {video.title}",
            f"Video ID: {video.video_id}",
            f"Video URL: {video.url}",
            f"Channel: {video.channel}",
            f"Subtitle language: {track.language}",
            f"Subtitle source: {track.source}",
            f"Generated at: {now_utc_iso()}",
            "",
            "-----",
            "",
            transcript_text,
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _scan_existing_transcripts(self, channel_dir: Path) -> dict[str, Path]:
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

    def _upsert_video(self, videos_by_id: dict[str, VideoRecord], incoming: VideoRecord) -> None:
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

    def _relative_path_str(self, path: Path | None, base: Path) -> str:
        if not path:
            return ""
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

    def _write_playlists_csv(
        self,
        path: Path,
        playlists: list[PlaylistRecord],
        playlist_folders: dict[str, Path],
        channel_dir: Path,
    ) -> None:
        headers = ["playlist_id", "title", "url", "video_count", "folder"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for playlist in playlists:
                writer.writerow(
                    {
                        "playlist_id": playlist.playlist_id,
                        "title": playlist.title,
                        "url": playlist.url,
                        "video_count": str(len(playlist.videos)),
                        "folder": self._relative_path_str(
                            playlist_folders.get(playlist.playlist_id),
                            channel_dir,
                        ),
                    }
                )

    def _write_video_index_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
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
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def select_subtitle_track(video_info: dict[str, Any], lang_priority: list[str]) -> SubtitleTrack | None:
    manual_tracks = _collect_tracks(video_info.get("subtitles"), source="manual")
    auto_tracks = _collect_tracks(video_info.get("automatic_captions"), source="auto")

    language_preferences = [lang.lower().strip() for lang in lang_priority if lang.strip()]
    if language_preferences:
        selected = _pick_by_language(manual_tracks, language_preferences)
        if selected:
            return selected
        selected = _pick_by_language(auto_tracks, language_preferences)
        if selected:
            return selected

    selected = _pick_best_any(manual_tracks)
    if selected:
        return selected
    return _pick_best_any(auto_tracks)


def _collect_tracks(payload: Any, source: str) -> list[SubtitleTrack]:
    if not isinstance(payload, dict):
        return []

    tracks: list[SubtitleTrack] = []
    for language, formats in payload.items():
        if not isinstance(formats, list):
            continue
        for track in formats:
            if not isinstance(track, dict):
                continue
            url = str(track.get("url") or "").strip()
            if not url:
                continue
            tracks.append(
                SubtitleTrack(
                    language=str(language),
                    ext=infer_subtitle_ext(track),
                    url=url,
                    source=source,
                )
            )
    return tracks


def _pick_by_language(tracks: list[SubtitleTrack], preferences: list[str]) -> SubtitleTrack | None:
    for preferred in preferences:
        candidates = [track for track in tracks if _language_matches(track.language, preferred)]
        if not candidates:
            continue
        return sorted(candidates, key=_track_rank)[0]
    return None


def _pick_best_any(tracks: list[SubtitleTrack]) -> SubtitleTrack | None:
    if not tracks:
        return None
    return sorted(tracks, key=_track_rank)[0]


def _language_matches(actual: str, preferred: str) -> bool:
    value = actual.lower()
    target = preferred.lower()
    if value == target:
        return True
    return value.startswith(f"{target}-") or value.startswith(f"{target}_")


def _track_rank(track: SubtitleTrack) -> tuple[int, str]:
    try:
        ext_rank = SUBTITLE_EXT_PRIORITY.index(track.ext)
    except ValueError:
        ext_rank = len(SUBTITLE_EXT_PRIORITY) + 1
    return (ext_rank, track.language)
