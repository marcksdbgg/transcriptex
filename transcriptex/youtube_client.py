from __future__ import annotations

import os
import re
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from transcriptex.models import ChannelInfo, PlaylistRecord, VideoRecord
from transcriptex.url_utils import (
    channel_tab_url,
    extract_playlist_id,

    extract_video_id,
    normalize_relative_youtube_url,
    video_url_from_id,
)

PLAYLIST_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _require_yt_dlp():
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:  # pragma: no cover - import path only
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return yt_dlp


class YouTubeClient:
    def __init__(self, retries: int = 2, timeout: int = 30, verbose: bool = False) -> None:
        self.retries = max(0, retries)
        self.timeout = max(5, timeout)
        self.verbose = verbose

    def _base_opts(self, ignoreerrors: bool = True) -> dict[str, Any]:
        return {
            "quiet": not self.verbose,
            "no_warnings": not self.verbose,
            "ignoreerrors": bool(ignoreerrors),
            "extractor_retries": self.retries,
            "retries": self.retries,
            "socket_timeout": self.timeout,
            "skip_download": True,
        }

    def extract_info(self, url: str, flat: bool, ignoreerrors: bool = True) -> dict[str, Any]:
        yt_dlp = _require_yt_dlp()
        options = self._base_opts(ignoreerrors=ignoreerrors)
        if flat:
            options["extract_flat"] = True
            options["lazy_playlist"] = True
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        return info or {}

    def discover_channel(self, channel_base_url: str) -> ChannelInfo:
        info = self.extract_info(channel_base_url, flat=True)
        channel_id = str(info.get("channel_id") or "")
        info_id = str(info.get("id") or "")
        if not channel_id and info_id.startswith("UC"):
            channel_id = info_id
        title = str(info.get("channel") or info.get("uploader") or info.get("title") or "").strip()
        handle = str(info.get("uploader_id") or "").strip()
        if not title and handle:
            title = handle.lstrip("@")
        if not title:
            title = channel_base_url.rstrip("/").split("/")[-1].lstrip("@") or "Channel"
        return ChannelInfo(
            base_url=channel_base_url,
            title=title,
            channel_id=channel_id,
            handle=handle,
        )

    def extract_tab_videos(self, channel_base_url: str, tab: str, fallback_channel: str = "") -> list[VideoRecord]:
        tab_url = channel_tab_url(channel_base_url, tab)
        info = self.extract_info(tab_url, flat=True)
        return self._videos_from_entries(info, fallback_channel=fallback_channel)

    def extract_playlists(self, channel_base_url: str) -> list[PlaylistRecord]:
        playlists_url = channel_tab_url(channel_base_url, "playlists")
        info = self.extract_info(playlists_url, flat=True)
        playlists: list[PlaylistRecord] = []
        seen: set[str] = set()
        for entry in self._iter_entries(info):
            playlist_id = extract_playlist_id(str(entry.get("id") or ""))
            if not playlist_id:
                playlist_id = extract_playlist_id(str(entry.get("url") or ""))
            if not playlist_id:
                playlist_id = extract_playlist_id(str(entry.get("webpage_url") or ""))
            if not playlist_id or playlist_id in seen:
                continue
            if not PLAYLIST_ID_RE.fullmatch(playlist_id):
                continue
            seen.add(playlist_id)
            title = str(entry.get("title") or "").strip() or playlist_id
            playlists.append(
                PlaylistRecord(
                    playlist_id=playlist_id,
                    title=title,
                    url=f"https://www.youtube.com/playlist?list={playlist_id}",
                )
            )
        return playlists

    def extract_playlist_videos(self, playlist_url: str, fallback_channel: str = "") -> list[VideoRecord]:
        info = self.extract_info(playlist_url, flat=True)
        return self._videos_from_entries(info, fallback_channel=fallback_channel)

    def extract_video_info(self, video_url: str) -> dict[str, Any]:
        return self.extract_info(video_url, flat=False, ignoreerrors=False)

    def fetch_binary(self, url: str) -> bytes:
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            },
        )
        with urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def download_audio(self, url: str, tmp_dir: Path, out_name: str = "audio") -> Path:
        """
        Download best audio from a YouTube URL to tmp_dir using yt-dlp.
        Returns the path to the downloaded audio file (.wav fallback).
        """
        yt_dlp = _require_yt_dlp()
        output_template = str(tmp_dir / f"{out_name}.%(ext)s")

        ydl_opts: dict[str, Any] = {
            "quiet": not self.verbose,
            "no_warnings": not self.verbose,
            "noprogress": not self.verbose,
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "0",
                }
            ],
            "ignoreerrors": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                raise RuntimeError("yt-dlp returned empty metadata — video may be private or unavailable")

        # Find the downloaded WAV file
        wav_candidates = list(tmp_dir.glob("*.wav"))
        if wav_candidates:
            return wav_candidates[0]

        # Fallback: any audio file
        for ext in ("m4a", "webm", "opus", "mp3", "ogg", "flac"):
            candidates = list(tmp_dir.glob(f"*.{ext}"))
            if candidates:
                return candidates[0]

        raise RuntimeError(f"No audio file found in {tmp_dir} after download")

    def _videos_from_entries(self, info: dict[str, Any], fallback_channel: str = "") -> list[VideoRecord]:
        videos: list[VideoRecord] = []
        seen: set[str] = set()
        for entry in self._iter_entries(info):
            record = self._record_from_entry(entry, fallback_channel=fallback_channel)
            if not record or record.video_id in seen:
                continue
            seen.add(record.video_id)
            videos.append(record)
        return videos

    def _record_from_entry(self, entry: dict[str, Any], fallback_channel: str = "") -> VideoRecord | None:
        raw_id = str(entry.get("id") or "")
        video_id = extract_video_id(raw_id)
        if not video_id:
            video_id = extract_video_id(str(entry.get("url") or ""))
        if not video_id:
            video_id = extract_video_id(str(entry.get("webpage_url") or ""))
        if not video_id:
            return None

        title = str(entry.get("title") or "").strip()
        url = normalize_relative_youtube_url(str(entry.get("webpage_url") or ""), video_id=video_id)
        if not url:
            url = normalize_relative_youtube_url(str(entry.get("url") or ""), video_id=video_id)
        if not url:
            url = video_url_from_id(video_id)
        channel = str(entry.get("channel") or entry.get("uploader") or fallback_channel or "").strip()
        channel_id = str(entry.get("channel_id") or "").strip()
        return VideoRecord(
            video_id=video_id,
            title=title,
            url=url,
            channel=channel,
            channel_id=channel_id,
        )

    def _iter_entries(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        entries = info.get("entries")
        if not entries:
            return []

        output: list[dict[str, Any]] = []
        queue = deque(entries)
        while queue:
            item = queue.popleft()
            if not isinstance(item, dict):
                continue
            nested = item.get("entries")
            if nested:
                queue.extend(nested)
                continue
            output.append(item)
        return output


def infer_subtitle_ext(track: dict[str, Any]) -> str:
    ext = str(track.get("ext") or "").strip().lower()
    if ext:
        return ext
    url = str(track.get("url") or "")
    path = urlparse(url).path.lower()
    for candidate in (".vtt", ".srt", ".ttml", ".xml", ".json3", ".json", ".srv3", ".srv2", ".srv1"):
        if path.endswith(candidate):
            return candidate.lstrip(".")
    return "vtt"
