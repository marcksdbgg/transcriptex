from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoRecord:
    video_id: str
    title: str
    url: str
    channel: str = ""
    channel_id: str = ""


@dataclass
class PlaylistRecord:
    playlist_id: str
    title: str
    url: str
    videos: list[VideoRecord] = field(default_factory=list)


@dataclass
class ChannelInfo:
    base_url: str
    title: str
    channel_id: str
    handle: str = ""


@dataclass
class SubtitleTrack:
    language: str
    ext: str
    url: str
    source: str


@dataclass
class TranscriptResult:
    status: str
    transcript_path: Path | None = None
    language: str = ""
    source: str = ""
    error: str = ""
