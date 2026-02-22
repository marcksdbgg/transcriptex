from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
KNOWN_CHANNEL_TABS = {
    "videos",
    "shorts",
    "streams",
    "playlists",
    "featured",
    "about",
    "community",
}


def normalize_channel_url(raw_url: str) -> str:
    value = (raw_url or "").strip()
    if not value:
        raise ValueError("channel_url is required")

    if "://" not in value:
        value = f"https://{value}"

    parsed = urlparse(value)
    host = parsed.netloc.lower()
    if "youtube.com" not in host:
        raise ValueError("Only youtube.com channel URLs are supported")

    segments = [segment for segment in parsed.path.split("/") if segment]
    if not segments:
        raise ValueError("Invalid channel URL path")

    base_segments: list[str]
    first = segments[0]
    if first.startswith("@"):
        base_segments = [first]
    elif first in {"channel", "user", "c"}:
        if len(segments) < 2:
            raise ValueError("Invalid channel URL path")
        base_segments = [first, segments[1]]
    else:
        if first in KNOWN_CHANNEL_TABS:
            raise ValueError("Expected a channel URL, got a tab URL without channel identifier")
        base_segments = [first]

    return f"https://www.youtube.com/{'/'.join(base_segments)}"


def channel_tab_url(channel_base_url: str, tab: str) -> str:
    tab_value = (tab or "").strip("/")
    if not tab_value:
        return channel_base_url
    return f"{channel_base_url}/{tab_value}"


def extract_video_id(candidate: str | None) -> str:
    value = (candidate or "").strip()
    if not value:
        return ""

    if VIDEO_ID_RE.fullmatch(value):
        return value

    parsed = urlparse(value)
    if parsed.scheme:
        query = parse_qs(parsed.query)
        if "v" in query and query["v"]:
            maybe = query["v"][0].strip()
            if VIDEO_ID_RE.fullmatch(maybe):
                return maybe
        parts = [segment for segment in parsed.path.split("/") if segment]
        if len(parts) >= 2 and parts[-2] in {"shorts", "embed", "live", "watch"}:
            maybe = parts[-1]
            if VIDEO_ID_RE.fullmatch(maybe):
                return maybe
        if parts:
            maybe = parts[-1]
            if VIDEO_ID_RE.fullmatch(maybe):
                return maybe
        return ""

    if "watch?v=" in value:
        parsed = urlparse(f"https://www.youtube.com/{value.lstrip('/')}")
        query = parse_qs(parsed.query)
        maybe = query.get("v", [""])[0]
        return maybe if VIDEO_ID_RE.fullmatch(maybe) else ""

    relative_parts = [segment for segment in value.split("/") if segment]
    if len(relative_parts) >= 2 and relative_parts[-2] in {"shorts", "embed", "live", "watch"}:
        maybe = relative_parts[-1]
        if VIDEO_ID_RE.fullmatch(maybe):
            return maybe
    if relative_parts:
        maybe = relative_parts[-1]
        if VIDEO_ID_RE.fullmatch(maybe):
            return maybe

    return ""


def extract_playlist_id(candidate: str | None) -> str:
    value = (candidate or "").strip()
    if not value:
        return ""

    if value.startswith(("PL", "UU", "LL", "RD", "OLAK5uy_")):
        return value

    parsed = urlparse(value)
    if parsed.scheme:
        query = parse_qs(parsed.query)
        maybe = query.get("list", [""])[0]
        return maybe.strip()

    return ""


def video_url_from_id(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def normalize_relative_youtube_url(candidate: str, video_id: str = "") -> str:
    value = (candidate or "").strip()
    if not value:
        return video_url_from_id(video_id) if video_id else ""
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return f"https://www.youtube.com{value}"
    return video_url_from_id(video_id) if video_id else value
