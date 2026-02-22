from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path

INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
MULTISPACE_RE = re.compile(r"\s+")
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
VIDEO_ID_BRACKET_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]$")
VIDEO_ID_ANYWHERE_RE = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")


def sanitize_name(value: str, max_len: int = 110, fallback: str = "untitled") -> str:
    text = html.unescape(value or "")
    text = INVALID_FILENAME_CHARS_RE.sub("_", text)
    text = MULTISPACE_RE.sub(" ", text).strip(" .")
    if not text:
        text = fallback
    if len(text) > max_len:
        text = text[:max_len].rstrip(" .")
    return text


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_list_arg(value: str) -> list[str]:
    parts: list[str] = []
    for chunk in (value or "").split(","):
        item = chunk.strip()
        if not item:
            continue
        if item not in parts:
            parts.append(item)
    return parts


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_transcript_filename(video_id: str, title: str, max_len: int = 170) -> str:
    suffix = f" [{video_id}]"
    safe_title = sanitize_name(title, max_len=max_len, fallback=video_id)
    allowed_title_len = max(10, max_len - len(suffix))
    if len(safe_title) > allowed_title_len:
        safe_title = safe_title[:allowed_title_len].rstrip(" .")
    return f"{safe_title}{suffix}.txt"


def extract_video_id_from_stem(stem: str) -> str:
    value = (stem or "").strip()
    if not value:
        return ""
    if VIDEO_ID_RE.fullmatch(value):
        return value
    bracket_match = VIDEO_ID_BRACKET_RE.search(value)
    if bracket_match:
        return bracket_match.group(1)
    any_match = VIDEO_ID_ANYWHERE_RE.search(value)
    if any_match:
        return any_match.group(1)
    return ""
