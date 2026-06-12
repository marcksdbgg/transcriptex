"""
Shared yt-dlp configuration for YouTube downloads.

Since yt-dlp 2025.11+, YouTube requires an external JavaScript runtime
(Node, Deno, …) plus EJS challenge-solver scripts to access audio/video
formats. Without them, only storyboard images are listed and downloads fail
with "Requested format is not available".
"""
from __future__ import annotations

import shutil
from typing import Any


class YouTubeRateLimitError(Exception):
    """Raised when YouTube temporarily rate-limits the whole session."""


class PermanentVideoError(Exception):
    """Raised when a video can never be downloaded (private, removed, etc.)."""


def detect_js_runtime() -> str | None:
    """Return the first supported JS runtime found on PATH."""
    for name in ("node", "deno", "bun"):
        if shutil.which(name):
            return name
    return None


def ensure_yt_dlp_ejs() -> None:
    """Ensure yt-dlp and the EJS solver script package are installed."""
    from transcriptex.dependency_manager import ensure

    ensure("yt_dlp:yt-dlp")
    try:
        import yt_dlp_ejs  # noqa: F401
    except ImportError:
        ensure("yt_dlp_ejs:yt-dlp-ejs")


def apply_youtube_ejs_opts(opts: dict[str, Any]) -> dict[str, Any]:
    """
    Merge YouTube EJS options into a yt-dlp options dict (in place).

    Enables Node/Deno/Bun for n-challenge solving when available.
    """
    ensure_yt_dlp_ejs()
    runtime = detect_js_runtime()
    if runtime:
        opts["js_runtimes"] = {runtime: {}}
    return opts


def apply_youtube_throttle_opts(
    opts: dict[str, Any],
    *,
    sleep_interval: float = 2.0,
    sleep_interval_requests: float = 1.0,
) -> dict[str, Any]:
    """Add yt-dlp request throttling to reduce YouTube rate-limit bans."""
    opts["sleep_interval"] = max(0.0, sleep_interval)
    opts["max_sleep_interval"] = max(5.0, sleep_interval * 5)
    opts["sleep_interval_requests"] = max(0.0, sleep_interval_requests)
    return opts


def is_rate_limit_error(exc: BaseException) -> bool:
    """True when YouTube temporarily blocked the session (not a dead video)."""
    msg = str(exc).lower()
    return (
        "rate-limited" in msg
        or "rate limited" in msg
        or ("try again later" in msg and "rate" in msg)
        or "this content isn't available, try again later" in msg
    )


def is_private_video_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "private video" in msg


def is_permanent_download_error(exc: BaseException) -> bool:
    """Errors that will not succeed by waiting — skip retries."""
    if is_rate_limit_error(exc):
        return False
    msg = str(exc).lower()
    permanent = (
        "private video",
        "video unavailable",
        "has been removed",
        "copyright",
        "blocked in your country",
        "not available",
        "sign in to confirm your age",
        "members only",
        "join this channel",
    )
    # "video unavailable" alone can be rate-limit; already excluded above
    return any(kw in msg for kw in permanent if kw != "video unavailable") or (
        "video unavailable" in msg and "try again later" not in msg
    )


def audio_format_selector() -> str:
    """Flexible audio format chain when bestaudio alone is unavailable."""
    return "bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio/best"
