"""
cookies.py — Browser cookie extraction and storage for yt-dlp.

Provides an interactive onboarding flow that extracts YouTube cookies
from the user's browser, stores them persistently, and auto-detects
them for all subcommands that need authentication.

Extraction strategy (tried in order):
  1. yt-dlp --cookies-from-browser (after killing background processes)
  2. browser_cookie3 library (handles DPAPI natively)
  3. Manual export via browser extension (guided instructions)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APPNAME = "transcriptex"

BROWSER_MAP: dict[str, str] = {
    "chrome": "Google Chrome",
    "firefox": "Mozilla Firefox",
    "edge": "Microsoft Edge",
    "brave": "Brave",
    "opera": "Opera",
    "vivaldi": "Vivaldi",
    "chromium": "Chromium",
    "safari": "Safari",
    "zen": "Zen Browser",
    "librewolf": "LibreWolf",
    "waterfox": "Waterfox",
}

# Process names to kill before cookie extraction (Windows)
_BROWSER_PROCS: dict[str, list[str]] = {
    "chrome": ["chrome.exe", "GoogleChrome.exe"],
    "firefox": ["firefox.exe"],
    "edge": ["msedge.exe"],
    "brave": ["brave.exe"],
    "opera": ["opera.exe", "launcher.exe"],
    "vivaldi": ["vivaldi.exe"],
    "chromium": ["chrome.exe", "chromium.exe"],
    "zen": ["zen.exe"],
    "librewolf": ["librewolf.exe"],
    "waterfox": ["waterfox.exe"],
}

# ---------------------------------------------------------------------------
# Config directory
# ---------------------------------------------------------------------------


def _config_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
        return Path(base) / _APPNAME
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APPNAME
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        return Path(base) / _APPNAME


def cookies_path() -> Path:
    return _config_dir() / "cookies.txt"


def has_stored_cookies() -> bool:
    return cookies_path().exists()


def get_cookies_file() -> str:
    """
    Return the cookies file path as a string, or empty string if none exists.
    Automatically trims oversized cookie files (>50KB) to prevent HTTP 413.
    """
    path = cookies_path()
    if not path.exists():
        return ""

    size_kb = path.stat().st_size / 1024
    if size_kb > 50:
        # Auto-trim on load to prevent HTTP 413
        try:
            tmp = path.with_suffix(".trim.tmp")
            kept = _filter_essential_cookies(path, tmp)
            if kept > 0 and tmp.exists():
                shutil.move(str(tmp), str(path))
        except Exception:
            pass

    return str(path)


# ---------------------------------------------------------------------------
# Browser detection — comprehensive
# ---------------------------------------------------------------------------

_SUPPORTED_BROWSERS: tuple[str, ...] = (
    "chrome", "firefox", "edge", "brave", "opera", "vivaldi", "chromium",
    "zen", "librewolf", "waterfox",
)
if sys.platform == "darwin":
    _SUPPORTED_BROWSERS = _SUPPORTED_BROWSERS + ("safari",)


def detect_available_browsers() -> list[str]:
    available: list[str] = []

    if sys.platform == "win32":
        paths: dict[str, list[str]] = {
            "chrome": [
                os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
            ],
            "firefox": [
                os.path.join(os.environ.get("PROGRAMFILES", ""), "Mozilla Firefox", "firefox.exe"),
                os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Mozilla Firefox", "firefox.exe"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Mozilla Firefox", "firefox.exe"),
            ],
            "edge": [
                os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(os.environ.get("PROGRAMFILES", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
            ],
            "brave": [
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
                os.path.join(os.environ.get("PROGRAMFILES", ""), "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
            ],
            "opera": [
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Opera", "opera.exe"),
                os.path.join(os.environ.get("PROGRAMFILES", ""), "Opera", "opera.exe"),
            ],
            "vivaldi": [
                os.path.join(os.environ.get("PROGRAMFILES", ""), "Vivaldi", "Application", "vivaldi.exe"),
                os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Vivaldi", "Application", "vivaldi.exe"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Vivaldi", "Application", "vivaldi.exe"),
            ],
            "chromium": [
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Chromium", "Application", "chrome.exe"),
                os.path.join(os.environ.get("PROGRAMFILES", ""), "Chromium", "Application", "chrome.exe"),
            ],
            "zen": [
                os.path.join(os.environ.get("PROGRAMFILES", ""), "Zen Browser", "zen.exe"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "zen-browser", "zen.exe"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "zen-browser", "zen.exe"),
                os.path.join(os.environ.get("PROGRAMFILES", ""), "zen-browser", "zen.exe"),
            ],
            "librewolf": [
                os.path.join(os.environ.get("PROGRAMFILES", ""), "LibreWolf", "librewolf.exe"),
                os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "LibreWolf", "librewolf.exe"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "LibreWolf", "librewolf.exe"),
            ],
            "waterfox": [
                os.path.join(os.environ.get("PROGRAMFILES", ""), "Waterfox", "waterfox.exe"),
                os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Waterfox", "waterfox.exe"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Waterfox", "waterfox.exe"),
            ],
        }
        for browser, locations in paths.items():
            if any(os.path.exists(p) for p in locations):
                available.append(browser)
    elif sys.platform == "darwin":
        mac_paths: dict[str, list[str]] = {
            "chrome": ["/Applications/Google Chrome.app"],
            "firefox": ["/Applications/Firefox.app"],
            "edge": ["/Applications/Microsoft Edge.app"],
            "brave": ["/Applications/Brave Browser.app"],
            "opera": ["/Applications/Opera.app"],
            "safari": ["/Applications/Safari.app"],
            "chromium": ["/Applications/Chromium.app"],
            "vivaldi": ["/Applications/Vivaldi.app"],
        }
        for browser, locations in mac_paths.items():
            if any(os.path.exists(p) for p in locations):
                available.append(browser)
    else:
        linux_bins: dict[str, str] = {
            "chrome": "google-chrome",
            "firefox": "firefox",
            "edge": "microsoft-edge",
            "brave": "brave",
            "opera": "opera",
            "vivaldi": "vivaldi",
            "chromium": "chromium-browser",
        }
        for browser, binary in linux_bins.items():
            if shutil.which(binary):
                available.append(browser)

    return available


# ---------------------------------------------------------------------------
# Kill browser background processes
# ---------------------------------------------------------------------------


def _kill_browser_processes(browser: str) -> int:
    """
    Force-kill all background processes of the given browser.
    Returns the number of processes killed.
    """
    if sys.platform != "win32":
        return 0

    procs = _BROWSER_PROCS.get(browser, [])
    killed = 0

    for proc_name in procs:
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", proc_name, "/T"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # taskkill returns 0 on success, 128 if not found
            if result.returncode == 0:
                # Count lines like "SUCCESS: The process ... has been terminated"
                killed += result.stdout.count("SUCCESS")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    # Give the OS a moment to release file locks
    if killed > 0:
        time.sleep(1.5)

    return killed


# ---------------------------------------------------------------------------
# Cookie extraction — method 1: yt-dlp subprocess
# ---------------------------------------------------------------------------


def _require_yt_dlp():
    from transcriptex.dependency_manager import ensure_yt_dlp

    ensure_yt_dlp()
    import yt_dlp  # type: ignore

    return yt_dlp


def _extract_via_ytdlp_subprocess(browser: str, dest: Path) -> tuple[bool, str]:
    """
    Try to extract cookies using yt-dlp CLI.
    Returns (success, error_message).
    """
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--cookies-from-browser", browser,
        "--cookies", str(dest),
        "--skip-download",
        "--no-warnings",
        "--no-check-certificates",
        "https://www.youtube.com/feed/guidelines",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, "yt-dlp timed out after 120 seconds"

    # yt-dlp may save cookies even with non-zero exit
    if dest.exists() and dest.stat().st_size > 0:
        return True, ""

    stderr = result.stderr.strip()
    # Check for DPAPI error
    if "DPAPI" in stderr or "dpapi" in stderr.lower():
        return False, (
            "Chrome/Edge encrypted the cookies with Windows DPAPI and yt-dlp "
            "could not decrypt them. This usually means the browser is still "
            "running in the background.\n\n"
            "Try again after fully closing the browser, or use Firefox instead "
            "(it does not use DPAPI encryption)."
        )

    return False, stderr[:500] if stderr else "yt-dlp returned no cookies"


# Firefox-based browsers store cookies in cookies.sqlite
_FIREFOX_BASED_PROFILES: dict[str, list[str]] = {
    "firefox": [
        os.path.join(os.environ.get("APPDATA", ""), "Mozilla", "Firefox", "Profiles"),
    ],
    "zen": [
        os.path.join(os.environ.get("APPDATA", ""), "zen", "Profiles"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "zen", "Profiles"),
    ],
    "librewolf": [
        os.path.join(os.environ.get("APPDATA", ""), "LibreWolf", "Profiles"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "LibreWolf", "Profiles"),
    ],
    "waterfox": [
        os.path.join(os.environ.get("APPDATA", ""), "Waterfox", "Profiles"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Waterfox", "Profiles"),
    ],
}


def _extract_firefox_cookies(browser: str, dest: Path) -> tuple[bool, str]:
    """
    Extract YouTube cookies from a Firefox-based browser's cookies.sqlite.
    Works for Firefox, Zen, LibreWolf, Waterfox, etc.
    """
    import sqlite3
    import tempfile

    profile_dirs = _FIREFOX_BASED_PROFILES.get(browser, [])
    if not profile_dirs:
        return False, f"No profile directories found for {browser}"

    # Find the cookies.sqlite file
    cookies_db = None
    for profile_dir in profile_dirs:
        if not os.path.exists(profile_dir):
            continue
        # Look for profile folders (e.g., abc123.default-release)
        for entry in os.listdir(profile_dir):
            candidate = os.path.join(profile_dir, entry, "cookies.sqlite")
            if os.path.exists(candidate):
                cookies_db = candidate
                break
        if cookies_db:
            break

    if not cookies_db:
        return False, f"cookies.sqlite not found for {browser}"

    # Copy to temp to avoid lock issues
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        shutil.copy2(cookies_db, tmp_path)
        conn = sqlite3.connect(tmp_path)
        cursor = conn.execute(
            "SELECT host, path, isSecure, expiry, name, value FROM moz_cookies WHERE host LIKE '%youtube%'"
        )
        rows = cursor.fetchall()
        conn.close()
    finally:
        os.unlink(tmp_path)

    if not rows:
        return False, "No YouTube cookies found in browser profile"

    # Write in Netscape format
    with open(dest, "w", encoding="utf-8") as f:
        f.write("# Netscape HTTP Cookie File\n")
        f.write("# Generated by transcriptex\n\n")
        for host, path, is_secure, expiry, name, value in rows:
            secure = "TRUE" if is_secure else "FALSE"
            f.write(f"{host}\tTRUE\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")

    return True, ""


def _extract_via_browser_cookie3(browser: str, dest: Path) -> tuple[bool, str]:
    """
    Try to extract cookies using the browser_cookie3 library.
    Returns (success, error_message).
    """
    try:
        import browser_cookie3  # type: ignore
    except ImportError:
        return False, "browser_cookie3 not installed"

    # Map browser name to browser_cookie3 function
    # Firefox-based browsers all use the firefox extractor
    extractors: dict[str, object] = {
        "chrome": browser_cookie3.chrome,
        "firefox": browser_cookie3.firefox,
        "edge": browser_cookie3.edge,
        "brave": browser_cookie3.brave,
        "opera": browser_cookie3.opera,
        "vivaldi": browser_cookie3.vivaldi,
        "chromium": browser_cookie3.chromium,
        "zen": browser_cookie3.firefox,  # Firefox fork
        "librewolf": browser_cookie3.firefox,  # Firefox fork
        "waterfox": browser_cookie3.firefox,  # Firefox fork
    }

    extractor = extractors.get(browser)
    if extractor is None:
        return False, f"browser_cookie3 does not support {browser}"

    try:
        cj = extractor(domain_name=".youtube.com")
        cookies = list(cj)
        if not cookies:
            return False, "No YouTube cookies found in browser"

        # Write in Netscape format
        with open(dest, "w", encoding="utf-8") as f:
            f.write("# Netscape HTTP Cookie File\n")
            f.write("# https://curl.haxx.se/rfc/cookie_spec.html\n")
            f.write("# This file was generated by transcriptex\n\n")
            for cookie in cookies:
                domain = cookie.domain or ""
                secure = "TRUE" if cookie.secure else "FALSE"
                path = cookie.path or "/"
                expires = str(cookie.expires or 0)
                name = cookie.name or ""
                value = cookie.value or ""
                f.write(f"{domain}\tTRUE\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")

        return True, ""
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------


def extract_cookies_from_browser(browser: str) -> Path:
    cfg = _config_dir()
    cfg.mkdir(parents=True, exist_ok=True)
    dest = cfg / "cookies.txt"

    browser_label = BROWSER_MAP.get(browser, browser)
    print(f"\nExtracting cookies from {browser_label}...")
    print()

    # Firefox-based browsers: use SQLite extraction directly
    if browser in _FIREFOX_BASED_PROFILES:
        print("  Reading cookies.sqlite from browser profile...")
        success, error = _extract_firefox_cookies(browser, dest)
        if success:
            print("    SUCCESS")
            return _finalize_cookies(dest, browser_label)
        print(f"    Failed: {error}")
    else:
        # Chromium-based browsers: try yt-dlp subprocess
        print("  Method 1: yt-dlp browser extraction...")
        killed = _kill_browser_processes(browser)
        if killed > 0:
            print(f"    Killed {killed} background process(es).")

        success, error = _extract_via_ytdlp_subprocess(browser, dest)
        if success:
            print("    SUCCESS")
            return _finalize_cookies(dest, browser_label)
        print(f"    Failed: {error[:120]}")

        # Method 2: browser_cookie3
        print("  Method 2: browser_cookie3 library...")
        try:
            _install_browser_cookie3()
        except Exception as install_err:
            print(f"    Skipped: {install_err}")
        else:
            success, error = _extract_via_browser_cookie3(browser, dest)
            if success:
                print("    SUCCESS")
                return _finalize_cookies(dest, browser_label)
            print(f"    Failed: {error[:120]}")

    # Manual export
    print()
    print("  Automatic extraction failed.")
    print()
    _print_manual_instructions(browser_label)

    raise RuntimeError(
        f"Could not extract cookies from {browser_label}. "
        "Try the manual export method above."
    )


# Cookies that are actually needed for YouTube authentication.
# Sending every single cookie creates a 200KB+ header that triggers HTTP 413.
_ESSENTIAL_YT_COOKIE_NAMES: set[str] = {
    "VISITOR_INFO1_LIVE",
    "PREF",
    "SID",
    "HSID",
    "SSID",
    "APISID",
    "SAPISID",
    "LOGIN_INFO",
    "YSC",
    "CONSISTENCY",
    "GOOGLE_ABUSE_EXEMPTION",
    "SIDCC",
    "__Secure-1PSID",
    "__Secure-3PSID",
    "__Secure-1PAPISID",
    "__Secure-3PAPISID",
    "__Secure-1PSIDTS",
    "__Secure-3PSIDTS",
    "__Secure-1PSIDCC",
    "__Secure-3PSIDCC",
    "__Secure-ROLLOUT_TOKEN",
    "__Secure-YNID",
    "SOCS",
    "AEC",
    "NID",
    "1P_JAR",
    "DV",
    "CONSENT",
}


def _filter_essential_cookies(src: Path, dest: Path) -> int:
    """
    Filter a Netscape cookies.txt to keep only essential YouTube cookies.
    Returns the number of cookies kept.
    """
    lines: list[str] = []
    kept = 0
    for line in src.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("#") or not line.strip():
            lines.append(line)
            continue
        parts = line.rstrip("\n").split("\t")
        # Netscape format: domain flag path secure expiry name value
        if len(parts) >= 6:
            name = parts[5]
            # Keep only explicitly essential cookies (NOT all ST-* state cookies)
            if name in _ESSENTIAL_YT_COOKIE_NAMES:
                lines.append(line)
                kept += 1
                continue
    dest.write_text("".join(lines), encoding="utf-8")
    return kept


def _finalize_cookies(dest: Path, browser_label: str) -> Path:
    size_before = dest.stat().st_size / 1024

    # Filter to essential cookies only
    tmp = dest.with_suffix(".tmp")
    kept = _filter_essential_cookies(dest, tmp)
    if kept > 0:
        shutil.move(str(tmp), str(dest))
    else:
        tmp.unlink(missing_ok=True)

    size_after = dest.stat().st_size / 1024
    print()
    print(f"  Cookies saved to: {dest}")
    print(f"  Before: {size_before:.1f} KB  |  After: {size_after:.1f} KB  |  Kept: {kept} cookies")
    print("  These cookies will be used automatically for all transcriptex commands.")
    return dest


def _install_browser_cookie3() -> None:
    """Install browser_cookie3 into the current interpreter."""
    try:
        import browser_cookie3  # noqa: F401
        return
    except ImportError:
        pass

    print("    Installing browser_cookie3...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "browser_cookie3", "--quiet"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pip install failed: {result.stderr[:200]}")


# ---------------------------------------------------------------------------
# Manual export instructions
# ---------------------------------------------------------------------------


def _print_manual_instructions(browser_label: str) -> None:
    print("=" * 60)
    print("  Manual Cookie Export")
    print("=" * 60)
    print()
    print(f"  Since automatic extraction from {browser_label} failed,")
    print("  please export cookies manually:")
    print()
    print("  Option A — Browser extension (easiest):")
    print("    1. Install 'Get cookies.txt LOCALLY' extension:")
    print("       https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc")
    print("    2. Open youtube.com and make sure you're logged in")
    print("    3. Click the extension icon to download cookies.txt")
    print("    4. Move the file to:")
    print(f"       {cookies_path()}")
    print()
    print("  Option B — yt-dlp command line:")
    print(f"    yt-dlp --cookies-from-browser {browser_label.lower()} --cookies \"{cookies_path()}\" --skip-download https://www.youtube.com")
    print()
    print("  After placing the file, run any transcriptex command.")
    print("  Cookies will be used automatically.")
    print()


# ---------------------------------------------------------------------------
# Interactive onboarding
# ---------------------------------------------------------------------------


def run_interactive_setup() -> Path:
    from transcriptex import __version__

    print()
    print("=" * 60)
    print("  transcriptex — YouTube Cookie Setup")
    print(f"  v{__version__}")
    print("=" * 60)
    print()
    print("  transcriptex needs YouTube cookies to avoid bot detection")
    print("  during bulk downloads. This is a one-time setup.")
    print()

    available = detect_available_browsers()

    if not available:
        print("  No supported browsers detected.")
        print()
        _print_manual_instructions("your browser")
        raise RuntimeError("No supported browsers found")

    print("  Detected browsers:")
    for i, browser in enumerate(available, 1):
        label = BROWSER_MAP.get(browser, browser)
        # Firefox is recommended because it doesn't use DPAPI
        marker = " (recommended — no DPAPI issues)" if browser == "firefox" else ""
        if not marker and browser == "chrome":
            marker = " (may require closing background processes)"
        print(f"    [{i}] {label}{marker}")
    print()

    while True:
        try:
            choice = input("  Select a browser [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Setup cancelled.")
            sys.exit(130)

        if not choice:
            choice = "1"

        try:
            idx = int(choice)
            if 1 <= idx <= len(available):
                selected = available[idx - 1]
                break
            else:
                print(f"  Please enter a number between 1 and {len(available)}.")
        except ValueError:
            name = choice.lower()
            if name in available:
                selected = name
                break
            matches = [b for b in available if b.startswith(name)]
            if len(matches) == 1:
                selected = matches[0]
                break
            print(f"  Unknown browser: {choice}")

    browser_label = BROWSER_MAP.get(selected, selected)

    # Firefox-based browsers don't need to be closed (SQLite read works while running)
    needs_close = selected not in _FIREFOX_BASED_PROFILES

    if needs_close:
        print()
        print(f"  IMPORTANT: Close all {BROWSER_MAP.get(selected, selected)} windows.")
        print("  Background processes will be killed automatically.")
        print()
        try:
            confirm = input("  Ready to continue? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Setup cancelled.")
            sys.exit(130)
        if confirm in ("n", "no"):
            print("  You can try again later. Run: transcriptex cookies")
            raise RuntimeError("User cancelled setup")
    else:
        print()
        print(f"  {browser_label} does not need to be closed for cookie extraction.")
        print()

    dest = extract_cookies_from_browser(selected)

    print()
    print("=" * 60)
    print("  Setup complete!")
    print("=" * 60)
    print()
    print(f"  Cookies file: {dest}")
    print("  All transcriptex commands will now use these cookies.")
    print()
    print("  To update cookies in the future, run: transcriptex cookies")
    print("  To use a different cookies file: transcriptex <cmd> --cookies path/to/file")
    print()

    return dest


# ---------------------------------------------------------------------------
# Cookie management
# ---------------------------------------------------------------------------


def remove_stored_cookies() -> bool:
    path = cookies_path()
    if path.exists():
        path.unlink()
        return True
    return False


def show_cookie_status() -> None:
    path = cookies_path()
    if path.exists():
        stat = path.stat()
        size_kb = stat.st_size / 1024
        from datetime import datetime

        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print()
        print("  Cookies are configured.")
        print(f"  File     : {path}")
        print(f"  Size     : {size_kb:.1f} KB")
        print(f"  Updated  : {modified}")
        print()
        print("  To refresh cookies, run: transcriptex cookies")
        print("  To remove cookies, run:  transcriptex cookies --remove")
        print()
    else:
        print()
        print("  No cookies configured.")
        print("  YouTube may block bulk downloads without authentication.")
        print()
        print("  To set up cookies, run: transcriptex cookies")
        print()
