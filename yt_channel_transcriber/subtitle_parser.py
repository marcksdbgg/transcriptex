from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET

TAG_RE = re.compile(r"<[^>]+>")
SRT_INDEX_RE = re.compile(r"^\d+$")
HEADER_RE = re.compile(r"^(WEBVTT|NOTE|STYLE|REGION)\b", re.IGNORECASE)


def _clean_line(line: str) -> str:
    value = line.replace("\u200b", " ")
    value = TAG_RE.sub("", value)
    value = html.unescape(value)
    value = " ".join(value.split())
    return value.strip()


def _dedupe_adjacent(lines: list[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        if not line:
            continue
        if result and result[-1] == line:
            continue
        result.append(line)
    return result


def parse_vtt_or_srt(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if HEADER_RE.match(line):
            continue
        if "-->" in line:
            continue
        if SRT_INDEX_RE.fullmatch(line):
            continue
        clean = _clean_line(line)
        if clean:
            lines.append(clean)
    return "\n".join(_dedupe_adjacent(lines))


def parse_json3(text: str) -> str:
    payload = json.loads(text)
    events = payload.get("events", [])
    lines: list[str] = []
    for event in events:
        segments = event.get("segs") or []
        if not isinstance(segments, list):
            continue
        merged = "".join(str(seg.get("utf8", "")) for seg in segments if isinstance(seg, dict))
        if not merged:
            continue
        for chunk in merged.splitlines():
            clean = _clean_line(chunk)
            if clean:
                lines.append(clean)
    return "\n".join(_dedupe_adjacent(lines))


def parse_xml_timedtext(text: str) -> str:
    root = ET.fromstring(text)
    lines: list[str] = []
    for node in root.iter():
        tag = node.tag if isinstance(node.tag, str) else ""
        local_tag = tag.rsplit("}", 1)[-1]
        if local_tag not in {"p", "text"}:
            continue
        merged = "".join(node.itertext())
        clean = _clean_line(merged)
        if clean:
            lines.append(clean)
    return "\n".join(_dedupe_adjacent(lines))


def parse_subtitle_bytes(data: bytes, ext_hint: str = "") -> str:
    text = data.decode("utf-8-sig", errors="replace")
    ext = (ext_hint or "").lower().strip(".")

    try:
        if ext in {"json", "json3"}:
            return parse_json3(text)
        if ext in {"ttml", "xml", "srv1", "srv2", "srv3"}:
            return parse_xml_timedtext(text)
        if ext in {"srt", "vtt", "sbv"}:
            return parse_vtt_or_srt(text)
    except Exception:
        # Fallback parser for malformed subtitle payloads.
        return parse_vtt_or_srt(text)

    if text.lstrip().startswith("{"):
        try:
            return parse_json3(text)
        except Exception:
            return parse_vtt_or_srt(text)
    if text.lstrip().startswith("<"):
        try:
            return parse_xml_timedtext(text)
        except Exception:
            return parse_vtt_or_srt(text)
    return parse_vtt_or_srt(text)
