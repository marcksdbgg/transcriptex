"""Timed subtitle decoding for single-video exports (no speech model needed)."""
from dataclasses import dataclass
import json
import math
import re
import xml.etree.ElementTree as ET

from transcriptex.subtitle_parser import _clean_line


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str


def _seconds(value):
    value = str(value).strip().replace(',', '.')
    if value.endswith('ms'):
        return float(value[:-2]) / 1000
    if value.endswith('s'):
        return float(value[:-1])
    parts = value.split(':')
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))


def parse_subtitle_segments(data: bytes, ext_hint: str = '') -> list[SubtitleSegment]:
    """Reject malformed timings instead of inventing timestamps for SRT/JSON."""
    text = data.decode('utf-8-sig')
    ext = ext_hint.lower().lstrip('.')
    result = []

    def add(start, end, content):
        clean = _clean_line(content)
        if clean and math.isfinite(start) and math.isfinite(end) and 0 <= start < end:
            # Only collapse identical overlapping cues, preserving intentional repetition.
            if result and result[-1].text == clean and start < result[-1].end:
                result[-1].end = max(end, result[-1].end)
            else:
                result.append(SubtitleSegment(start, end, clean))

    if ext in ('json', 'json3') or text.lstrip().startswith('{'):
        for event in json.loads(text).get('events', []):
            if 'tStartMs' not in event:
                continue
            content = ''.join(str(s.get('utf8', '')) for s in event.get('segs', []) if isinstance(s, dict))
            start = float(event['tStartMs']) / 1000
            add(start, start + float(event.get('dDurationMs', 0)) / 1000, content)
    elif ext in ('xml', 'ttml', 'srv1', 'srv2', 'srv3') or text.lstrip().startswith('<'):
        for node in ET.fromstring(text).iter():
            if node.tag.rsplit('}', 1)[-1] not in ('p', 'text'):
                continue
            attrs = node.attrib
            if 't' in attrs:
                start = float(attrs['t']) / 1000
                end = start + float(attrs.get('d', 0)) / 1000
            else:
                start = _seconds(attrs.get('begin', attrs.get('start', '0')))
                end = _seconds(attrs['end']) if 'end' in attrs else start + _seconds(attrs.get('dur', '0'))
            add(start, end, ' '.join(node.itertext()))
    else:
        cue = re.compile(r'(\d{1,2}:\d{2}(?::\d{2})?[.,]\d+)\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d+)')
        lines = text.splitlines()
        index = 0
        while index < len(lines):
            match = cue.search(lines[index])
            index += 1
            if not match:
                continue
            content = []
            while index < len(lines) and lines[index].strip():
                if cue.search(lines[index]):
                    break
                content.append(lines[index])
                index += 1
            add(_seconds(match[1]), _seconds(match[2]), ' '.join(content))
    if not result:
        raise ValueError('No usable timed subtitle cues found')
    return result
