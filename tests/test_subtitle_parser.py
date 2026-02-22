import unittest

from yt_channel_transcriber.subtitle_parser import (
    parse_json3,
    parse_subtitle_bytes,
    parse_vtt_or_srt,
    parse_xml_timedtext,
)


class SubtitleParserTests(unittest.TestCase):
    def test_parse_vtt(self) -> None:
        payload = """WEBVTT

00:00:00.000 --> 00:00:01.000
Hola

00:00:01.000 --> 00:00:02.000
Mundo
"""
        self.assertEqual(parse_vtt_or_srt(payload), "Hola\nMundo")

    def test_parse_json3(self) -> None:
        payload = """
{
  "events": [
    {"segs": [{"utf8": "Linea 1"}]},
    {"segs": [{"utf8": "Linea 2"}]}
  ]
}
"""
        self.assertEqual(parse_json3(payload), "Linea 1\nLinea 2")

    def test_parse_xml_timedtext(self) -> None:
        payload = """
<transcript>
  <text start="0.0" dur="1.0">Alpha</text>
  <text start="1.0" dur="1.0">Beta</text>
</transcript>
"""
        self.assertEqual(parse_xml_timedtext(payload), "Alpha\nBeta")

    def test_parse_subtitle_bytes_fallback(self) -> None:
        payload = b"1\n00:00:00,000 --> 00:00:01,000\nHola\n"
        self.assertEqual(parse_subtitle_bytes(payload, ext_hint="srt"), "Hola")


if __name__ == "__main__":
    unittest.main()
