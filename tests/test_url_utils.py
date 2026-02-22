import unittest

from yt_channel_transcriber.url_utils import extract_video_id, normalize_channel_url


class UrlUtilsTests(unittest.TestCase):
    def test_normalize_handle_channel(self) -> None:
        self.assertEqual(
            normalize_channel_url("https://www.youtube.com/@openai/videos"),
            "https://www.youtube.com/@openai",
        )

    def test_normalize_channel_id(self) -> None:
        self.assertEqual(
            normalize_channel_url("https://www.youtube.com/channel/UC1234567890abcdef/videos"),
            "https://www.youtube.com/channel/UC1234567890abcdef",
        )

    def test_extract_video_id_from_watch_url(self) -> None:
        self.assertEqual(
            extract_video_id("https://www.youtube.com/watch?v=abc123XYZ_9"),
            "abc123XYZ_9",
        )

    def test_extract_video_id_from_shorts_url(self) -> None:
        self.assertEqual(
            extract_video_id("https://www.youtube.com/shorts/abc123XYZ_9"),
            "abc123XYZ_9",
        )

    def test_extract_video_id_from_relative_shorts_url(self) -> None:
        self.assertEqual(
            extract_video_id("/shorts/abc123XYZ_9"),
            "abc123XYZ_9",
        )


if __name__ == "__main__":
    unittest.main()
