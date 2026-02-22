import unittest

from transcriptex.fs_utils import build_transcript_filename, extract_video_id_from_stem


class FsUtilsTests(unittest.TestCase):
    def test_build_transcript_filename_contains_id(self) -> None:
        filename = build_transcript_filename("abc123XYZ_9", "Titulo de prueba")
        self.assertTrue(filename.endswith("[abc123XYZ_9].txt"))

    def test_extract_video_id_from_legacy_name(self) -> None:
        self.assertEqual(extract_video_id_from_stem("abc123XYZ_9"), "abc123XYZ_9")

    def test_extract_video_id_from_title_name(self) -> None:
        stem = "Mi video favorito [abc123XYZ_9]"
        self.assertEqual(extract_video_id_from_stem(stem), "abc123XYZ_9")


if __name__ == "__main__":
    unittest.main()
