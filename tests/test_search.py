import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from transcriptex.search import SearchConfig, search_channel_output


class SearchTests(unittest.TestCase):
    def test_search_by_title_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            channel_dir = self._prepare_fixture(Path(tmp_dir))
            report = search_channel_output(
                SearchConfig(
                    channel_dir=channel_dir,
                    query="fiesta del chivo",
                    fields=["title"],
                    statuses=[],
                    top=5,
                    print_results=False,
                )
            )
            self.assertEqual(report["total_matches"], 1)

    def test_search_by_transcript_keywords_match_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            channel_dir = self._prepare_fixture(Path(tmp_dir))
            report = search_channel_output(
                SearchConfig(
                    channel_dir=channel_dir,
                    query="trujillo urania",
                    fields=["transcript"],
                    statuses=["ok"],
                    match_all_tokens=True,
                    top=5,
                    print_results=False,
                )
            )
            self.assertEqual(report["total_matches"], 1)

    def test_export_top_creates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            channel_dir = self._prepare_fixture(Path(tmp_dir))
            export_dir = channel_dir / "_exports" / "search_manual"
            report = search_channel_output(
                SearchConfig(
                    channel_dir=channel_dir,
                    query="quijote",
                    fields=["title", "transcript"],
                    statuses=["ok"],
                    top=5,
                    print_results=False,
                    export_top=1,
                    export_dir=export_dir,
                )
            )
            self.assertEqual(report["exported_count"], 1)
            exported = list(export_dir.glob("*.txt"))
            self.assertEqual(len(exported), 1)
            self.assertTrue((export_dir / "export_manifest.csv").exists())
            self.assertTrue((export_dir / "export_manifest.json").exists())

    def test_search_unicode_token_with_accent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            channel_dir = self._prepare_fixture(Path(tmp_dir))
            report = search_channel_output(
                SearchConfig(
                    channel_dir=channel_dir,
                    query="filosofía",
                    fields=["transcript"],
                    statuses=["ok"],
                    top=5,
                    print_results=False,
                )
            )
            self.assertEqual(report["total_matches"], 1)

    def test_search_print_summary_false_silent(self) -> None:
        """print_summary=False must silence the human summary table
        even when print_results=True."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            channel_dir = self._prepare_fixture(Path(tmp_dir))
            buf = io.StringIO()
            with redirect_stdout(buf):
                report = search_channel_output(
                    SearchConfig(
                        channel_dir=channel_dir,
                        query="quijote",
                        fields=["title"],
                        statuses=[],
                        top=5,
                        print_results=True,
                        print_summary=False,
                    )
                )
            output = buf.getvalue()
            self.assertNotIn("Search summary", output)
            self.assertEqual(report["total_matches"], 1)

    def _prepare_fixture(self, root: Path) -> Path:
        channel_dir = root / "My Channel"
        meta_dir = channel_dir / "_meta"
        playlist_dir = channel_dir / "Playlist A [PL123]"
        playlist_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        transcript_1 = playlist_dir / "La fiesta del chivo [abc123XYZ_9].txt"
        transcript_2 = playlist_dir / "Cervantes y Quijote [def456UVW_8].txt"
        transcript_1.write_text(
            "Title: La fiesta del chivo\n\n-----\n\nTrujillo, Urania, dictadura y filosofía política.\n",
            encoding="utf-8",
        )
        transcript_2.write_text(
            "Title: Cervantes y Quijote\n\n-----\n\nDon Quijote caballero andante.\n",
            encoding="utf-8",
        )

        rows = [
            {
                "video_id": "abc123XYZ_9",
                "title": "La fiesta del chivo y Trujillo",
                "url": "https://youtube.com/watch?v=abc123XYZ_9",
                "assigned_playlist_id": "PL123",
                "assigned_playlist_title": "Playlist A",
                "all_playlist_ids": "PL123",
                "all_playlist_titles": "Playlist A",
                "folder": "Playlist A [PL123]",
                "transcript_path": "Playlist A [PL123]/La fiesta del chivo [abc123XYZ_9].txt",
                "status": "ok",
                "language": "es",
                "source": "manual",
                "error": "",
            },
            {
                "video_id": "def456UVW_8",
                "title": "Cervantes y Quijote",
                "url": "https://youtube.com/watch?v=def456UVW_8",
                "assigned_playlist_id": "PL123",
                "assigned_playlist_title": "Playlist A",
                "all_playlist_ids": "PL123",
                "all_playlist_titles": "Playlist A",
                "folder": "Playlist A [PL123]",
                "transcript_path": "Playlist A [PL123]/Cervantes y Quijote [def456UVW_8].txt",
                "status": "ok",
                "language": "es",
                "source": "manual",
                "error": "",
            },
            {
                "video_id": "ghi789RST_7",
                "title": "Video sin subtitulos",
                "url": "https://youtube.com/watch?v=ghi789RST_7",
                "assigned_playlist_id": "",
                "assigned_playlist_title": "",
                "all_playlist_ids": "",
                "all_playlist_titles": "",
                "folder": "Sin_Playlist",
                "transcript_path": "",
                "status": "no_subtitles",
                "language": "",
                "source": "",
                "error": "No subtitle tracks found",
            },
        ]

        with (meta_dir / "video_index.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return channel_dir


if __name__ == "__main__":
    unittest.main()
