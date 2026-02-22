import csv
import tempfile
import unittest
from pathlib import Path

from transcriptex.audit import AuditConfig, audit_channel_output


class AuditTests(unittest.TestCase):
    def test_audit_counts_missing_and_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            channel_dir = Path(tmp_dir) / "My Channel"
            meta_dir = channel_dir / "_meta"
            playlist_dir = channel_dir / "Playlist A [PL123]"
            meta_dir.mkdir(parents=True, exist_ok=True)
            playlist_dir.mkdir(parents=True, exist_ok=True)

            rows = [
                {
                    "video_id": "abc123XYZ_9",
                    "title": "Video ok",
                    "url": "https://youtube.com/watch?v=abc123XYZ_9",
                    "assigned_playlist_id": "PL123",
                    "assigned_playlist_title": "Playlist A",
                    "all_playlist_ids": "PL123",
                    "all_playlist_titles": "Playlist A",
                    "folder": "Playlist A [PL123]",
                    "transcript_path": "Playlist A [PL123]/Video ok [abc123XYZ_9].txt",
                    "status": "ok",
                    "language": "es",
                    "source": "manual",
                    "error": "",
                },
                {
                    "video_id": "def456UVW_8",
                    "title": "Video missing",
                    "url": "https://youtube.com/watch?v=def456UVW_8",
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

            (playlist_dir / "Video ok [abc123XYZ_9].txt").write_text("ok", encoding="utf-8")
            (playlist_dir / "Orphan [ghi789RST_7].txt").write_text("orphan", encoding="utf-8")

            report = audit_channel_output(
                AuditConfig(
                    channel_dir=channel_dir,
                    write_files=False,
                    max_rows_in_console=1,
                )
            )

            self.assertEqual(report["total_indexed_videos"], 2)
            self.assertEqual(report["transcript_ids_found"], 2)
            self.assertEqual(report["indexed_without_transcript_count"], 1)
            self.assertEqual(report["orphan_transcripts_count"], 1)

    def test_audit_can_rename_legacy_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            channel_dir = Path(tmp_dir) / "My Channel"
            meta_dir = channel_dir / "_meta"
            channel_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)

            rows = [
                {
                    "video_id": "abc123XYZ_9",
                    "title": "Video con titulo",
                    "url": "https://youtube.com/watch?v=abc123XYZ_9",
                    "assigned_playlist_id": "",
                    "assigned_playlist_title": "",
                    "all_playlist_ids": "",
                    "all_playlist_titles": "",
                    "folder": "Sin_Playlist",
                    "transcript_path": "Sin_Playlist/abc123XYZ_9.txt",
                    "status": "ok",
                    "language": "es",
                    "source": "manual",
                    "error": "",
                }
            ]

            with (meta_dir / "video_index.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerow(rows[0])

            legacy = channel_dir / "abc123XYZ_9.txt"
            legacy.write_text("ok", encoding="utf-8")

            report = audit_channel_output(
                AuditConfig(
                    channel_dir=channel_dir,
                    write_files=False,
                    max_rows_in_console=1,
                    rename_legacy_files=True,
                )
            )

            self.assertEqual(report["renamed_legacy_files_count"], 1)
            self.assertFalse(legacy.exists())
            renamed_matches = [p for p in channel_dir.glob("*.txt") if "[abc123XYZ_9]" in p.name]
            self.assertTrue(renamed_matches)


if __name__ == "__main__":
    unittest.main()
