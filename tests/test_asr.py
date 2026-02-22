"""
test_asr.py — Unit tests for yt_channel_transcriber.asr module.

All tests use mocks for faster-whisper and yt-dlp to avoid real GPU/network calls.
Pattern follows existing tests (tempfile.TemporaryDirectory, csv.DictWriter, unittest).
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from yt_channel_transcriber.asr import (
    AsrConfig,
    _write_asr_transcript,
    run_asr,
)
from yt_channel_transcriber.cli import build_parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VIDEO_INDEX_HEADERS = [
    "video_id", "title", "url",
    "assigned_playlist_id", "assigned_playlist_title",
    "all_playlist_ids", "all_playlist_titles",
    "folder", "transcript_path", "status", "language", "source", "error",
]


def _make_row(
    video_id: str,
    title: str,
    status: str,
    folder: str = "Sin_Playlist",
    transcript_path: str = "",
    url: str = "",
) -> dict[str, str]:
    return {
        "video_id": video_id,
        "title": title,
        "url": url or f"https://youtube.com/watch?v={video_id}",
        "assigned_playlist_id": "",
        "assigned_playlist_title": "",
        "all_playlist_ids": "",
        "all_playlist_titles": "",
        "folder": folder,
        "transcript_path": transcript_path,
        "status": status,
        "language": "",
        "source": "",
        "error": "No subtitle tracks found" if status == "no_subtitles" else "",
    }


def _write_index(meta_dir: Path, rows: list[dict[str, str]]) -> None:
    with (meta_dir / "video_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VIDEO_INDEX_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_index(meta_dir: Path) -> list[dict[str, str]]:
    with (meta_dir / "video_index.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _make_mock_model(text: str = "Texto transcrito de prueba.", lang: str = "es") -> MagicMock:
    """Return a mock faster-whisper WhisperModel that yields one segment."""
    segment = MagicMock()
    segment.text = text

    info = MagicMock()
    info.language = lang

    model = MagicMock()
    model.transcribe.return_value = (iter([segment]), info)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class AsrWriteTranscriptTests(unittest.TestCase):
    """Tests for _write_asr_transcript — the file format must match transcriber.py."""

    def test_transcript_format_matches_transcriber(self) -> None:
        """Verify all required header fields are present in the correct order."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test [abc123XYZ_9].txt"
            _write_asr_transcript(
                path=path,
                title="Mi Video de Prueba",
                video_id="abc123XYZ_9",
                url="https://youtube.com/watch?v=abc123XYZ_9",
                channel="Canal de Prueba",
                language="es",
                transcript_text="Hola mundo. Esto es una prueba.",
            )

            content = path.read_text(encoding="utf-8")
            lines = content.splitlines()

            self.assertTrue(lines[0].startswith("Title: "), f"Line 0: {lines[0]!r}")
            self.assertTrue(lines[1].startswith("Video ID: "), f"Line 1: {lines[1]!r}")
            self.assertTrue(lines[2].startswith("Video URL: "), f"Line 2: {lines[2]!r}")
            self.assertTrue(lines[3].startswith("Channel: "), f"Line 3: {lines[3]!r}")
            self.assertTrue(lines[4].startswith("Subtitle language: "), f"Line 4: {lines[4]!r}")
            self.assertIn("Subtitle source: asr", content)
            self.assertIn("Generated at:", content)
            self.assertIn("-----", content)
            self.assertIn("Hola mundo. Esto es una prueba.", content)

    def test_transcript_source_is_asr(self) -> None:
        """Source field must always be 'asr', not 'manual' or 'auto'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "out.txt"
            _write_asr_transcript(
                path=path,
                title="Test",
                video_id="abc123XYZ_9",
                url="https://youtube.com/watch?v=abc123XYZ_9",
                channel="",
                language="es",
                transcript_text="Contenido.",
            )
            content = path.read_text(encoding="utf-8")
            self.assertIn("Subtitle source: asr", content)
            self.assertNotIn("Subtitle source: manual", content)
            self.assertNotIn("Subtitle source: auto", content)


class AsrRunTests(unittest.TestCase):
    """Integration-style tests for run_asr using mocked download + transcription."""

    def _prepare_channel(
        self,
        root: Path,
        rows: list[dict[str, str]],
    ) -> Path:
        channel_dir = root / "My Channel"
        meta_dir = channel_dir / "_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (channel_dir / "Sin_Playlist").mkdir(parents=True, exist_ok=True)
        _write_index(meta_dir, rows)
        return channel_dir

    @patch("yt_channel_transcriber.asr._load_model")
    @patch("yt_channel_transcriber.asr._download_audio")
    def test_asr_writes_transcript_and_updates_index(
        self, mock_download: MagicMock, mock_load_model: MagicMock
    ) -> None:
        """Happy path: transcript file is created and video_index.csv updated to ok/asr."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_make_row("abc123XYZ_9", "Video Sin Subtitulos", "no_subtitles")]
            channel_dir = self._prepare_channel(Path(tmp_dir), rows)

            # Mock audio file
            fake_audio = Path(tmp_dir) / "audio.wav"
            fake_audio.write_bytes(b"RIFF")
            mock_download.return_value = fake_audio

            # Mock model
            mock_model = _make_mock_model("Texto transcrito de prueba.", "es")
            mock_load_model.return_value = mock_model

            config = AsrConfig(channel_dir=channel_dir, device="cpu", compute_type="int8")
            report = run_asr(config)

            self.assertEqual(report["ok"], 1)
            self.assertEqual(report["errors"], 0)

            # Transcript file must exist
            txt_files = list(channel_dir.rglob("*.txt"))
            txt_files = [f for f in txt_files if "_meta" not in f.parts]
            self.assertEqual(len(txt_files), 1, f"Expected 1 .txt file, found: {txt_files}")

            content = txt_files[0].read_text(encoding="utf-8")
            self.assertIn("Subtitle source: asr", content)
            self.assertIn("Texto transcrito de prueba.", content)

            # video_index.csv must be updated
            updated_rows = _read_index(channel_dir / "_meta")
            row = next(r for r in updated_rows if r["video_id"] == "abc123XYZ_9")
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["source"], "asr")
            self.assertEqual(row["language"], "es")
            self.assertEqual(row["error"], "")
            self.assertNotEqual(row["transcript_path"], "")

    @patch("yt_channel_transcriber.asr._load_model")
    @patch("yt_channel_transcriber.asr._download_audio")
    def test_asr_skips_existing_transcript_without_force(
        self, mock_download: MagicMock, mock_load_model: MagicMock
    ) -> None:
        """Without --force, existing transcripts must not be re-downloaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_make_row("abc123XYZ_9", "Video Con Transcript", "no_subtitles")]
            channel_dir = self._prepare_channel(Path(tmp_dir), rows)

            # Pre-create transcript file
            existing = channel_dir / "Sin_Playlist" / "Video Con Transcript [abc123XYZ_9].txt"
            existing.write_text("Title: Video Con Transcript\n\n-----\n\nYa existe.\n", encoding="utf-8")

            mock_load_model.return_value = _make_mock_model()

            config = AsrConfig(channel_dir=channel_dir, device="cpu", compute_type="int8", force=False)
            report = run_asr(config)

            # Download must NOT have been called
            mock_download.assert_not_called()
            self.assertEqual(report["skipped"], 1)
            self.assertEqual(report["ok"], 0)

    @patch("yt_channel_transcriber.asr._load_model")
    @patch("yt_channel_transcriber.asr._download_audio")
    def test_asr_force_overwrites_existing(
        self, mock_download: MagicMock, mock_load_model: MagicMock
    ) -> None:
        """With --force, existing transcripts must be re-transcribed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_make_row("abc123XYZ_9", "Video Forzado", "no_subtitles")]
            channel_dir = self._prepare_channel(Path(tmp_dir), rows)

            existing = channel_dir / "Sin_Playlist" / "Video Forzado [abc123XYZ_9].txt"
            existing.write_text("Title: Video Forzado\n\n-----\n\nViejo.\n", encoding="utf-8")

            fake_audio = Path(tmp_dir) / "audio.wav"
            fake_audio.write_bytes(b"RIFF")
            mock_download.return_value = fake_audio
            mock_load_model.return_value = _make_mock_model("Nuevo contenido.", "es")

            config = AsrConfig(channel_dir=channel_dir, device="cpu", compute_type="int8", force=True)
            report = run_asr(config)

            mock_download.assert_called_once()
            self.assertEqual(report["ok"], 1)

            content = existing.read_text(encoding="utf-8")
            self.assertIn("Nuevo contenido.", content)

    @patch("yt_channel_transcriber.asr._load_model")
    @patch("yt_channel_transcriber.asr._download_audio")
    def test_asr_skips_non_eligible_statuses(
        self, mock_download: MagicMock, mock_load_model: MagicMock
    ) -> None:
        """Videos with status 'ok' or 'existing' must not be processed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _make_row("abc123XYZ_9", "Video OK", "ok"),
                _make_row("def456UVW_8", "Video Existing", "existing"),
                _make_row("ghi789RST_7", "Video Sin Subtitulos", "no_subtitles"),
            ]
            channel_dir = self._prepare_channel(Path(tmp_dir), rows)

            fake_audio = Path(tmp_dir) / "audio.wav"
            fake_audio.write_bytes(b"RIFF")
            mock_download.return_value = fake_audio
            mock_load_model.return_value = _make_mock_model()

            config = AsrConfig(channel_dir=channel_dir, device="cpu", compute_type="int8")
            report = run_asr(config)

            # Only 1 eligible video (no_subtitles)
            self.assertEqual(report["total_processed"], 1)
            self.assertEqual(mock_download.call_count, 1)

    @patch("yt_channel_transcriber.asr._load_model")
    @patch("yt_channel_transcriber.asr._download_audio")
    def test_asr_limit_respected(
        self, mock_download: MagicMock, mock_load_model: MagicMock
    ) -> None:
        """--limit N must process at most N videos."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _make_row(f"vid{i:011d}", f"Video {i}", "no_subtitles")
                for i in range(5)
            ]
            channel_dir = self._prepare_channel(Path(tmp_dir), rows)

            fake_audio = Path(tmp_dir) / "audio.wav"
            fake_audio.write_bytes(b"RIFF")
            mock_download.return_value = fake_audio
            mock_load_model.return_value = _make_mock_model()

            config = AsrConfig(channel_dir=channel_dir, device="cpu", compute_type="int8", limit=2)
            report = run_asr(config)

            self.assertEqual(report["total_processed"], 2)
            self.assertEqual(mock_download.call_count, 2)

    @patch("yt_channel_transcriber.asr._load_model")
    @patch("yt_channel_transcriber.asr._download_audio")
    def test_asr_include_failed_processes_failed_videos(
        self, mock_download: MagicMock, mock_load_model: MagicMock
    ) -> None:
        """--include-failed must also process videos with status='failed'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [
                _make_row("abc123XYZ_9", "Video Failed", "failed"),
                _make_row("def456UVW_8", "Video No Subtitles", "no_subtitles"),
            ]
            channel_dir = self._prepare_channel(Path(tmp_dir), rows)

            fake_audio = Path(tmp_dir) / "audio.wav"
            fake_audio.write_bytes(b"RIFF")
            mock_download.return_value = fake_audio
            mock_load_model.return_value = _make_mock_model()

            config = AsrConfig(
                channel_dir=channel_dir, device="cpu", compute_type="int8", include_failed=True
            )
            report = run_asr(config)

            self.assertEqual(report["total_processed"], 2)
            self.assertEqual(mock_download.call_count, 2)

    @patch("yt_channel_transcriber.asr._load_model")
    @patch("yt_channel_transcriber.asr._download_audio")
    def test_asr_download_error_recorded_in_report(
        self, mock_download: MagicMock, mock_load_model: MagicMock
    ) -> None:
        """If download fails, the video is counted as an error, not a crash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rows = [_make_row("abc123XYZ_9", "Video Privado", "no_subtitles")]
            channel_dir = self._prepare_channel(Path(tmp_dir), rows)

            mock_download.side_effect = RuntimeError("Video unavailable")
            mock_load_model.return_value = _make_mock_model()

            config = AsrConfig(channel_dir=channel_dir, device="cpu", compute_type="int8")
            report = run_asr(config)

            self.assertEqual(report["errors"], 1)
            self.assertEqual(report["ok"], 0)

            # No transcript file should have been created
            txt_files = [
                f for f in channel_dir.rglob("*.txt")
                if "_meta" not in f.parts
            ]
            self.assertEqual(len(txt_files), 0)


class AsrCliTests(unittest.TestCase):
    """Tests for CLI integration of the asr subcommand."""

    def test_cli_asr_command_registered(self) -> None:
        """build_parser() must accept 'asr' as a valid subcommand."""
        parser = build_parser()
        # Should not raise
        args = parser.parse_args(["asr", "output/My Channel"])
        self.assertEqual(args.command, "asr")
        self.assertEqual(args.channel_dir, "output/My Channel")

    def test_cli_asr_defaults(self) -> None:
        """Verify default values for all asr arguments."""
        parser = build_parser()
        args = parser.parse_args(["asr", "output/My Channel"])
        self.assertEqual(args.model, "large-v3-turbo")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.compute_type, "float16")
        self.assertEqual(args.language, "es")
        self.assertEqual(args.beam_size, 5)
        self.assertEqual(args.batch_size, 8)
        self.assertFalse(args.force)
        self.assertEqual(args.limit, 0)
        self.assertEqual(args.sleep, 0.0)
        self.assertFalse(args.include_failed)
        self.assertFalse(args.verbose)

    def test_cli_asr_custom_args(self) -> None:
        """Verify custom argument parsing for asr subcommand."""
        parser = build_parser()
        args = parser.parse_args([
            "asr", "output/My Channel",
            "--model", "medium",
            "--device", "cpu",
            "--compute-type", "int8",
            "--language", "en",
            "--beam-size", "3",
            "--limit", "10",
            "--force",
            "--include-failed",
            "--verbose",
        ])
        self.assertEqual(args.model, "medium")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.compute_type, "int8")
        self.assertEqual(args.language, "en")
        self.assertEqual(args.beam_size, 3)
        self.assertEqual(args.limit, 10)
        self.assertTrue(args.force)
        self.assertTrue(args.include_failed)
        self.assertTrue(args.verbose)


if __name__ == "__main__":
    unittest.main()
