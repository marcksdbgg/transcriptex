"""
Unit tests for the `scan` command and the `extract_uploads_playlist_videos`
helper in `youtube_client`.
"""
import csv
import io
import json
import tempfile
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from transcriptex import scan as scan_module
from transcriptex.cli import KNOWN_SUBCOMMANDS, build_parser
from transcriptex.models import ChannelInfo, PlaylistRecord, VideoRecord
from transcriptex.scan import (
    PAGE_SIZE,
    ScanConfig,
    ScanEntry,
    is_transcribed_status,
    scan_channel,
)
from transcriptex.youtube_client import YouTubeClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHANNEL_ID = "UC" + "abcdefghijk"  # 13 chars total
UPLOADS_ID = "UU" + "abcdefghijk"


def _make_record(video_id: str, title: str) -> VideoRecord:
    return VideoRecord(
        video_id=video_id,
        title=title,
        url=f"https://www.youtube.com/watch?v={video_id}",
        channel="Test Channel",
        channel_id=CHANNEL_ID,
    )


def _make_playlist(playlist_id: str, title: str, video_ids: list[str]) -> PlaylistRecord:
    pl = PlaylistRecord(
        playlist_id=playlist_id,
        title=title,
        url=f"https://www.youtube.com/playlist?list={playlist_id}",
    )
    pl.videos = [_make_record(vid, f"Title {vid}") for vid in video_ids]
    return pl


def _build_mock_client(
    *,
    channel_info: ChannelInfo | None = None,
    uploads: list[VideoRecord] | None = None,
    playlists: list[PlaylistRecord] | None = None,
    playlist_videos_by_id: dict[str, list[VideoRecord]] | None = None,
) -> mock.MagicMock:
    """Create a mock YouTubeClient that returns deterministic data."""
    if channel_info is None:
        channel_info = ChannelInfo(
            base_url="https://www.youtube.com/@test",
            title="Test Channel",
            channel_id=CHANNEL_ID,
            handle="@test",
        )
    client = mock.MagicMock(spec=YouTubeClient)
    client.discover_channel.return_value = channel_info
    client.extract_uploads_playlist_videos.return_value = uploads or []
    client.extract_playlists.return_value = playlists or []
    # Default per-playlist lookup: empty
    lookup: dict[str, list[VideoRecord]] = playlist_videos_by_id or {}
    client.extract_playlist_videos.side_effect = (
        lambda playlist_url, fallback_channel="": lookup.get(
            # Last path segment is the playlist ID.
            playlist_url.rsplit("list=", 1)[-1],
            [],
        )
    )
    return client


# ---------------------------------------------------------------------------
# YouTubeClient: extract_uploads_playlist_videos
# ---------------------------------------------------------------------------

class UploadsPlaylistConversionTests(unittest.TestCase):
    def test_channel_id_to_uploads_playlist_id(self) -> None:
        self.assertEqual(
            YouTubeClient.channel_id_to_uploads_playlist_id(CHANNEL_ID),
            UPLOADS_ID,
        )

    def test_channel_id_to_uploads_playlist_id_strips_whitespace(self) -> None:
        self.assertEqual(
            YouTubeClient.channel_id_to_uploads_playlist_id(f"  {CHANNEL_ID}  "),
            UPLOADS_ID,
        )

    def test_channel_id_to_uploads_playlist_id_invalid(self) -> None:
        with self.assertRaises(ValueError):
            YouTubeClient.channel_id_to_uploads_playlist_id("")
        with self.assertRaises(ValueError):
            YouTubeClient.channel_id_to_uploads_playlist_id("UUabc")  # wrong prefix
        with self.assertRaises(ValueError):
            YouTubeClient.channel_id_to_uploads_playlist_id("UC")  # too short

    def test_extract_uploads_playlist_videos_returns_empty_for_invalid_id(self) -> None:
        client = YouTubeClient()
        # No network call should be made for invalid IDs
        self.assertEqual(client.extract_uploads_playlist_videos(""), [])
        self.assertEqual(client.extract_uploads_playlist_videos("not-a-uc-id"), [])

    def test_extract_uploads_playlist_videos_delegates_to_extract_playlist(self) -> None:
        client = YouTubeClient()
        expected = [_make_record("abc123XYZ_9", "Test Video")]
        with mock.patch.object(
            client, "extract_playlist_videos", return_value=expected
        ) as mock_extract:
            result = client.extract_uploads_playlist_videos(CHANNEL_ID, fallback_channel="Fallback")
        self.assertEqual(result, expected)
        mock_extract.assert_called_once_with(
            f"https://www.youtube.com/playlist?list={UPLOADS_ID}",
            fallback_channel="Fallback",
        )


# ---------------------------------------------------------------------------
# YouTubeClient: cookies are forwarded to yt-dlp
# ---------------------------------------------------------------------------

class CookiesOptPropagationTests(unittest.TestCase):
    def test_cookies_stored_on_client(self) -> None:
        client = YouTubeClient(cookies="/tmp/cookies.txt")
        self.assertEqual(client.cookies, "/tmp/cookies.txt")

    def test_cookies_stripped(self) -> None:
        client = YouTubeClient(cookies="  /tmp/cookies.txt  ")
        self.assertEqual(client.cookies, "/tmp/cookies.txt")

    def test_base_opts_includes_cookiefile_when_set(self) -> None:
        client = YouTubeClient(cookies="/tmp/cookies.txt")
        opts = client._base_opts()
        self.assertEqual(opts.get("cookiefile"), "/tmp/cookies.txt")

    def test_base_opts_omits_cookiefile_when_empty(self) -> None:
        client = YouTubeClient()
        opts = client._base_opts()
        self.assertNotIn("cookiefile", opts)


# ---------------------------------------------------------------------------
# scan_channel: directory mode
# ---------------------------------------------------------------------------

class ScanChannelDirModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.channel_dir = Path(self._tmp.name) / "Test Channel"
        self.meta_dir = self.channel_dir / "_meta"
        self.playlist_dir = self.channel_dir / "Playlist A [PL123]"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.playlist_dir.mkdir(parents=True, exist_ok=True)

    def _write_rows(self, rows: list[dict[str, str]]) -> None:
        path = self.meta_dir / "video_index.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_scan_dir_writes_status_json(self) -> None:
        rows = [
            {
                "video_id": "abc123XYZ_9",
                "title": "Video ok",
                "url": "https://youtube.com/watch?v=abc123XYZ_9",
                "assigned_playlist_id": "PL123",
                "assigned_playlist_title": "Playlist A",
                "all_playlist_ids": "PL123|PL456",
                "all_playlist_titles": "Playlist A|Playlist B",
                "folder": "Playlist A [PL123]",
                "transcript_path": "Playlist A [PL123]/Video ok [abc123XYZ_9].txt",
                "status": "ok",
                "language": "es",
                "source": "manual",
                "error": "",
            },
            {
                "video_id": "def456UVW_8",
                "title": "Video pending",
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
        self._write_rows(rows)
        (self.playlist_dir / "Video ok [abc123XYZ_9].txt").write_text("ok", encoding="utf-8")

        config = ScanConfig(
            channel_url_or_dir=str(self.channel_dir),
            no_interactive=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            report = scan_channel(config)

        # Verify return value
        self.assertEqual(report["total_videos"], 2)
        self.assertEqual(report["transcribed_count"], 1)
        self.assertEqual(report["pending_count"], 1)
        self.assertEqual(report["status_breakdown"]["ok"], 1)
        self.assertEqual(report["status_breakdown"]["no_subtitles"], 1)

        # Verify channel_status.json file
        status_path = self.meta_dir / "channel_status.json"
        self.assertTrue(status_path.exists())
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["total_videos"], 2)
        self.assertEqual(payload["transcribed_count"], 1)
        self.assertEqual(payload["pending_count"], 1)
        self.assertIn("last_updated", payload)
        self.assertEqual(payload["channel_dir"], str(self.channel_dir))

    def test_scan_dir_detects_multiple_playlists(self) -> None:
        rows = [
            {
                "video_id": "multiPlay01",
                "title": "Multi playlist",
                "url": "https://youtube.com/watch?v=multiPlay01",
                "assigned_playlist_id": "PL1",
                "assigned_playlist_title": "Playlist A",
                "all_playlist_ids": "PL1|PL2",
                "all_playlist_titles": "Playlist A|Playlist B",
                "folder": "Playlist A [PL1]",
                "transcript_path": "Playlist A [PL1]/Multi playlist [multiPlay01].txt",
                "status": "ok",
                "language": "es",
                "source": "manual",
                "error": "",
            }
        ]
        self._write_rows(rows)
        (self.playlist_dir / "Multi playlist [multiPlay01].txt").write_text(
            "ok", encoding="utf-8"
        )
        # Also create the duplicate transcript in playlist B
        playlist_b = self.channel_dir / "Playlist B [PL2]"
        playlist_b.mkdir(parents=True, exist_ok=True)
        (playlist_b / "Multi playlist [multiPlay01].txt").write_text("ok", encoding="utf-8")

        config = ScanConfig(
            channel_url_or_dir=str(self.channel_dir),
            no_interactive=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            report = scan_channel(config)
        self.assertEqual(report["total_videos"], 1)
        # Both copies are detected and the entry is still marked as transcribed
        self.assertEqual(report["transcribed_count"], 1)

    def test_scan_dir_missing_index_raises(self) -> None:
        empty_dir = Path(self._tmp.name) / "Empty"
        empty_dir.mkdir()
        config = ScanConfig(channel_url_or_dir=str(empty_dir), no_interactive=True)
        with self.assertRaises(FileNotFoundError):
            scan_channel(config)


# ---------------------------------------------------------------------------
# scan_channel: URL mode (mocked YouTubeClient)
# ---------------------------------------------------------------------------

class ScanChannelUrlModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = Path(self._tmp.name) / "out"

    def test_scan_url_writes_status_and_persists_index(self) -> None:
        uploads = [
            _make_record("uploadVid01", "Uploads Video 1"),
            _make_record("uploadVid02", "Uploads Video 2"),
        ]
        playlists = [
            _make_playlist("PL1", "Playlist A", ["uploadVid01"]),
        ]
        playlist_videos = {
            "PL1": [_make_record("uploadVid01", "Uploads Video 1")],
        }
        client = _build_mock_client(
            uploads=uploads,
            playlists=playlists,
            playlist_videos_by_id=playlist_videos,
        )

        with mock.patch.object(scan_module, "YouTubeClient", return_value=client):
            config = ScanConfig(
                channel_url_or_dir="https://www.youtube.com/@test",
                output_dir=self.output_dir,
                no_interactive=True,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                report = scan_channel(config)

        self.assertEqual(report["total_videos"], 2)
        self.assertEqual(report["transcribed_count"], 0)
        self.assertEqual(report["pending_count"], 2)
        self.assertEqual(report["total_playlists"], 1)

        channel_dir = self.output_dir / "Test Channel"
        self.assertTrue((channel_dir / "_meta" / "channel_status.json").exists())
        self.assertTrue((channel_dir / "_meta" / "video_index.csv").exists())
        self.assertTrue((channel_dir / "_meta" / "playlists.csv").exists())

        # Verify video_index.csv has the right rows
        with (channel_dir / "_meta" / "video_index.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        ids = {row["video_id"] for row in rows}
        self.assertIn("uploadVid01", ids)
        self.assertIn("uploadVid02", ids)

        # The video that is in a playlist should be assigned to it
        playlist_row = next(r for r in rows if r["video_id"] == "uploadVid01")
        self.assertEqual(playlist_row["assigned_playlist_id"], "PL1")
        self.assertEqual(playlist_row["status"], "indexed")

    def test_scan_url_uses_uploads_playlist(self) -> None:
        client = _build_mock_client(uploads=[_make_record("abc", "X")])
        with mock.patch.object(scan_module, "YouTubeClient", return_value=client):
            config = ScanConfig(
                channel_url_or_dir="https://www.youtube.com/@test",
                output_dir=self.output_dir,
                no_interactive=True,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                scan_channel(config)
        client.extract_uploads_playlist_videos.assert_called_once_with(
            channel_id=CHANNEL_ID, fallback_channel="Test Channel"
        )

    def test_scan_url_falls_back_when_uploads_empty(self) -> None:
        # When the uploads playlist returns nothing, the scan should still
        # complete gracefully (without raising). The folder won't be created
        # because there are no videos, but the call must not error.
        client = _build_mock_client(
            uploads=[],
            playlists=[],
        )
        with mock.patch.object(scan_module, "YouTubeClient", return_value=client):
            config = ScanConfig(
                channel_url_or_dir="https://www.youtube.com/@test",
                output_dir=self.output_dir,
                no_interactive=True,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                report = scan_channel(config)
        # No videos discovered → 0 everywhere
        self.assertEqual(report["total_videos"], 0)

    def test_scan_url_detects_existing_transcripts(self) -> None:
        uploads = [_make_record("uploadVid01", "Already on disk")]
        client = _build_mock_client(uploads=uploads, playlists=[])

        # Pre-create a transcript file in the resolved channel folder
        channel_dir = self.output_dir / "Test Channel"
        playlist_dir = channel_dir / "_meta"  # should be skipped
        playlist_dir.mkdir(parents=True, exist_ok=True)
        # Real transcript placed somewhere inside the channel folder
        some_sub = channel_dir / "Somewhere"
        some_sub.mkdir(parents=True, exist_ok=True)
        (some_sub / "Already on disk [uploadVid01].txt").write_text("hi", encoding="utf-8")

        with mock.patch.object(scan_module, "YouTubeClient", return_value=client):
            config = ScanConfig(
                channel_url_or_dir="https://www.youtube.com/@test",
                output_dir=self.output_dir,
                no_interactive=True,
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                report = scan_channel(config)
        self.assertEqual(report["transcribed_count"], 1)
        self.assertEqual(report["pending_count"], 0)


# ---------------------------------------------------------------------------
# URL vs directory detection
# ---------------------------------------------------------------------------

class ScanInputTypeTests(unittest.TestCase):
    def test_url_detection(self) -> None:
        self.assertTrue(scan_module._looks_like_url("https://www.youtube.com/@foo"))
        self.assertTrue(scan_module._looks_like_url("http://youtube.com/foo"))
        self.assertTrue(scan_module._looks_like_url("www.youtube.com/foo"))
        self.assertTrue(scan_module._looks_like_url("youtube.com/@foo"))

    def test_directory_detection(self) -> None:
        self.assertFalse(scan_module._looks_like_url(r"C:\Users\me\output\My Channel"))
        self.assertFalse(scan_module._looks_like_url("./output/My Channel"))
        self.assertFalse(scan_module._looks_like_url("My Channel"))


# ---------------------------------------------------------------------------
# Scan helpers
# ---------------------------------------------------------------------------

class ScanHelpersTests(unittest.TestCase):
    def test_is_transcribed_status(self) -> None:
        self.assertTrue(is_transcribed_status("ok"))
        self.assertTrue(is_transcribed_status("OK"))
        self.assertTrue(is_transcribed_status("existing"))
        self.assertFalse(is_transcribed_status("no_subtitles"))
        self.assertFalse(is_transcribed_status("failed"))
        self.assertFalse(is_transcribed_status(""))


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

class CliRegistrationTests(unittest.TestCase):
    def test_scan_is_known_subcommand(self) -> None:
        self.assertIn("scan", KNOWN_SUBCOMMANDS)

    def test_scan_parser_registered(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "scan",
                "https://www.youtube.com/@test",
                "--no-interactive",
                "--cookies",
                "/tmp/c.txt",
                "--output",
                "./out",
            ]
        )
        self.assertEqual(args.command, "scan")
        self.assertEqual(args.channel_url_or_dir, "https://www.youtube.com/@test")
        self.assertTrue(args.no_interactive)
        self.assertEqual(args.cookies, "/tmp/c.txt")
        self.assertEqual(args.output, "./out")

    def test_run_parser_has_cookies(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "https://www.youtube.com/@test",
                "--cookies",
                "/tmp/cookies.txt",
            ]
        )
        self.assertEqual(args.cookies, "/tmp/cookies.txt")

    def test_asr_parser_has_cookies(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "asr",
                "./output/My Channel",
                "--cookies",
                "/tmp/cookies.txt",
            ]
        )
        self.assertEqual(args.cookies, "/tmp/cookies.txt")


# ---------------------------------------------------------------------------
# Transcriber integration: replication to multiple playlists
# ---------------------------------------------------------------------------

class TranscriptReplicatorTests(unittest.TestCase):
    """
    Verify that when a video is in multiple playlists, the transcript is
    physically copied into each playlist folder.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.channel_dir = Path(self._tmp.name) / "Channel"
        self.channel_dir.mkdir(parents=True, exist_ok=True)
        self.ungrouped = self.channel_dir / "Sin_Playlist"
        self.ungrouped.mkdir(parents=True, exist_ok=True)
        self.pl_a = self.channel_dir / "A [PL1]"
        self.pl_b = self.channel_dir / "B [PL2]"
        self.pl_a.mkdir(parents=True, exist_ok=True)
        self.pl_b.mkdir(parents=True, exist_ok=True)

    def _make_config(self) -> Any:
        from transcriptex.transcriber import TranscriberConfig
        return TranscriberConfig(
            channel_url="https://www.youtube.com/@dummy",
            output_dir=Path(self._tmp.name),
            lang_priority=["es", "en"],
            include_tabs=[],
        )

    def test_replicate_to_multiple_folders(self) -> None:
        from transcriptex.transcriber import ChannelTranscriber
        from transcriptex.models import VideoRecord

        config = self._make_config()
        transcriber = ChannelTranscriber(config)
        source = self.ungrouped / "Shared [abc123XYZ_9].txt"
        source.write_text("body", encoding="utf-8")

        memberships = ["PL1", "PL2"]
        playlist_folders = {"PL1": self.pl_a, "PL2": self.pl_b}
        video = VideoRecord(
            video_id="abc123XYZ_9",
            title="Shared",
            url="https://youtube.com/watch?v=abc123XYZ_9",
        )

        copies = transcriber._replicate_transcript_to_memberships(
            video=video,
            memberships=memberships,
            playlist_folders=playlist_folders,
            ungrouped_dir=self.ungrouped,
            source_path=source,
        )
        self.assertEqual(copies, 2)
        self.assertTrue((self.pl_a / "Shared [abc123XYZ_9].txt").exists())
        self.assertTrue((self.pl_b / "Shared [abc123XYZ_9].txt").exists())

    def test_replicate_skips_source_folder(self) -> None:
        from transcriptex.transcriber import ChannelTranscriber
        from transcriptex.models import VideoRecord

        config = self._make_config()
        transcriber = ChannelTranscriber(config)
        # Source is already in PL1
        source = self.pl_a / "Shared [abc123XYZ_9].txt"
        source.write_text("body", encoding="utf-8")

        memberships = ["PL1", "PL2"]
        playlist_folders = {"PL1": self.pl_a, "PL2": self.pl_b}
        video = VideoRecord(
            video_id="abc123XYZ_9",
            title="Shared",
            url="https://youtube.com/watch?v=abc123XYZ_9",
        )

        copies = transcriber._replicate_transcript_to_memberships(
            video=video,
            memberships=memberships,
            playlist_folders=playlist_folders,
            ungrouped_dir=self.ungrouped,
            source_path=source,
        )
        # Only PL2 should be created as a copy
        self.assertEqual(copies, 1)
        self.assertTrue((self.pl_b / "Shared [abc123XYZ_9].txt").exists())

    def test_replicate_returns_zero_for_empty_memberships(self) -> None:
        from transcriptex.transcriber import ChannelTranscriber
        from transcriptex.models import VideoRecord

        config = self._make_config()
        transcriber = ChannelTranscriber(config)
        source = self.ungrouped / "Lonely [abc123XYZ_9].txt"
        source.write_text("body", encoding="utf-8")

        copies = transcriber._replicate_transcript_to_memberships(
            video=VideoRecord(video_id="abc123XYZ_9", title="Lonely", url=""),
            memberships=[],
            playlist_folders={"PL1": self.pl_a},
            ungrouped_dir=self.ungrouped,
            source_path=source,
        )
        self.assertEqual(copies, 0)


# ---------------------------------------------------------------------------
# Transcriber: parallel playlist fetching
# ---------------------------------------------------------------------------

class ParallelPlaylistTests(unittest.TestCase):
    """
    Verify that the parallel playlist-fetcher is used (i.e. that
    extract_playlist_videos is invoked once per playlist and that the
    results are merged correctly into the videos index and membership map).
    """

    def test_playlists_fetched_in_parallel_and_merged(self) -> None:
        from transcriptex.models import ChannelInfo, PlaylistRecord, VideoRecord
        from transcriptex.transcriber import ChannelTranscriber, TranscriberConfig

        config = TranscriberConfig(
            channel_url="https://www.youtube.com/@dummy",
            output_dir=Path(tempfile.gettempdir()),
            lang_priority=["es"],
            include_tabs=[],
        )

        pl_a = PlaylistRecord(
            playlist_id="PL1",
            title="A",
            url="https://www.youtube.com/playlist?list=PL1",
        )
        pl_b = PlaylistRecord(
            playlist_id="PL2",
            title="B",
            url="https://www.youtube.com/playlist?list=PL2",
        )

        rec_a = VideoRecord(video_id="vid_a_1", title="A1", url="u1")
        rec_b = VideoRecord(video_id="vid_b_1", title="B1", url="u2")
        rec_shared = VideoRecord(video_id="vid_shared", title="Shared", url="u3")

        def fake_extract_playlist_videos(playlist_url: str, fallback_channel: str = ""):
            if "PL1" in playlist_url:
                return [rec_a, rec_shared]
            if "PL2" in playlist_url:
                return [rec_b, rec_shared]
            return []

        client = mock.MagicMock()
        client.extract_playlists.return_value = [pl_a, pl_b]
        client.extract_playlist_videos.side_effect = fake_extract_playlist_videos

        transcriber = ChannelTranscriber(config)
        transcriber.client = client

        with tempfile.TemporaryDirectory() as tmp_dir:
            channel_dir = Path(tmp_dir) / "Channel"
            channel_dir.mkdir(parents=True, exist_ok=True)
            videos_by_id: dict[str, VideoRecord] = {}
            membership: dict[str, list[str]] = defaultdict(list)
            playlists = transcriber._collect_playlists_and_membership(
                channel_base_url="https://www.youtube.com/@dummy",
                videos_by_id=videos_by_id,
                playlist_membership=membership,
                fallback_channel="Fallback",
                channel_dir=channel_dir,
            )

        # Both playlists should have been fetched
        self.assertEqual(client.extract_playlist_videos.call_count, 2)
        # Videos should be merged
        self.assertEqual(set(videos_by_id.keys()), {"vid_a_1", "vid_b_1", "vid_shared"})
        # The shared video should be in both playlists
        self.assertCountEqual(membership["vid_shared"], ["PL1", "PL2"])
        # The A-only video should be only in PL1
        self.assertEqual(membership["vid_a_1"], ["PL1"])
        self.assertEqual(membership["vid_b_1"], ["PL2"])


# ---------------------------------------------------------------------------
# Page-size constant exposed for callers
# ---------------------------------------------------------------------------

class ScanConfigTests(unittest.TestCase):
    def test_page_size_default(self) -> None:
        config = ScanConfig(channel_url_or_dir="x")
        self.assertEqual(config.page_size, PAGE_SIZE)

    def test_page_size_custom(self) -> None:
        config = ScanConfig(channel_url_or_dir="x", page_size=5)
        self.assertEqual(config.page_size, 5)


if __name__ == "__main__":
    unittest.main()
