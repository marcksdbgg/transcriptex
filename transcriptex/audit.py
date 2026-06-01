from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transcriptex.fs_utils import (
    build_transcript_filename,
    extract_video_id_from_stem,
    now_utc_iso,
)

META_FOLDER = "_meta"


@dataclass
class AuditConfig:
    channel_dir: Path
    write_files: bool = True
    max_rows_in_console: int = 10
    rename_legacy_files: bool = False
    print_summary: bool = True


def audit_channel_output(config: AuditConfig) -> dict[str, Any]:
    channel_dir = config.channel_dir.resolve()
    meta_dir = channel_dir / META_FOLDER
    video_index_path = meta_dir / "video_index.csv"
    if not video_index_path.exists():
        raise FileNotFoundError(f"Missing file: {video_index_path}")

    rows = _read_video_index(video_index_path)
    indexed_ids = {row["video_id"] for row in rows if row.get("video_id")}
    status_counter = Counter((row.get("status") or "unknown") for row in rows)

    transcript_files, id_to_file, duplicate_file_ids, unparsed_files = _scan_transcript_files(channel_dir)

    renamed_files_count = 0
    if config.rename_legacy_files:
        title_by_id = {row.get("video_id", ""): row.get("title", "") for row in rows}
        renamed_files_count = _rename_legacy_files(id_to_file=id_to_file, title_by_id=title_by_id)
        transcript_files, id_to_file, duplicate_file_ids, unparsed_files = _scan_transcript_files(channel_dir)

    transcript_ids = set(id_to_file.keys())
    ok_expected_ids = {
        row["video_id"]
        for row in rows
        if row.get("video_id") and (row.get("status") in {"ok", "existing"})
    }

    indexed_without_transcript = sorted(indexed_ids - transcript_ids)
    missing_files_for_ok = sorted(ok_expected_ids - transcript_ids)
    orphan_transcripts = sorted(transcript_ids - indexed_ids)

    missing_rows = [row for row in rows if row.get("video_id") in set(indexed_without_transcript)]
    error_counter = Counter(
        (row.get("error") or "").strip() or "(no error message)"
        for row in missing_rows
        if (row.get("status") or "") in {"failed", "no_subtitles", "no_text"}
    )
    status_missing_counter = Counter((row.get("status") or "unknown") for row in missing_rows)

    report = {
        "generated_at": now_utc_iso(),
        "channel_dir": str(channel_dir),
        "total_indexed_videos": len(indexed_ids),
        "total_rows_video_index": len(rows),
        "status_counts": dict(status_counter),
        "transcript_files_found": len(transcript_files),
        "transcript_ids_found": len(transcript_ids),
        "indexed_without_transcript_count": len(indexed_without_transcript),
        "missing_files_for_ok_count": len(missing_files_for_ok),
        "orphan_transcripts_count": len(orphan_transcripts),
        "duplicate_file_ids_count": len(duplicate_file_ids),
        "unparsed_transcript_files_count": len(unparsed_files),
        "renamed_legacy_files_count": renamed_files_count,
        "missing_status_counts": dict(status_missing_counter),
        "top_missing_errors": [
            {"error": error, "count": count}
            for error, count in error_counter.most_common(10)
        ],
    }

    if config.write_files:
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "audit_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_missing_csv(meta_dir / "missing_transcripts.csv", missing_rows, channel_dir)
        _write_orphan_csv(meta_dir / "orphan_transcripts.csv", orphan_transcripts, id_to_file, channel_dir)
        _write_duplicate_csv(meta_dir / "duplicate_transcripts.csv", duplicate_file_ids, channel_dir)

    if config.print_summary:
        _print_console_summary(
            report=report,
            missing_rows=missing_rows,
            max_rows=max(1, config.max_rows_in_console),
        )
    return report


def _read_video_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_missing_csv(path: Path, rows: list[dict[str, str]], channel_dir: Path) -> None:
    headers = [
        "video_id",
        "title",
        "url",
        "status",
        "error",
        "assigned_playlist_title",
        "assigned_playlist_id",
        "suggestion",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_id": row.get("video_id", ""),
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "status": row.get("status", ""),
                    "error": row.get("error", ""),
                    "assigned_playlist_title": row.get("assigned_playlist_title", ""),
                    "assigned_playlist_id": row.get("assigned_playlist_id", ""),
                    "suggestion": _suggest_action(row.get("status", ""), row.get("error", "")),
                }
            )


def _write_orphan_csv(path: Path, orphan_ids: list[str], id_to_file: dict[str, Path], channel_dir: Path) -> None:
    headers = ["video_id", "transcript_path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for video_id in orphan_ids:
            transcript_path = id_to_file.get(video_id)
            writer.writerow(
                {
                    "video_id": video_id,
                    "transcript_path": _relative_path_str(transcript_path, channel_dir),
                }
            )


def _write_duplicate_csv(path: Path, duplicate_rows: list[dict[str, str]], channel_dir: Path) -> None:
    headers = ["video_id", "first_path", "duplicate_path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in duplicate_rows:
            writer.writerow(
                {
                    "video_id": row.get("video_id", ""),
                    "first_path": _relative_path_text(row.get("first_path", ""), channel_dir),
                    "duplicate_path": _relative_path_text(row.get("duplicate_path", ""), channel_dir),
                }
            )


def _relative_path_str(path: Path | None, base: Path) -> str:
    if not path:
        return ""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _relative_path_text(path_text: str, base: Path) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _suggest_action(status: str, error: str) -> str:
    status_value = (status or "").strip().lower()
    error_value = (error or "").strip().lower()
    if status_value == "no_subtitles":
        return "No subtitles available on YouTube. Use ASR fallback if transcript is required."
    if status_value == "no_text":
        return "Subtitle track exists but parsed empty. Retry with --force and latest yt-dlp."
    if status_value == "failed":
        if any(token in error_value for token in ("private", "members-only", "members only", "login")):
            return "Likely restricted/private video. Retry with cookies/auth."
        if any(token in error_value for token in ("unavailable", "deleted", "removed", "not available")):
            return "Video unavailable (deleted or region restricted). Cannot transcribe."
        if "empty video metadata" in error_value:
            return "Metadata extraction failed. Retry with updated yt-dlp and JS runtime."
        return "Extraction failure. Retry with --retries, --verbose, and updated yt-dlp."
    return "No action needed."


def _print_console_summary(report: dict[str, Any], missing_rows: list[dict[str, str]], max_rows: int) -> None:
    print("")
    print("Audit summary:")
    print(f"  Channel dir: {report.get('channel_dir', '')}")
    print(f"  Indexed videos: {report.get('total_indexed_videos', 0)}")
    print(f"  Transcript files found: {report.get('transcript_files_found', 0)}")
    print(f"  Transcript IDs found: {report.get('transcript_ids_found', 0)}")
    print(f"  Indexed without transcript: {report.get('indexed_without_transcript_count', 0)}")
    print(f"  Missing files for status ok/existing: {report.get('missing_files_for_ok_count', 0)}")
    print(f"  Orphan transcripts: {report.get('orphan_transcripts_count', 0)}")
    print(f"  Duplicate transcript IDs: {report.get('duplicate_file_ids_count', 0)}")
    print(f"  Renamed legacy files: {report.get('renamed_legacy_files_count', 0)}")

    status_counts = report.get("missing_status_counts", {})
    if status_counts:
        print("  Missing by status:")
        for key, value in status_counts.items():
            print(f"    - {key}: {value}")

    if not missing_rows:
        return
    print("")
    print(f"Top {min(max_rows, len(missing_rows))} missing videos:")
    for row in missing_rows[:max_rows]:
        status = row.get("status", "")
        title = row.get("title", "")
        video_id = row.get("video_id", "")
        suggestion = _suggest_action(status, row.get("error", ""))
        print(f"  - {video_id} | {status} | {title}")
        print(f"    suggestion: {suggestion}")


def _scan_transcript_files(
    channel_dir: Path,
) -> tuple[list[Path], dict[str, Path], list[dict[str, str]], list[Path]]:
    transcript_files = [path for path in channel_dir.rglob("*.txt") if META_FOLDER not in path.parts]
    id_to_file: dict[str, Path] = {}
    duplicate_file_ids: list[dict[str, str]] = []
    unparsed_files: list[Path] = []

    for path in transcript_files:
        video_id = extract_video_id_from_stem(path.stem)
        if not video_id:
            unparsed_files.append(path)
            continue
        if video_id in id_to_file:
            duplicate_file_ids.append(
                {
                    "video_id": video_id,
                    "first_path": str(id_to_file[video_id]),
                    "duplicate_path": str(path),
                }
            )
            continue
        id_to_file[video_id] = path
    return transcript_files, id_to_file, duplicate_file_ids, unparsed_files


def _rename_legacy_files(id_to_file: dict[str, Path], title_by_id: dict[str, str]) -> int:
    renamed = 0
    for video_id, path in list(id_to_file.items()):
        if path.stem.strip() != video_id:
            continue
        title = title_by_id.get(video_id, "") or video_id
        expected_name = build_transcript_filename(video_id=video_id, title=title)
        target_path = path.with_name(expected_name)
        if target_path == path:
            continue
        if target_path.exists():
            target_path = _with_numeric_suffix(target_path)
        path.rename(target_path)
        renamed += 1
    return renamed


def _with_numeric_suffix(target: Path) -> Path:
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    index = 2
    candidate = target
    while candidate.exists():
        candidate = parent / f"{stem} ({index}){suffix}"
        index += 1
    return candidate
