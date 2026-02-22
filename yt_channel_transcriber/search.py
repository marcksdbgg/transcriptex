from __future__ import annotations

import csv
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yt_channel_transcriber.fs_utils import (
    extract_video_id_from_stem,
    normalize_list_arg,
    now_utc_iso,
)

META_FOLDER = "_meta"
DEFAULT_FIELDS = ("title", "playlist", "video_id", "transcript")
FIELD_SET = set(DEFAULT_FIELDS)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)

FIELD_TOKEN_WEIGHTS = {
    "video_id": 50,
    "title": 14,
    "playlist": 10,
    "transcript": 2,
}
FIELD_PHRASE_WEIGHTS = {
    "video_id": 200,
    "title": 80,
    "playlist": 60,
    "transcript": 30,
}


@dataclass
class SearchConfig:
    channel_dir: Path
    query: str
    fields: list[str]
    statuses: list[str]
    top: int = 25
    match_all_tokens: bool = False
    regex: bool = False
    case_sensitive: bool = False
    snippet_len: int = 140
    save_csv: bool = False
    print_results: bool = True
    export_dir: Path | None = None
    export_top: int = 0
    export_all: bool = False
    export_ids: list[str] | None = None
    export_indices: list[int] | None = None
    zip_export: bool = False


@dataclass
class SearchResult:
    score: int
    video_id: str
    title: str
    status: str
    playlist: str
    transcript_path: Path | None
    url: str
    snippet: str
    matched_fields: list[str]
    match_count: int
    has_transcript: bool


def search_channel_output(config: SearchConfig) -> dict[str, Any]:
    channel_dir = config.channel_dir.resolve()
    meta_dir = channel_dir / META_FOLDER
    video_index_path = meta_dir / "video_index.csv"
    if not video_index_path.exists():
        raise FileNotFoundError(f"Missing file: {video_index_path}")

    query = (config.query or "").strip()
    if not query:
        raise ValueError("query is required")

    fields = _normalize_fields(config.fields)
    statuses = {value.strip().lower() for value in config.statuses if value.strip()}

    rows = _read_video_index(video_index_path)
    file_by_id = _scan_transcript_files(channel_dir)

    matcher = _build_matcher(query=query, regex=config.regex, case_sensitive=config.case_sensitive)
    results: list[SearchResult] = []

    for row in rows:
        status = (row.get("status") or "").strip().lower()
        if statuses and status not in statuses:
            continue

        video_id = (row.get("video_id") or "").strip()
        title = (row.get("title") or "").strip()
        playlist = (row.get("assigned_playlist_title") or "").strip()
        url = (row.get("url") or "").strip()
        transcript_path = _resolve_transcript_path(row=row, channel_dir=channel_dir, file_by_id=file_by_id)
        has_transcript = transcript_path is not None and transcript_path.exists()

        transcript_text = ""
        if "transcript" in fields and has_transcript:
            try:
                transcript_text = transcript_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                transcript_text = ""

        field_values = {
            "video_id": video_id,
            "title": title,
            "playlist": playlist,
            "transcript": transcript_text,
        }

        match_data = _score_row(
            field_values=field_values,
            fields=fields,
            matcher=matcher,
            match_all_tokens=config.match_all_tokens,
            case_sensitive=config.case_sensitive,
        )
        if not match_data["matched"]:
            continue

        snippet = _build_snippet(
            text=transcript_text,
            matcher=matcher,
            snippet_len=max(60, int(config.snippet_len)),
            case_sensitive=config.case_sensitive,
        )
        results.append(
            SearchResult(
                score=int(match_data["score"]),
                video_id=video_id,
                title=title,
                status=status,
                playlist=playlist,
                transcript_path=transcript_path if has_transcript else None,
                url=url,
                snippet=snippet,
                matched_fields=sorted(match_data["matched_fields"]),
                match_count=int(match_data["match_count"]),
                has_transcript=has_transcript,
            )
        )

    results.sort(
        key=lambda item: (
            -item.score,
            -(1 if item.has_transcript else 0),
            item.title.lower(),
            item.video_id,
        )
    )
    top_n = max(1, int(config.top))
    top_results = results[:top_n]

    export_report = _export_selected_results(config=config, results=top_results, channel_dir=channel_dir)
    search_csv_path = ""
    if config.save_csv:
        search_csv_path = str(_write_search_results_csv(meta_dir=meta_dir, results=top_results, channel_dir=channel_dir))

    if config.print_results:
        _print_search_results(
            query=query,
            fields=fields,
            statuses=statuses,
            total=len(results),
            top_results=top_results,
            search_csv_path=search_csv_path,
            export_report=export_report,
        )

    return {
        "generated_at": now_utc_iso(),
        "channel_dir": str(channel_dir),
        "query": query,
        "fields": fields,
        "statuses": sorted(statuses),
        "total_matches": len(results),
        "returned_results": len(top_results),
        "csv_path": search_csv_path,
        "exported_count": export_report["exported_count"],
        "export_skipped_count": export_report["skipped_count"],
        "export_dir": export_report["export_dir"],
        "zip_path": export_report["zip_path"],
    }


def parse_comma_int_list(value: str) -> list[int]:
    values: list[int] = []
    for item in normalize_list_arg(value):
        try:
            number = int(item)
        except ValueError:
            continue
        if number > 0 and number not in values:
            values.append(number)
    return values


def _normalize_fields(fields: list[str]) -> list[str]:
    normalized: list[str] = []
    for field in fields:
        name = field.strip().lower()
        if not name or name not in FIELD_SET:
            continue
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        return list(DEFAULT_FIELDS)
    return normalized


def _read_video_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _scan_transcript_files(channel_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for file_path in channel_dir.rglob("*.txt"):
        if META_FOLDER in file_path.parts:
            continue
        video_id = extract_video_id_from_stem(file_path.stem)
        if not video_id or video_id in mapping:
            continue
        mapping[video_id] = file_path
    return mapping


def _resolve_transcript_path(row: dict[str, str], channel_dir: Path, file_by_id: dict[str, Path]) -> Path | None:
    raw_path = (row.get("transcript_path") or "").strip()
    if raw_path:
        candidate = channel_dir / raw_path
        if candidate.exists():
            return candidate
    video_id = (row.get("video_id") or "").strip()
    if not video_id:
        return None
    return file_by_id.get(video_id)


def _build_matcher(query: str, regex: bool, case_sensitive: bool) -> dict[str, Any]:
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(query, flags=flags)
        return {
            "type": "regex",
            "pattern": pattern,
            "tokens": [],
            "phrase": query if case_sensitive else query.casefold(),
        }
    token_source = query if case_sensitive else query.casefold()
    tokens = list(dict.fromkeys(TOKEN_RE.findall(token_source)))
    phrase = query.strip()
    phrase_norm = phrase if case_sensitive else phrase.casefold()
    return {
        "type": "plain",
        "tokens": tokens,
        "phrase": phrase_norm,
    }


def _score_row(
    field_values: dict[str, str],
    fields: list[str],
    matcher: dict[str, Any],
    match_all_tokens: bool,
    case_sensitive: bool,
) -> dict[str, Any]:
    if matcher["type"] == "regex":
        return _score_row_regex(field_values=field_values, fields=fields, pattern=matcher["pattern"])
    return _score_row_plain(
        field_values=field_values,
        fields=fields,
        tokens=matcher["tokens"],
        phrase=matcher["phrase"],
        match_all_tokens=match_all_tokens,
        case_sensitive=case_sensitive,
    )


def _score_row_regex(field_values: dict[str, str], fields: list[str], pattern: re.Pattern[str]) -> dict[str, Any]:
    score = 0
    match_count = 0
    matched_fields: set[str] = set()
    for field in fields:
        text = field_values.get(field, "")
        if not text:
            continue
        count = 0
        for _ in pattern.finditer(text):
            count += 1
        if count <= 0:
            continue
        matched_fields.add(field)
        match_count += count
        score += FIELD_PHRASE_WEIGHTS.get(field, 20) * min(count, 3)
    return {
        "matched": bool(matched_fields),
        "score": score,
        "match_count": match_count,
        "matched_fields": matched_fields,
    }


def _score_row_plain(
    field_values: dict[str, str],
    fields: list[str],
    tokens: list[str],
    phrase: str,
    match_all_tokens: bool,
    case_sensitive: bool,
) -> dict[str, Any]:
    score = 0
    match_count = 0
    matched_fields: set[str] = set()
    tokens_found: set[str] = set()

    for field in fields:
        raw_text = field_values.get(field, "")
        if not raw_text:
            continue
        text = raw_text if case_sensitive else raw_text.casefold()

        phrase_hits = 0
        if phrase:
            phrase_hits = text.count(phrase)
            if phrase_hits > 0:
                matched_fields.add(field)
                match_count += phrase_hits
                score += FIELD_PHRASE_WEIGHTS.get(field, 20) * min(phrase_hits, 3)

        token_hits = 0
        for token in tokens:
            count = text.count(token)
            if count <= 0:
                continue
            token_hits += count
            tokens_found.add(token)
        if token_hits > 0:
            matched_fields.add(field)
            match_count += token_hits
            score += FIELD_TOKEN_WEIGHTS.get(field, 1) * min(token_hits, 10)

        if field == "video_id" and phrase and text == phrase:
            score += 200

    if match_all_tokens and tokens and any(token not in tokens_found for token in tokens):
        return {"matched": False, "score": 0, "match_count": 0, "matched_fields": set()}

    if tokens and all(token in tokens_found for token in tokens):
        score += 20

    return {
        "matched": bool(matched_fields),
        "score": score,
        "match_count": match_count,
        "matched_fields": matched_fields,
    }


def _build_snippet(text: str, matcher: dict[str, Any], snippet_len: int, case_sensitive: bool) -> str:
    if not text:
        return ""

    index = -1
    query_len = 0
    if matcher["type"] == "regex":
        match = matcher["pattern"].search(text)
        if match:
            index = match.start()
            query_len = max(1, match.end() - match.start())
    else:
        phrase = matcher["phrase"]
        probe = text if case_sensitive else text.casefold()
        if phrase:
            index = probe.find(phrase)
            query_len = len(phrase)
        if index < 0:
            for token in matcher["tokens"]:
                token_index = probe.find(token)
                if token_index >= 0:
                    index = token_index
                    query_len = len(token)
                    break

    if index < 0:
        index = 0
    half = max(20, snippet_len // 2)
    start = max(0, index - half)
    end = min(len(text), index + query_len + half)
    snippet = " ".join(text[start:end].split())
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(text):
        snippet = f"{snippet}..."
    return snippet


def _write_search_results_csv(meta_dir: Path, results: list[SearchResult], channel_dir: Path) -> Path:
    meta_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_utc_iso().replace(":", "-")
    path = meta_dir / f"search_results_{timestamp}.csv"
    headers = [
        "rank",
        "score",
        "video_id",
        "title",
        "status",
        "playlist",
        "url",
        "transcript_path",
        "matched_fields",
        "match_count",
        "snippet",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for index, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "rank": index,
                    "score": result.score,
                    "video_id": result.video_id,
                    "title": result.title,
                    "status": result.status,
                    "playlist": result.playlist,
                    "url": result.url,
                    "transcript_path": _relative_path(result.transcript_path, channel_dir),
                    "matched_fields": "|".join(result.matched_fields),
                    "match_count": result.match_count,
                    "snippet": result.snippet,
                }
            )
    return path


def _export_selected_results(config: SearchConfig, results: list[SearchResult], channel_dir: Path) -> dict[str, Any]:
    selection = _select_results_for_export(config=config, results=results)
    if not selection:
        return {
            "exported_count": 0,
            "skipped_count": 0,
            "export_dir": "",
            "zip_path": "",
        }

    target_dir = config.export_dir
    if target_dir is None:
        timestamp = now_utc_iso().replace(":", "-")
        target_dir = channel_dir / "_exports" / f"search_{timestamp}"
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    exported_rows: list[dict[str, str]] = []
    exported_count = 0
    skipped_count = 0
    used_names: set[str] = set()

    for index, result in enumerate(selection, start=1):
        if not result.transcript_path or not result.transcript_path.exists():
            skipped_count += 1
            continue
        file_name = result.transcript_path.name
        if file_name in used_names:
            file_name = f"{result.video_id}_{file_name}"
        used_names.add(file_name)

        destination = target_dir / file_name
        shutil.copy2(result.transcript_path, destination)
        exported_count += 1
        exported_rows.append(
            {
                "rank": str(index),
                "score": str(result.score),
                "video_id": result.video_id,
                "title": result.title,
                "status": result.status,
                "playlist": result.playlist,
                "source_path": _relative_path(result.transcript_path, channel_dir),
                "exported_path": str(destination.name),
                "url": result.url,
            }
        )

    manifest_path = target_dir / "export_manifest.csv"
    _write_export_manifest(manifest_path, exported_rows)
    payload = {
        "query": config.query,
        "generated_at": now_utc_iso(),
        "exported_count": exported_count,
        "skipped_count": skipped_count,
        "source_channel_dir": str(channel_dir),
    }
    (target_dir / "export_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    zip_path = ""
    if config.zip_export and exported_count > 0:
        zip_path = str(_zip_export_folder(target_dir))

    return {
        "exported_count": exported_count,
        "skipped_count": skipped_count,
        "export_dir": str(target_dir),
        "zip_path": zip_path,
    }


def _select_results_for_export(config: SearchConfig, results: list[SearchResult]) -> list[SearchResult]:
    if not results:
        return []

    if config.export_all:
        return list(results)

    if config.export_ids:
        ids = {item.strip() for item in config.export_ids if item.strip()}
        return [result for result in results if result.video_id in ids]

    if config.export_indices:
        output: list[SearchResult] = []
        for item in config.export_indices:
            index = item - 1
            if index < 0 or index >= len(results):
                continue
            output.append(results[index])
        return output

    if config.export_top > 0:
        return results[: config.export_top]

    return []


def _write_export_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    headers = [
        "rank",
        "score",
        "video_id",
        "title",
        "status",
        "playlist",
        "source_path",
        "exported_path",
        "url",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _zip_export_folder(folder: Path) -> Path:
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in folder.rglob("*"):
            if not file_path.is_file():
                continue
            archive.write(file_path, arcname=str(file_path.relative_to(folder)))
    return zip_path


def _relative_path(path: Path | None, base: Path) -> str:
    if not path:
        return ""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _print_search_results(
    query: str,
    fields: list[str],
    statuses: set[str],
    total: int,
    top_results: list[SearchResult],
    search_csv_path: str,
    export_report: dict[str, Any],
) -> None:
    print("")
    print("Search summary:")
    print(f"  Query: {query}")
    print(f"  Fields: {','.join(fields)}")
    if statuses:
        print(f"  Status filter: {','.join(sorted(statuses))}")
    else:
        print("  Status filter: (all)")
    print(f"  Total matches: {total}")
    print(f"  Returned results: {len(top_results)}")
    if search_csv_path:
        print(f"  CSV: {search_csv_path}")

    if top_results:
        print("")
        print("Results:")
        for index, result in enumerate(top_results, start=1):
            print(
                f"  {index}. [{result.score}] {result.video_id} | {result.status} | {result.title}"
            )
            print(f"     playlist: {result.playlist or '(none)'}")
            print(f"     transcript: {'yes' if result.has_transcript else 'no'}")
            if result.transcript_path:
                print(f"     path: {result.transcript_path}")
            if result.snippet:
                print(f"     snippet: {result.snippet}")

    if export_report["exported_count"] > 0 or export_report["skipped_count"] > 0:
        print("")
        print("Export summary:")
        print(f"  Exported: {export_report['exported_count']}")
        print(f"  Skipped (without transcript): {export_report['skipped_count']}")
        print(f"  Export dir: {export_report['export_dir']}")
        if export_report["zip_path"]:
            print(f"  Zip: {export_report['zip_path']}")
