import os
import sys


def _safe_print(text: str) -> None:
    """Print to stdout, replacing any characters the current console can't encode."""
    try:
        print(text)
    except UnicodeEncodeError:
        stream = getattr(sys.stdout, "buffer", None)
        if stream is not None:
            stream.write((text + "\n").encode("utf-8", errors="replace"))
            stream.flush()
        else:
            print(text.encode("ascii", errors="replace").decode("ascii"))


def check_for_old_name(start_dir="."):
    invalid_patterns = ["yt_channel", "yt-channel"]
    ignore_dirs = {".git", "__pycache__", ".venv", "output", "transcriptex.egg-info"}

    # Files that are allowed to mention the old name for documentation purposes
    # (e.g. AGENTS.md explaining the rename history).
    allowed_files = {"AGENTS.md"}

    found_issues = []

    for root, dirs, files in os.walk(start_dir):
        # Filtering ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            file_path = os.path.join(root, file)

            # Check file name
            if any(pattern in file.lower() for pattern in invalid_patterns):
                found_issues.append(f"FileName Match: {file_path}")

            if file == "validate_refactor.py":
                continue

            # Check content (skip documentation files that intentionally
            # mention the old name as part of the rename history).
            if file in allowed_files:
                continue

            if file.endswith(('.py', '.md', '.txt', '.csv', '.json', '.toml')):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines, 1):
                            if any(pattern in line.lower() for pattern in invalid_patterns):
                                found_issues.append(f"Line Match in '{file_path}' (line {i}): {line.strip()}")
                except Exception as e:
                    pass

    return found_issues

if __name__ == "__main__":
    _safe_print("Starting validation sweep across the codebase...\n")
    issues = check_for_old_name()
    if not issues:
        _safe_print("[OK] VALIDATION SUCCESS: The old name 'yt_channel_transcriber' / 'yt-channel' was NOT found anywhere in the codebase.")
        _safe_print("     The refactor to 'transcriptEX' is complete and clean.")
    else:
        _safe_print("[FAIL] VALIDATION FAILED: Found lingering occurrences of the old name:")
        for issue in issues:
            _safe_print("  - " + issue)
