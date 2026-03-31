"""
normalize_utf8.py
-----------------
Converts all text-based source files in a codebase to UTF-8 encoding.
Runs on the directory it lives in by default, or accepts a path as an argument.

Usage:
    python normalize_utf8.py                  # scans script's own directory
    python normalize_utf8.py C:/path/to/proj  # scans given directory
    python normalize_utf8.py --dry-run # Preview changes first (no files touched)
"""

import os
import sys
import argparse
from pathlib import Path

# File extensions to scan and normalize
TEXT_EXTENSIONS = {
    ".py", ".txt", ".md", ".rst", ".toml", ".cfg", ".ini",
    ".json", ".yaml", ".yml", ".env", ".sh", ".bat", ".ps1",
    ".html", ".css", ".js", ".ts", ".csv", ".xml", ".sql",
}

# Encodings to try when reading, in order of likelihood
CANDIDATE_ENCODINGS = [
    "utf-8",
    "latin-1",
    "iso-8859-1",
    "cp1252",
    "utf-16",
]

# Directories to always skip
SKIP_DIRS = {
    ".venv", "venv", "env", ".env",
    "__pycache__", ".git", ".hg", ".svn",
    "node_modules", ".tox", "dist", "build", ".mypy_cache",
}


def detect_and_read(filepath: Path):
    """Try each candidate encoding until one works. Returns (text, encoding_used)."""
    for enc in CANDIDATE_ENCODINGS:
        try:
            text = filepath.read_text(encoding=enc)
            return text, enc
        except (UnicodeDecodeError, LookupError):
            continue
    return None, None


def normalize_file(filepath: Path, dry_run: bool) -> dict:
    """
    Attempt to read the file, detect its encoding, and re-save as UTF-8.
    Returns a result dict describing what happened.
    """
    result = {"file": str(filepath), "status": None, "original_encoding": None}

    # First, check if it's already valid UTF-8
    try:
        filepath.read_text(encoding="utf-8")
        result["status"] = "skipped"  # Already UTF-8
        return result
    except UnicodeDecodeError:
        pass  # Needs conversion

    # Try to detect encoding
    text, detected_enc = detect_and_read(filepath)

    if text is None:
        result["status"] = "failed"
        return result

    result["original_encoding"] = detected_enc

    if not dry_run:
        filepath.write_text(text, encoding="utf-8")

    result["status"] = "converted"
    return result


def run(root: Path, dry_run: bool):
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Scanning: {root.resolve()}\n")

    results = {"converted": [], "failed": [], "skipped": 0}

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            filepath = Path(dirpath) / filename

            if filepath.suffix.lower() not in TEXT_EXTENSIONS:
                continue

            result = normalize_file(filepath, dry_run)

            if result["status"] == "converted":
                results["converted"].append(result)
                tag = "[DRY RUN] Would convert" if dry_run else "Converted"
                print(f"  ✔  {tag}: {result['file']}  ({result['original_encoding']} → utf-8)")
            elif result["status"] == "failed":
                results["failed"].append(result)
                print(f"  ✘  Failed (unknown encoding): {result['file']}")
            else:
                results["skipped"] += 1

    # Summary
    print("\n" + "─" * 60)
    print(f"  {'Would convert' if dry_run else 'Converted'} : {len(results['converted'])} file(s)")
    print(f"  Failed        : {len(results['failed'])} file(s)")
    print(f"  Already UTF-8 : {results['skipped']} file(s)")
    print("─" * 60)

    if results["failed"]:
        print("\nFiles that could not be decoded:")
        for r in results["failed"]:
            print(f"  {r['file']}")

    if dry_run and results["converted"]:
        print("\nRun without --dry-run to apply changes.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Normalize all text files in a codebase to UTF-8."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Root folder to scan (default: directory this script lives in).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be changed without modifying any files.",
    )
    args = parser.parse_args()

    # Default: the directory the script itself lives in
    root = Path(args.folder) if args.folder else Path(__file__).parent

    if not root.exists() or not root.is_dir():
        print(f"Error: '{root}' is not a valid directory.")
        sys.exit(1)

    run(root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()