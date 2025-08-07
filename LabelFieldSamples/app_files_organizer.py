#!/usr/bin/env python3
"""
Organize and rename image files under 07_08 using compiled.csv.

- Reads 07_08/compiled.csv (columns: QR, Corn)
- Parses Corn as a list of filenames separated by ';' or ','
- Uses the full QR value (including E1/E2 suffix) as the ucode, e.g., '001E1'
- For each ucode, creates folder 07_08/<ucode>/ and moves images into it as
  <ucode>_ear_<index>.<ext> in the order they appear in compiled.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_corn_filenames(text: str) -> List[str]:
    if not text:
        return []
    parts: List[str] = []
    for seg in text.replace("\n", " ").split(";"):
        parts.extend(seg.split(","))
    return [p.strip() for p in parts if p.strip()]


def derive_ucode(qr_value: str) -> str:
    # Keep the full QR including the E# suffix
    return qr_value.strip()


def build_ucode_to_files(compiled_csv: Path) -> Dict[str, List[str]]:
    ucode_to_files: Dict[str, List[str]] = defaultdict(list)
    seen_per_ucode: Dict[str, set] = defaultdict(set)

    with compiled_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qr = (row.get("QR") or row.get("qr") or "").strip()
            corn = (row.get("Corn") or row.get("corn") or "").strip()
            if not qr or not corn:
                continue
            ucode = derive_ucode(qr)
            for name in parse_corn_filenames(corn):
                if name not in seen_per_ucode[ucode]:
                    ucode_to_files[ucode].append(name)
                    seen_per_ucode[ucode].add(name)

    return ucode_to_files


def collect_existing_files(base_dir: Path, extensions: Iterable[str]) -> Dict[str, List[Path]]:
    by_lower: Dict[str, List[Path]] = defaultdict(list)
    for ext in extensions:
        for p in base_dir.rglob(f"*.{ext}"):
            if p.is_file():
                by_lower[p.name.lower()].append(p)
        for p in base_dir.rglob(f"*.{ext.upper()}"):
            if p.is_file():
                by_lower[p.name.lower()].append(p)
    return by_lower


def resolve_source_path(name: str, index: Dict[str, List[Path]]) -> Optional[Path]:
    key = name.lower()
    candidates = index.get(key, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # If there are multiple candidates, prefer exact case match
    for p in candidates:
        if p.name == name:
            return p
    # Fall back to the first found deterministically (sorted by path)
    return sorted(candidates)[0]


def move_images(
    base_dir: Path,
    compiled_csv: Path,
    extensions: List[str],
    dry_run: bool,
) -> Tuple[int, int, int]:
    ucode_to_files = build_ucode_to_files(compiled_csv)
    existing_index = collect_existing_files(base_dir, extensions)

    moved = 0
    missing = 0
    skipped = 0

    for ucode in sorted(ucode_to_files.keys()):
        target_dir = base_dir / ucode
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        files = ucode_to_files[ucode]
        for idx, name in enumerate(files, start=1):
            src = resolve_source_path(name, existing_index)
            if src is None:
                print(f"Missing source for {ucode}: {name}")
                missing += 1
                continue
            ext = src.suffix.lower() or ".jpg"
            dst_name = f"{ucode}_ear_{idx}{ext}"
            dst = target_dir / dst_name
            if dst.exists():
                print(f"Skip existing: {dst}")
                skipped += 1
                continue
            print(f"{('DRY ' if dry_run else '')}MOVE {src} -> {dst}")
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved += 1
    return moved, missing, skipped


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Organize and rename images using compiled.csv")
    default_base = Path(__file__).parent / "07_08"
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help=f"Base directory to operate on (default: {default_base})",
    )
    parser.add_argument(
        "--compiled",
        type=Path,
        default=(default_base / "compiled.csv"),
        help="Path to compiled.csv (default: BASE_DIR/compiled.csv)",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=["jpg", "jpeg", "png"],
        help="Image extensions to consider (default: jpg jpeg png)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without moving files",
    )

    args = parser.parse_args(argv)

    base_dir: Path = args.base_dir.resolve()
    compiled_csv: Path = args.compiled.resolve()
    if not base_dir.exists():
        print(f"Error: base directory not found: {base_dir}")
        return 2
    if not compiled_csv.exists():
        print(f"Error: compiled.csv not found: {compiled_csv}")
        return 2

    moved, missing, skipped = move_images(base_dir, compiled_csv, args.ext, args.dry_run)
    print("-" * 72)
    print(f"Moved: {moved}")
    print(f"Missing sources: {missing}")
    print(f"Skipped (already existed): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
