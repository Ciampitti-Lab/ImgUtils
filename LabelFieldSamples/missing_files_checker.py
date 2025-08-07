#!/usr/bin/env python3
"""
Check whether all image files referenced in compiled.csv exist somewhere under 07_08.

- Reads 07_08/compiled.csv (columns: QR, Corn)
- The Corn column may contain one or many filenames separated by ';' or ','
- Crawls 07_08 recursively to collect all image filenames
- Reports any filenames in compiled.csv that do not exist under 07_08
- Outputs results grouped by their originating QR code
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from collections import defaultdict


def parse_corn_filenames(corn_field: str) -> List[str]:
    if not corn_field:
        return []
    # Split primarily on ';', but also allow commas just in case
    raw_parts: List[str] = []
    for part in corn_field.replace("\n", " ").split(";"):
        raw_parts.extend(part.split(","))
    cleaned = [p.strip() for p in raw_parts]
    return [p for p in cleaned if p]


def collect_compiled_index(
    compiled_csv_path: Path,
) -> Tuple[Set[str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Read compiled.csv and build indices:

    Returns:
    - referenced_files: set of all filenames referenced by any row
    - qr_to_files: mapping QR -> set of filenames for that QR
    - file_to_qrs: mapping filename -> set of QRs that reference it
    """
    referenced_files: Set[str] = set()
    qr_to_files: Dict[str, Set[str]] = defaultdict(set)
    file_to_qrs: Dict[str, Set[str]] = defaultdict(set)

    with compiled_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qr_value = (row.get("QR") or row.get("qr") or "").strip()
            corn_value = (row.get("Corn") or row.get("corn") or "").strip()
            for name in parse_corn_filenames(corn_value):
                referenced_files.add(name)
                qr_to_files[qr_value].add(name)
                file_to_qrs[name].add(qr_value)

    return referenced_files, qr_to_files, file_to_qrs


def collect_existing_image_filenames(base_dir: Path, extensions: Iterable[str]) -> Set[str]:
    existing: Set[str] = set()
    # Use rglob for each extension and collect basename only
    for ext in extensions:
        for p in base_dir.rglob(f"*.{ext}"):
            if p.is_file():
                existing.add(p.name)
        for p in base_dir.rglob(f"*.{ext.upper()}"):
            if p.is_file():
                existing.add(p.name)
    return existing


def find_case_mismatches(missing: Set[str], existing: Set[str]) -> Tuple[List[Tuple[str, str]], Set[str]]:
    existing_lower = {name.lower(): name for name in existing}
    mismatches: List[Tuple[str, str]] = []
    actually_missing: Set[str] = set()
    for name in missing:
        lower = name.lower()
        if lower in existing_lower:
            mismatches.append((name, existing_lower[lower]))
        else:
            actually_missing.add(name)
    return mismatches, actually_missing


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify all image files referenced by compiled.csv exist somewhere under the base directory."
        )
    )
    default_base = Path(__file__).parent / "07_08"
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help=f"Base directory to scan (default: {default_base})",
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

    args = parser.parse_args(argv)
    base_dir: Path = args.base_dir.resolve()
    compiled_csv: Path = args.compiled.resolve()

    if not base_dir.exists():
        print(f"Error: base directory does not exist: {base_dir}")
        return 2
    if not compiled_csv.exists():
        print(f"Error: compiled.csv not found: {compiled_csv}")
        return 2

    referenced_files, qr_to_files, file_to_qrs = collect_compiled_index(compiled_csv)
    existing_files = collect_existing_image_filenames(base_dir, args.ext)

    missing = referenced_files - existing_files
    mismatches, actually_missing = find_case_mismatches(missing, existing_files)

    print("=" * 72)
    print(f"Base dir: {base_dir}")
    print(f"Compiled CSV: {compiled_csv}")
    print(f"Referenced files: {len(referenced_files)}")
    print(f"Existing files under base: {len(existing_files)}")
    print(f"Case mismatches: {len(mismatches)}")
    print(f"Actually missing: {len(actually_missing)}")
    print("=" * 72)

    if mismatches:
        # Group mismatches by QR code(s)
        mismatches_by_qr: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for csv_name, actual_name in mismatches:
            qrs = file_to_qrs.get(csv_name, {""})
            for qr in qrs:
                mismatches_by_qr[qr].append((csv_name, actual_name))

        print("\nCase-only mismatches by QR (CSV name -> actual file):")
        for qr in sorted(mismatches_by_qr.keys()):
            print(f"  QR {qr} ({len(mismatches_by_qr[qr])}):")
            for csv_name, actual_name in sorted(mismatches_by_qr[qr]):
                print(f"    {csv_name} -> {actual_name}")

    if actually_missing:
        # Group missing by QR code(s)
        missing_by_qr: Dict[str, List[str]] = defaultdict(list)
        for name in actually_missing:
            qrs = file_to_qrs.get(name, {""})
            for qr in qrs:
                missing_by_qr[qr].append(name)

        print("\nMissing files by QR (not found anywhere under base):")
        for qr in sorted(missing_by_qr.keys()):
            print(f"  QR {qr} ({len(missing_by_qr[qr])}):")
            for name in sorted(missing_by_qr[qr]):
                print(f"    {name}")

    if not mismatches and not actually_missing:
        print("\nAll referenced files are present.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
