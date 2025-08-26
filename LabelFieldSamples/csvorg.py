from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class TransformedRow:
    qr: str
    corn: str


def normalize_header(header: str) -> str:
    """Normalize header names to match flexibly across files."""
    return "".join(ch for ch in header.lower().strip() if ch.isalnum())


def find_column_keys(headers: Iterable[str]) -> Dict[str, Optional[str]]:
    """Map normalized logical names to actual CSV header names if present.

    Returns a mapping with keys: plot, qr1, corn1, qr2, corn2
    Values are the actual header strings from the file (or None if missing).
    """
    normalized_to_actual: Dict[str, str] = {
        normalize_header(h): h for h in headers
    }

    def actual(name: str) -> Optional[str]:
        return normalized_to_actual.get(name)

    return {
        "plot": actual("plot"),
        "qr1": actual("qr1") or actual("qr_1") or actual("qr 1".replace(" ", "")),
        "corn1": actual("corn1") or actual("corn_1") or actual("corn 1".replace(" ", "")),
        "qr2": actual("qr2") or actual("qr_2") or actual("qr 2".replace(" ", "")),
        "corn2": actual("corn2") or actual("corn_2") or actual("corn 2".replace(" ", "")),
    }


def coalesce(values: List[Optional[str]]) -> Optional[str]:
    for v in values:
        if v is not None:
            return v
    return None


def extract_rows_from_csv(csv_path: Path) -> List[TransformedRow]:
    rows: List[TransformedRow] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return rows

            colmap = find_column_keys(reader.fieldnames)

            # We require at least qr1/corn1 OR qr2/corn2 to proceed
            if not (colmap["qr1"] or colmap["qr2"]):
                return rows

            for raw in reader:
                def get(col_key: str) -> str:
                    col_name = colmap[col_key]
                    return (raw.get(col_name, "") if col_name else "").strip()

                qr1 = get("qr1")
                corn1 = get("corn1")
                qr2 = get("qr2")
                corn2 = get("corn2")

                if qr1:
                    rows.append(TransformedRow(qr=qr1, corn=corn1))
                if qr2:
                    rows.append(TransformedRow(qr=qr2, corn=corn2))
    except Exception as exc:
        print(f"Warning: failed to read {csv_path}: {exc}", file=sys.stderr)

    return rows


def iter_csv_files(base_dir: Path) -> Iterable[Path]:
    for path in base_dir.rglob("*.csv"):
        # Skip compiled outputs if re-running
        if path.name.lower().startswith("compiled"):
            continue
        yield path


def compile_csvs(base_dir: Path) -> List[TransformedRow]:
    all_rows: List[TransformedRow] = []
    for csv_file in iter_csv_files(base_dir):
        all_rows.extend(extract_rows_from_csv(csv_file))
    
    # Remove duplicates based on QR code, keeping first occurrence
    seen_qrs = set()
    unique_rows = []
    for row in all_rows:
        if row.qr and row.qr not in seen_qrs:
            seen_qrs.add(row.qr)
            unique_rows.append(row)
    
    # Sort by QR code lexicographically
    unique_rows.sort(key=lambda r: r.qr)
    return unique_rows


def write_compiled(rows: List[TransformedRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["QR", "Corn"])
        for r in rows:
            writer.writerow([r.qr, r.corn])


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl a base directory for CSVs, drop Plot column, convert pairs (QR 1/Corn 1, QR 2/Corn 2)\n"
            "into individual rows (QR, Corn), merge all, and sort by QR."
        )
    )
    default_base = Path(__file__).parent / "07_08"
    default_output = default_base / "compiled.csv"

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help=f"Base directory to crawl (default: {default_base})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output CSV file path (default: {default_output})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    base_dir: Path = args.base_dir.resolve()
    output_path: Path = args.output.resolve()

    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Error: base directory not found: {base_dir}", file=sys.stderr)
        return 2

    # Count total rows before deduplication for reporting
    total_rows = 0
    for csv_file in iter_csv_files(base_dir):
        total_rows += len(extract_rows_from_csv(csv_file))
    
    rows = compile_csvs(base_dir)
    write_compiled(rows, output_path)
    
    duplicates_removed = total_rows - len(rows)
    if duplicates_removed > 0:
        print(f"Wrote {len(rows)} unique rows to {output_path} (removed {duplicates_removed} duplicates)")
    else:
        print(f"Wrote {len(rows)} rows to {output_path} (no duplicates found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


