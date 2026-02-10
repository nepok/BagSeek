#!/usr/bin/env python3
"""
Migration script: Add per-MCAP first/last timestamps to existing summary.json files.

Reads parquet file footers (no row data) to extract Reference Timestamp
min/max statistics for each MCAP, then patches the summary.json in-place.

Usage:
    python preprocessing/migrate_summary_timestamps.py
"""
import json
import sys
from pathlib import Path

# Ensure bagseek root is on path so 'preprocessing' package resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from preprocessing.config import Config


def get_parquet_timestamps(parquet_path: Path) -> tuple[int | None, int | None]:
    """Extract min/max Reference Timestamp from parquet metadata (no row data read)."""
    pf = pq.ParquetFile(parquet_path)
    metadata = pf.metadata
    if metadata.num_rows == 0:
        return None, None

    schema = pf.schema_arrow
    col_idx = schema.get_field_index("Reference Timestamp")
    file_min = None
    file_max = None

    for rg_idx in range(metadata.num_row_groups):
        col_stats = metadata.row_group(rg_idx).column(col_idx).statistics
        if col_stats is not None and col_stats.has_min_max:
            if file_min is None or col_stats.min < file_min:
                file_min = col_stats.min
            if file_max is None or col_stats.max > file_max:
                file_max = col_stats.max

    # Fallback: read just the column if statistics aren't available
    if file_min is None or file_max is None:
        import pandas as pd
        df = pd.read_parquet(parquet_path, columns=["Reference Timestamp"])
        file_min = int(df["Reference Timestamp"].min())
        file_max = int(df["Reference Timestamp"].max())

    return int(file_min) if file_min is not None else None, int(file_max) if file_max is not None else None


def migrate_summary(summary_path: Path) -> bool:
    """Patch a single summary.json with per-MCAP timestamps. Returns True if modified."""
    with open(summary_path, "r") as f:
        summary = json.load(f)

    mcap_ranges = summary.get("mcap_ranges", [])
    if not mcap_ranges:
        return False

    # Check if already migrated
    if "first_timestamp_ns" in mcap_ranges[0]:
        return False

    lookup_dir = summary_path.parent
    modified = False

    for entry in mcap_ranges:
        mcap_id = entry["mcap_id"]
        parquet_path = lookup_dir / f"{mcap_id}.parquet"
        if not parquet_path.exists():
            print(f"  WARNING: {parquet_path} not found, skipping MCAP {mcap_id}")
            continue

        try:
            first_ts, last_ts = get_parquet_timestamps(parquet_path)
            if first_ts is not None:
                entry["first_timestamp_ns"] = first_ts
                entry["last_timestamp_ns"] = last_ts
                modified = True
        except Exception as e:
            print(f"  WARNING: Failed to read {parquet_path}: {e}")

    if modified:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    return modified


def main():
    config = Config.load_config()
    lookup_tables_dir = config.lookup_tables_dir

    if not lookup_tables_dir.exists():
        print(f"Lookup tables directory not found: {lookup_tables_dir}")
        sys.exit(1)

    summary_files = sorted(lookup_tables_dir.rglob("summary.json"))
    print(f"Found {len(summary_files)} summary.json file(s) in {lookup_tables_dir}\n")

    migrated = 0
    skipped = 0
    for summary_path in summary_files:
        rosbag_name = summary_path.parent.relative_to(lookup_tables_dir)
        try:
            if migrate_summary(summary_path):
                print(f"  MIGRATED: {rosbag_name}")
                migrated += 1
            else:
                print(f"  SKIPPED (already up to date): {rosbag_name}")
                skipped += 1
        except Exception as e:
            print(f"  ERROR: {rosbag_name}: {e}")

    print(f"\nDone. Migrated: {migrated}, Already up to date: {skipped}")


if __name__ == "__main__":
    main()
