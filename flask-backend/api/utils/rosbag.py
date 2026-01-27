"""Rosbag utility functions."""
import logging
import json
from pathlib import Path
import pandas as pd
from ..config import LOOKUP_TABLES, POSITIONAL_LOOKUP_TABLE
from ..state import (
    _lookup_table_cache,
    _positional_lookup_cache,
)


def extract_rosbag_name_from_path(rosbag_path: str) -> str:
    """Extract the correct rosbag name from a path, handling multipart rosbags.
    
    For multipart rosbags, the lookup tables are stored in a directory named after
    the parent directory (e.g., 'rosbag2_xxx_multi_parts'), not the individual part.
    
    Args:
        rosbag_path: Full path to the rosbag (can be a multipart rosbag part)
    
    Returns:
        Rosbag name to use for lookup tables (parent directory name for multipart, basename for regular)
    
    Examples:
        - Regular: '/path/to/rosbag2_2025_07_25-12_17_25' -> 'rosbag2_2025_07_25-12_17_25'
        - Multipart: '/path/to/rosbag2_xxx_multi_parts/Part_1' -> 'rosbag2_xxx_multi_parts/Part_1'
    """
    path_obj = Path(rosbag_path)
    basename = path_obj.name
    parent_name = path_obj.parent.name
    
    # Check if this is a multipart rosbag (parent ends with _multi_parts and current is Part_N)
    if parent_name.endswith("_multi_parts") and basename.startswith("Part_"):
        # For multipart rosbags, return parent_name/Part_N (e.g., 'rosbag2_xxx_multi_parts/Part_1')
        return f"{parent_name}/{basename}"
    else:
        # For regular rosbags, use the directory name itself
        return basename


def load_lookup_tables_for_rosbag(rosbag_name: str, use_cache: bool = True) -> pd.DataFrame:
    """Load and combine all mcap CSV files for a rosbag.
    
    Args:
        rosbag_name: Name of the rosbag (directory name, not full path)
        use_cache: Whether to use cached data if available
    
    Returns:
        Combined DataFrame with all lookup table data, or empty DataFrame if none found.
    """
    if not LOOKUP_TABLES:
        return pd.DataFrame()
    
    lookup_rosbag_dir = LOOKUP_TABLES / rosbag_name
    if not lookup_rosbag_dir.exists():
        return pd.DataFrame()
    
    # Check cache
    if use_cache and rosbag_name in _lookup_table_cache:
        cached_df, cached_mtime = _lookup_table_cache[rosbag_name]
        # Check if any CSV file was modified
        csv_files = sorted(lookup_rosbag_dir.glob("*.csv"))
        if csv_files:
            latest_mtime = max(f.stat().st_mtime for f in csv_files)
            if latest_mtime <= cached_mtime:
                return cached_df
    
    csv_files = sorted(lookup_rosbag_dir.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()
    
    all_dfs = []
    latest_mtime = 0.0
    for csv_path in sorted(csv_files):
        mcap_id = csv_path.stem  # Extract mcap_id from filename (without .csv extension)
        try:
            stat = csv_path.stat()
            latest_mtime = max(latest_mtime, stat.st_mtime)
            df = pd.read_csv(csv_path, dtype=str)
            if len(df) > 0:
                # Add mcap_id as a column to track which mcap this row came from
                df['_mcap_id'] = mcap_id
                all_dfs.append(df)
        except Exception as e:
            logging.warning(f"Failed to load CSV {csv_path}: {e}")
            continue
    
    if not all_dfs:
        return pd.DataFrame()
    
    result_df = pd.concat(all_dfs, ignore_index=True)
    
    # Cache the result
    if use_cache:
        _lookup_table_cache[rosbag_name] = (result_df, latest_mtime)
    
    return result_df


def _load_positional_lookup() -> dict[str, dict[str, dict[str, int | dict[str, int]]]]:
    """
    Load and cache the positional lookup JSON, refreshing when the file changes.
    
    Returns structure: {
        "rosbag_name": {
            "lat,lon": {
                "total": int,
                "mcaps": {"mcap_id": int, ...}
            }
        }
    }
    """
    if not POSITIONAL_LOOKUP_TABLE.exists():
        raise FileNotFoundError(f"Positional lookup file not found at {POSITIONAL_LOOKUP_TABLE}")

    stat = POSITIONAL_LOOKUP_TABLE.stat()
    cached_mtime = _positional_lookup_cache.get("mtime")
    if _positional_lookup_cache.get("data") is None or cached_mtime != stat.st_mtime:
        with POSITIONAL_LOOKUP_TABLE.open("r", encoding="utf-8") as fp:
            _positional_lookup_cache["data"] = json.load(fp)
        _positional_lookup_cache["mtime"] = stat.st_mtime

    return _positional_lookup_cache["data"]  # type: ignore[return-value]
