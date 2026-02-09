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
    """Load and combine all mcap lookup table files for a rosbag.

    Supports Parquet format.

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

    # Sort files numerically by mcap_id (filename stem)
    def get_mcap_id_numeric(file_path: Path) -> int:
        """Extract numeric mcap_id from filename for sorting."""
        try:
            return int(file_path.stem)
        except (ValueError, TypeError):
            return float('inf')

    parquet_files = sorted(lookup_rosbag_dir.glob("*.parquet"), key=get_mcap_id_numeric)
    if not parquet_files:
        return pd.DataFrame()
    data_files = parquet_files

    # Check cache
    if use_cache and rosbag_name in _lookup_table_cache:
        cached_df, cached_mtime = _lookup_table_cache[rosbag_name]
        latest_mtime = max(f.stat().st_mtime for f in data_files)
        if latest_mtime <= cached_mtime:
            return cached_df

    all_dfs = []
    latest_mtime = 0.0

    for file_path in data_files:
        mcap_id = file_path.stem  # Extract mcap_id from filename
        try:
            stat = file_path.stat()
            latest_mtime = max(latest_mtime, stat.st_mtime)

            df = pd.read_parquet(file_path)

            if len(df) > 0:
                # Add mcap_id as a column to track which mcap this row came from
                df['_mcap_id'] = mcap_id
                all_dfs.append(df)
        except Exception as e:
            logging.warning(f"Failed to load Parquet {file_path}: {e}")
            continue

    if not all_dfs:
        return pd.DataFrame()

    result_df = pd.concat(all_dfs, ignore_index=True)

    # Sort by Reference Timestamp to ensure consistent ordering
    # This is critical for correct mcap_id assignment in ranges
    if 'Reference Timestamp' in result_df.columns and len(result_df) > 0:
        try:
            result_df = result_df.sort_values('Reference Timestamp').reset_index(drop=True)
        except (ValueError, TypeError) as e:
            logging.warning(f"Failed to sort DataFrame by Reference Timestamp: {e}")

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
