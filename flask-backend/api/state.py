"""State module for Flask API.

Manages shared global state including caches, progress tracking, and runtime variables.
Thread-safe access is provided via locks for mutable state.
"""
from pathlib import Path
from threading import Lock
import pandas as pd
from .config import PRESELECTED_ROSBAG, PRESELECTED_MODEL

# Locks for thread-safe access to mutable state
_aligned_data_lock = Lock()
_selected_rosbag_lock = Lock()
_reference_timestamp_lock = Lock()

# Runtime state
_SELECTED_ROSBAG = PRESELECTED_ROSBAG
SELECTED_MODEL = PRESELECTED_MODEL

# ALIGNED_DATA: DataFrame mapping reference timestamps to per-topic timestamps for alignment
# Initialize lazily to avoid circular import issues
_ALIGNED_DATA: pd.DataFrame | None = None


def get_aligned_data() -> pd.DataFrame:
    """Get or initialize ALIGNED_DATA (thread-safe)."""
    global _ALIGNED_DATA
    with _aligned_data_lock:
        if _ALIGNED_DATA is None:
            # Lazy import to avoid circular dependency
            from .utils.rosbag import extract_rosbag_name_from_path, load_lookup_tables_for_rosbag
            rosbag_name = extract_rosbag_name_from_path(str(_SELECTED_ROSBAG))
            _ALIGNED_DATA = load_lookup_tables_for_rosbag(rosbag_name)
        return _ALIGNED_DATA


def set_aligned_data(data: pd.DataFrame) -> None:
    """Set ALIGNED_DATA (thread-safe)."""
    global _ALIGNED_DATA
    with _aligned_data_lock:
        _ALIGNED_DATA = data


def get_selected_rosbag():
    """Thread-safe getter for SELECTED_ROSBAG."""
    with _selected_rosbag_lock:
        return _SELECTED_ROSBAG


def set_selected_rosbag(path_value) -> None:
    """Set SELECTED_ROSBAG (thread-safe)."""
    global _SELECTED_ROSBAG, SELECTED_ROSBAG
    with _selected_rosbag_lock:
        _SELECTED_ROSBAG = path_value
        # Keep backward-compatible module variable in sync
        SELECTED_ROSBAG = path_value


# Backward-compatible module-level access (prefer get_selected_rosbag() for thread safety)
SELECTED_ROSBAG = _SELECTED_ROSBAG


# Progress tracking (use update_*_progress functions for thread-safe writes)
_export_progress_lock = Lock()
_search_progress_lock = Lock()
EXPORT_PROGRESS = {"status": "idle", "progress": -1, "message": "Waiting for export..."}
SEARCH_PROGRESS = {"status": "idle", "progress": -1, "message": "Waiting for search..."}


def update_export_progress(status: str = None, progress: float = None, message: str = None) -> None:
    """Thread-safe update for EXPORT_PROGRESS."""
    with _export_progress_lock:
        if status is not None:
            EXPORT_PROGRESS["status"] = status
        if progress is not None:
            EXPORT_PROGRESS["progress"] = progress
        if message is not None:
            EXPORT_PROGRESS["message"] = message


def update_search_progress(status: str = None, progress: float = None, message: str = None) -> None:
    """Thread-safe update for SEARCH_PROGRESS."""
    with _search_progress_lock:
        if status is not None:
            SEARCH_PROGRESS["status"] = status
        if progress is not None:
            SEARCH_PROGRESS["progress"] = progress
        if message is not None:
            SEARCH_PROGRESS["message"] = message


SEARCHED_ROSBAGS: list[str] = []

# Reference timestamp tracking
_current_reference_timestamp: int | None = None
_mapped_timestamps: dict = {}


def get_reference_timestamp():
    """Thread-safe getter for current_reference_timestamp."""
    with _reference_timestamp_lock:
        return _current_reference_timestamp


def set_reference_timestamp(timestamp: int | None) -> None:
    """Thread-safe setter for current_reference_timestamp."""
    global _current_reference_timestamp
    with _reference_timestamp_lock:
        _current_reference_timestamp = timestamp


def get_mapped_timestamps() -> dict:
    """Thread-safe getter for mapped_timestamps."""
    with _reference_timestamp_lock:
        return _mapped_timestamps.copy()


def set_mapped_timestamps(timestamps: dict) -> None:
    """Thread-safe setter for mapped_timestamps."""
    global _mapped_timestamps
    with _reference_timestamp_lock:
        _mapped_timestamps = timestamps


# Backward-compatible module-level access (prefer thread-safe functions)
current_reference_timestamp = _current_reference_timestamp
mapped_timestamps = _mapped_timestamps

# Caches
_lookup_table_cache: dict[str, tuple[pd.DataFrame, float]] = {}
_matching_rosbag_cache = {"paths": [], "timestamp": 0.0}
_file_path_cache_lock = Lock()
_positional_lookup_cache: dict[str, dict[str, dict[str, int]]] = {"data": None, "mtime": None}  # type: ignore[assignment]
