"""State module for Flask API.

Manages shared global state including caches, progress tracking, and runtime variables.
Thread-safe access is provided via locks for mutable state.
"""
import uuid
from pathlib import Path
from threading import Lock
import pandas as pd
from .config import PRESELECTED_MODEL

# Locks for thread-safe access to mutable state
_reference_timestamp_lock = Lock()

SELECTED_MODEL = PRESELECTED_MODEL


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


# Search cancellation via generation ID
_search_id_lock = Lock()
_current_search_id: str | None = None


def start_new_search() -> str:
    """Mark a new search as current, superseding any running search. Returns the new search ID."""
    global _current_search_id
    with _search_id_lock:
        _current_search_id = uuid.uuid4().hex
        return _current_search_id


def is_search_cancelled(search_id: str) -> bool:
    """Check if the given search has been superseded by a newer one."""
    with _search_id_lock:
        return _current_search_id != search_id


def cancel_current_search() -> None:
    """Cancel the current search (without starting a new one)."""
    global _current_search_id
    with _search_id_lock:
        _current_search_id = None


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
_positional_boundaries_cache: dict = {"data": None, "mtime": None}
