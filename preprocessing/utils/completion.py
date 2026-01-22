"""
Completion tracking system for resumable pipeline processing.

Uses flat key structure for rosbag-level tracking, and hierarchical structure
for mcap-level tracking (parent JSON per rosbag, mcap entries inside).
Works with Path objects directly or context objects that have get_relative_path().
"""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, Union
import fcntl


class CompletionTracker:
    """Manages completion.json files for tracking processed items"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize completion tracker.
        
        Args:
            output_dir: Directory where completion.json will be created
        """
        self.output_dir = Path(output_dir)
        self.completion_file = self.output_dir / "completion.json"
        self.completion_file.parent.mkdir(parents=True, exist_ok=True)
    
    def is_completed(
        self, 
        key_or_context: Union[str, Any], 
        output_path: Optional[Path] = None,
        mcap_name: Optional[str] = None,
        processor: Optional[Any] = None
    ) -> bool:
        """
        Check if an item is marked as completed.
        
        Automatically uses hierarchical structure when mcap_name is provided:
        - With mcap_name: parent key is rosbag path, child is mcap_name
        - Without mcap_name: flat structure with just the key
        
        If JSON entry is missing but output_path exists and is valid,
        automatically repairs the completion.json by marking it as completed.
        
        If output_path is not provided, tries to:
        1. Look up stored output_file from completion.json entry
        2. Get output_path from processor.get_output_path() if processor provided
        
        Accepts either a string key or a context object with get_relative_path() method.
        
        Args:
            key_or_context: Unique key string OR context object with get_relative_path() method
            output_path: Optional path to output file for validation and auto-repair
            mcap_name: Optional name of the mcap (for mcap-level processors)
            processor: Optional processor instance to get output_path from
        
        Returns:
            True if marked complete AND output exists (if output_path provided), False otherwise
        """
        data = self._load()
        
        # Extract key from context object or use string directly
        if isinstance(key_or_context, str):
            key = key_or_context
        else:
            # Assume it's a context object with get_relative_path() method
            key = str(key_or_context.get_relative_path())
        
        # Try to get output_path if not provided
        if output_path is None:
            # First, check if we have a stored entry with output_file
            stored_path = None
            if mcap_name:
                if key in data and isinstance(data[key], dict) and "status" not in data[key]:
                    if mcap_name in data[key]:
                        stored_entry = data[key][mcap_name]
                        if isinstance(stored_entry, dict):
                            stored_path_str = stored_entry.get("output_file")
                            if stored_path_str:
                                stored_path = self.completion_file.parent / stored_path_str
            else:
                if key in data:
                    stored_entry = data[key]
                    if isinstance(stored_entry, dict) and "status" in stored_entry:
                        stored_path_str = stored_entry.get("output_file")
                        if stored_path_str:
                            stored_path = self.completion_file.parent / stored_path_str
            
            # If still no path, try to get from processor
            if stored_path is None and processor is not None:
                if hasattr(processor, 'get_output_path'):
                    try:
                        stored_path = processor.get_output_path(key_or_context)
                    except Exception:
                        pass  # If processor can't provide path, continue without it
            
            output_path = stored_path
        
        # Handle hierarchical vs flat structure
        if mcap_name:
            # Hierarchical: key is rosbag path, entry contains mcap dict
            if key not in data:
                # No JSON entry - check fallback if output_path provided
                if output_path is not None and self._is_valid_output_file(output_path):
                    # Auto-repair: file exists but JSON entry missing
                    self.mark_completed(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            entry = data[key]
            if not isinstance(entry, dict):
                # Invalid entry structure - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            # Check if this is a completion entry or a parent dict
            if "status" in entry:
                # This is a flat entry, not hierarchical - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            # Look for mcap_name in the parent dict
            if mcap_name not in entry:
                # MCAP entry missing - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            mcap_entry = entry[mcap_name]
            if not isinstance(mcap_entry, dict):
                # Invalid mcap entry - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            status = mcap_entry.get("status")
            if status not in ("complete", "completed"):
                # Status not completed - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            # If output_path provided, validate it exists
            if output_path is not None:
                return self._validate_output(mcap_entry, output_path)
            
            return True
        else:
            # Flat structure: just use the key
            if key not in data:
                # No JSON entry - check fallback if output_path provided
                if output_path is not None and self._is_valid_output_file(output_path):
                    # Auto-repair: file exists but JSON entry missing
                    self.mark_completed(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            entry = data[key]
            if not isinstance(entry, dict):
                # Invalid entry - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            # Accept both "complete" and "completed"
            status = entry.get("status")
            if status not in ("complete", "completed"):
                # Status not completed - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            # If output_path provided, validate it exists
            if output_path is not None:
                return self._validate_output(entry, output_path)
            
            # Otherwise just check status
            return True
    
    def mark_completed(
        self, 
        key_or_context: Union[str, Any],
        output_path: Path,
        mcap_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Mark an item as completed (incremental update).
        
        Automatically uses hierarchical structure when mcap_name is provided:
        - With mcap_name: parent key is rosbag path, child is mcap_name
        - Without mcap_name: flat structure with just the key
        
        Accepts either a string key or a context object with get_relative_path() method.
        
        Args:
            key_or_context: Unique key string OR context object with get_relative_path() method
            output_path: Path to the output file/folder
            mcap_name: Optional name of the mcap (for mcap-level processors)
            metadata: Optional additional metadata to store
        """
        data = self._load()
        
        # Extract key from context object or use string directly
        if isinstance(key_or_context, str):
            key = key_or_context
        else:
            # Assume it's a context object with get_relative_path() method
            key = str(key_or_context.get_relative_path())
        
        # Calculate relative path from completion file's parent (the output_dir)
        if output_path.is_absolute():
            relative_output = output_path.relative_to(self.completion_file.parent)
        else:
            relative_output = output_path
        
        # Create entry
        entry = {
            "status": "completed",
            "completed_at": datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S"),
            "output_file": str(relative_output)
        }
        
        if metadata:
            entry.update(metadata)
        
        # Handle hierarchical vs flat structure
        if mcap_name:
            # Hierarchical: key is rosbag path, create/update parent dict
            if key not in data:
                data[key] = {}
            elif not isinstance(data[key], dict) or "status" in data[key]:
                # Convert existing flat entry to hierarchical structure
                # If old entry was a completion entry, we'll replace it with hierarchical
                data[key] = {}
            
            # Store mcap entry in parent dict
            data[key][mcap_name] = entry
        else:
            # Flat structure: just use the key
            data[key] = entry
        
        self._save(data)
    
    def mark_mcap_completed_with_model(
        self,
        model_name: str,
        rosbag_name: str,
        mcap_id: str,
        output_path: Path
    ) -> None:
        """
        Mark an MCAP as completed in a multi-level hierarchy structure.
        
        Creates/updates structure: {model_name: {rosbag_name: {mcaps: {mcap_id: {...}}, output_file: "..."}}}
        
        Args:
            model_name: Name of the model (e.g., "ViT-B-32-quickgelu__openai")
            rosbag_name: Name of the rosbag (e.g., "rosbag2_2025_07_25-12_17_25")
            mcap_id: MCAP identifier (e.g., "rosbag2_2025_07_25-12_17_25_0.mcap")
            output_path: Path to the manifest.parquet file (will be stored relative to completion.json)
        """
        data = self._load()
        
        # Calculate relative path from completion file's parent
        if output_path.is_absolute():
            relative_output = output_path.relative_to(self.completion_file.parent)
        else:
            relative_output = output_path
        
        # Create entry for this MCAP
        mcap_entry = {
            "status": "completed",
            "completed_at": datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Initialize model level if needed
        if model_name not in data:
            data[model_name] = {}
        
        # Initialize rosbag level if needed
        if rosbag_name not in data[model_name]:
            data[model_name][rosbag_name] = {}
        
        rosbag_entry = data[model_name][rosbag_name]
        
        # Initialize mcaps dict if needed
        if "mcaps" not in rosbag_entry:
            rosbag_entry["mcaps"] = {}
        
        # Add/update MCAP entry
        rosbag_entry["mcaps"][mcap_id] = mcap_entry
        
        # Store output_file at rosbag level (only if not already set)
        if "output_file" not in rosbag_entry:
            rosbag_entry["output_file"] = str(relative_output)
        
        self._save(data)
    
    def is_mcap_completed_with_model(
        self,
        model_name: str,
        rosbag_name: str,
        mcap_id: str
    ) -> bool:
        """
        Check if an MCAP is marked as completed in the multi-level hierarchy structure.
        
        Args:
            model_name: Name of the model
            rosbag_name: Name of the rosbag
            mcap_id: MCAP identifier
        
        Returns:
            True if the MCAP is marked as completed, False otherwise
        """
        data = self._load()
        
        if model_name not in data:
            return False
        
        if rosbag_name not in data[model_name]:
            return False
        
        rosbag_entry = data[model_name][rosbag_name]
        
        if "mcaps" not in rosbag_entry:
            return False
        
        if mcap_id not in rosbag_entry["mcaps"]:
            return False
        
        mcap_entry = rosbag_entry["mcaps"][mcap_id]
        
        if not isinstance(mcap_entry, dict):
            return False
        
        status = mcap_entry.get("status")
        return status in ("complete", "completed")
    
    def _load(self) -> Dict:
        """Load completion data from disk"""
        if not self.completion_file.exists():
            return {}
        
        try:
            with open(self.completion_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save(self, data: Dict):
        """Save completion data to disk atomically with numeric key sorting"""
        # Write to temporary file first
        temp_file = self.completion_file.with_suffix('.tmp')
        
        try:
            with open(temp_file, 'w') as f:
                # Lock the file for atomic write
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                # Sort keys numerically before saving
                sorted_data = self._sort_keys_numerically(data)
                json.dump(sorted_data, f, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic rename
            temp_file.replace(self.completion_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def _sort_keys_numerically(self, obj: Any) -> Any:
        """
        Recursively sort dictionary keys numerically where possible.
        
        Keys that are numeric strings (e.g., "0", "1", "10") are sorted numerically.
        Other keys are sorted lexicographically.
        
        Args:
            obj: Dictionary, list, or other value to sort
        
        Returns:
            Object with keys sorted numerically where applicable
        """
        if isinstance(obj, dict):
            # Sort keys: try numeric sort first, fallback to lexicographic
            def sort_key(key: str) -> tuple:
                """Return tuple for sorting: (is_numeric, numeric_value, string_value)"""
                try:
                    # Try to convert to int for numeric sorting
                    return (0, int(key), '')
                except ValueError:
                    # Not numeric, sort lexicographically
                    return (1, 0, key)
            
            sorted_items = sorted(obj.items(), key=lambda x: sort_key(str(x[0])))
            return {k: self._sort_keys_numerically(v) for k, v in sorted_items}
        elif isinstance(obj, list):
            return [self._sort_keys_numerically(item) for item in obj]
        else:
            return obj
    
    def _is_valid_output_file(self, output_path: Path) -> bool:
        """
        Check if output file exists and is valid (not empty).
        
        Args:
            output_path: Path to the output file to validate
        
        Returns:
            True if file exists and appears valid, False otherwise
        """
        if not output_path.exists():
            return False
        
        # Check if it's a file (not a directory)
        if not output_path.is_file():
            return False
        
        # Check if file is not empty
        if output_path.stat().st_size == 0:
            return False
        
        # Basic validation based on file extension
        suffix = output_path.suffix.lower()
        
        if suffix == '.csv':
            # For CSV, check if it has at least 2 lines (header + 1 data row)
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) < 2:
                        return False
            except (IOError, UnicodeDecodeError):
                return False
        
        elif suffix == '.json':
            # For JSON, check if it's valid JSON
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    json.load(f)
            except (json.JSONDecodeError, IOError, UnicodeDecodeError):
                return False
        
        # For other file types, just check existence and non-empty
        return True
    
    def _validate_output(self, entry: Dict[str, Any], output_path: Path) -> bool:
        """
        Validate that output actually exists (fallback check).
        
        Args:
            entry: Completion entry dict containing output path info
            output_path: Path to validate
        
        Returns:
            True if output exists, False otherwise
        """
        # First check if the provided output_path exists
        if output_path.exists():
            return True
        
        # Fallback: check path from entry
        output_file = entry.get("output_path") or entry.get("output_file")
        if not output_file:
            return False
        
        # If it's a relative path, it's relative to the completion file's parent
        path = Path(output_file)
        if not path.is_absolute():
            path = self.completion_file.parent / output_file
        
        return path.exists()

