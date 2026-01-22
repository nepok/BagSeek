"""
Completion tracking system for resumable pipeline processing.

Uses flat key structure based on relative paths for simplicity.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import fcntl

# Import for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .base import ProcessingContext


class CompletionTracker:
    """Manages completion.json files for tracking processed items"""
    
    def __init__(self, completion_file: Path):
        """
        Initialize completion tracker.
        
        Args:
            completion_file: Path to completion.json file
        """
        self.completion_file = Path(completion_file)
        self.completion_file.parent.mkdir(parents=True, exist_ok=True)
    
    def is_completed(self, context: "ProcessingContext", mcap_name: Optional[str] = None) -> bool:
        """
        Check if an item is marked as completed.
        
        Uses flat key structure: relative_path or relative_path/mcap_name
        
        Args:
            context: Processing context containing rosbag path
            mcap_name: Optional name of the mcap (for mcap-level processors)
        
        Returns:
            True if marked complete AND output exists, False otherwise
        """
        data = self._load()
        
        # Build key from relative path
        key = str(context.get_relative_path())
        if mcap_name:
            key = f"{key}/{mcap_name}"
        
        # Check if key exists
        if key not in data:
            return False
        
        entry = data[key]
        if not isinstance(entry, dict):
                return False
        
        # Accept both "complete" and "completed"
        status = entry.get("status")
        if status not in ("complete", "completed"):
                return False
        
        return self._validate_output(entry)
    
    def mark_completed(
        self, 
        context: "ProcessingContext",
        output_path: Path,
        mcap_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Mark an item as completed (incremental update).
        
        Uses flat key structure: relative_path or relative_path/mcap_name
        
        Args:
            context: Processing context containing rosbag path
            output_path: Path to the output file/folder
            mcap_name: Optional name of the mcap (for mcap-level processors)
            metadata: Optional additional metadata to store
        """
        data = self._load()
        
        # Build key from relative path
        key = str(context.get_relative_path())
        if mcap_name:
            key = f"{key}/{mcap_name}"
        
        # Calculate relative path from completion file's parent (the output_dir)
        if output_path.is_absolute():
            relative_output = output_path.relative_to(self.completion_file.parent)
        else:
            relative_output = output_path
        
        # Create entry
        entry = {
            "status": "completed",
            "completed_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "output_file": str(relative_output)
        }
        
        if metadata:
            entry.update(metadata)
        
        # Store with flat key
        data[key] = entry
        
        self._save(data)
    
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
        """Save completion data to disk atomically"""
        # Write to temporary file first
        temp_file = self.completion_file.with_suffix('.tmp')
        
        try:
            with open(temp_file, 'w') as f:
                # Lock the file for atomic write
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(data, f, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic rename
            temp_file.replace(self.completion_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def _validate_output(self, entry: Dict[str, Any]) -> bool:
        """
        Validate that output actually exists (fallback check).
        
        Args:
            entry: Completion entry dict containing output path info
        
        Returns:
            True if output exists, False otherwise
        """
        # Check both "output_path" and "output_file" for compatibility
        output_path = entry.get("output_path") or entry.get("output_file")
        if not output_path:
            return False
        
        # If it's a relative path, it's relative to the completion file's parent
        path = Path(output_path)
        if not path.is_absolute():
            path = self.completion_file.parent / output_path
        
        return path.exists()

