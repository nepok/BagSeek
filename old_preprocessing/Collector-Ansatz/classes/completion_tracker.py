# completion_tracker.py
"""
Reusable completion tracking utility for collectors.

Each collector can use this to track which rosbags have been processed,
with support for both normal and multi-part rosbags.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Callable

# Create logger for completion tracker
logger = logging.getLogger("CompletionTracker")


class CompletionTracker:
    """
    Tracks completion status for collectors.
    
    Each collector should create its own CompletionTracker instance pointing
    to its output directory. The tracker manages a completion.json file
    in that directory.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize completion tracker.
        
        Args:
            output_dir: Directory where completion.json will be stored
                       (typically the collector's output directory)
        """
        self.output_dir = Path(output_dir)
        self.completion_file = self.output_dir / "completion.json"
        self._completion_data = None
    
    def _load_completion(self) -> dict:
        """Load completion data from JSON file."""
        if self._completion_data is not None:
            return self._completion_data
        
        if not self.completion_file.exists():
            self._completion_data = {}
            return self._completion_data
        
        try:
            with open(self.completion_file, 'r', encoding='utf-8') as f:
                self._completion_data = json.load(f)
            return self._completion_data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load completion file {self.completion_file}: {e}")
            self._completion_data = {}
            return self._completion_data
    
    def _save_completion(self, data: dict):
        """Save completion data to JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.completion_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self._completion_data = data
        except IOError as e:
            logger.error(f"Failed to save completion file {self.completion_file}: {e}")
            raise
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in readable format."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_rosbag_key(self, rosbag_path: Path) -> Tuple[str, Optional[str]]:
        """
        Extract base name and part identifier from rosbag path.
        
        Args:
            rosbag_path: Path to rosbag directory
            
        Returns:
            Tuple of (base_name, part_identifier)
            - For normal rosbags: (rosbag_name, None)
            - For multi-part rosbags: (base_name, "Part_N")
        """
        if rosbag_path.name.startswith("Part_"):
            # Multi-part rosbag: extract base name from parent
            parent_name = rosbag_path.parent.name
            base_name = parent_name.replace("_multi_parts", "")
            part_identifier = rosbag_path.name  # e.g., "Part_1"
            return (base_name, part_identifier)
        else:
            # Normal rosbag
            return (rosbag_path.name, None)
    
    def is_completed(self, rosbag_path: Path, output_file: Path = None) -> bool:
        """
        Check if rosbag is already completed.
        
        Checks both completion.json and verifies output file exists.
        
        Args:
            rosbag_path: Path to rosbag directory
            output_file: Optional path to expected output file (for verification)
            
        Returns:
            True if completed, False otherwise
        """
        completion = self._load_completion()
        base_name, part_identifier = self._get_rosbag_key(rosbag_path)
        
        # Check completion.json
        if base_name not in completion:
            return False
        
        entry = completion[base_name]
        
        # Handle multi-part vs normal rosbag
        if part_identifier is not None:
            # Multi-part rosbag: check Part_N entry
            if part_identifier not in entry:
                return False
            part_entry = entry[part_identifier]
            if part_entry.get("status") != "completed":
                return False
            # Verify output file exists if provided
            if output_file is not None:
                if not output_file.exists():
                    logger.warning(f"Completion marked but output file missing: {output_file}")
                    return False
            return True
        else:
            # Normal rosbag: check direct entry
            if entry.get("status") != "completed":
                return False
            # Verify output file exists if provided
            if output_file is not None:
                if not output_file.exists():
                    logger.warning(f"Completion marked but output file missing: {output_file}")
                    return False
            return True
    
    def mark_completed(self, rosbag_path: Path, output_file: Path):
        """
        Mark rosbag as completed in completion.json.
        
        Args:
            rosbag_path: Path to rosbag directory
            output_file: Path to output file (relative to output_dir or absolute)
        """
        completion = self._load_completion()
        base_name, part_identifier = self._get_rosbag_key(rosbag_path)
        
        # Make output_file path relative to output_dir if it's absolute
        if output_file.is_absolute():
            try:
                output_file_rel = str(output_file.relative_to(self.output_dir))
            except ValueError:
                # Output file is not under output_dir, use full path as string
                output_file_rel = str(output_file)
        else:
            output_file_rel = str(output_file)
        
        # Initialize base entry if needed
        if base_name not in completion:
            completion[base_name] = {}
        
        entry = completion[base_name]
        
        # Handle multi-part vs normal rosbag
        if part_identifier is not None:
            # Multi-part rosbag: create Part_N entry
            entry[part_identifier] = {
                "status": "completed",
                "completed_at": self._get_timestamp(),
                "output_file": output_file_rel
            }
        else:
            # Normal rosbag: update direct entry
            entry["status"] = "completed"
            entry["completed_at"] = self._get_timestamp()
            entry["output_file"] = output_file_rel
        
        self._save_completion(completion)
        logger.debug(f"Marked {rosbag_path.name} as completed")
    
    def get_pending_rosbags(
        self, 
        rosbags_dir: Path, 
        get_rosbags_func: Callable[[Path], List[Path]],
        output_file_pattern: Optional[Callable[[Path], Path]] = None
    ) -> List[Path]:
        """
        Filter rosbags to only those that need processing.
        
        Args:
            rosbags_dir: Directory containing rosbags
            get_rosbags_func: Function that returns list of rosbag paths
                             (e.g., get_rosbags from main.py)
            output_file_pattern: Optional function that takes rosbag_path and returns
                                expected output_file path (for file existence verification)
            
        Returns:
            List of rosbag paths that are not yet completed
        """
        all_rosbags = get_rosbags_func(rosbags_dir)
        pending = []
        
        for rosbag in all_rosbags:
            # Check completion status
            output_file = None
            if output_file_pattern:
                output_file = output_file_pattern(rosbag)
            
            if not self.is_completed(rosbag, output_file=output_file):
                pending.append(rosbag)
        
        logger.info(f"Found {len(pending)} pending rosbags out of {len(all_rosbags)} total")
        return pending
    
    def scan_existing_outputs(self, output_file_pattern: Callable[[Path], Path]) -> dict:
        """
        Scan output directory for existing files and update completion.json.
        
        This is useful for initializing completion.json from existing output files.
        
        Args:
            output_file_pattern: Function that takes rosbag_path and returns expected output_file path
            
        Returns:
            Dictionary of scanned completion data
        """
        completion = self._load_completion()
        scanned_count = 0
        
        # This method would need to know how to discover rosbags
        # For now, it's a placeholder that can be extended
        # The actual scanning would be done by iterating over known rosbags
        # and checking if their output files exist
        
        logger.info(f"Scanned {scanned_count} existing outputs")
        return completion

