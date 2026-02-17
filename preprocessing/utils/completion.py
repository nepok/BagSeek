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
from typing import Dict, Any, Optional, Union, List
import fcntl


class CompletionTracker:
    """Manages completion.json files for tracking processed items"""
    
    def __init__(self, output_dir: Path, processor_name: str):
        """
        Initialize completion tracker.
        
        Args:
            output_dir: Directory where completion.json will be created
            processor_name: Name of the processor (e.g., "topics_extraction_processor")
        """
        self.output_dir = Path(output_dir)
        self.processor_name = processor_name
        self.completion_file = self.output_dir / "completion.json"
        self.completion_file.parent.mkdir(parents=True, exist_ok=True)
    
    def is_completed_old(
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
                    self.mark_completed_old(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            entry = data[key]
            if not isinstance(entry, dict):
                # Invalid entry structure - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed_old(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            # Check if this is a completion entry or a parent dict
            if "status" in entry:
                # This is a flat entry, not hierarchical - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed_old(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            # Look for mcap_name in the parent dict
            if mcap_name not in entry:
                # MCAP entry missing - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed_old(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            mcap_entry = entry[mcap_name]
            if not isinstance(mcap_entry, dict):
                # Invalid mcap entry - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed_old(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            status = mcap_entry.get("status")
            if status not in ("complete", "completed"):
                # Status not completed - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed_old(key_or_context, output_path, mcap_name=mcap_name)
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
                    self.mark_completed_old(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            entry = data[key]
            if not isinstance(entry, dict):
                # Invalid entry - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed_old(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            # Accept both "complete" and "completed"
            status = entry.get("status")
            if status not in ("complete", "completed"):
                # Status not completed - check fallback
                if output_path is not None and self._is_valid_output_file(output_path):
                    self.mark_completed_old(key_or_context, output_path, mcap_name=mcap_name)
                    return True
                return False
            
            # If output_path provided, validate it exists
            if output_path is not None:
                return self._validate_output(entry, output_path)
            
            # Otherwise just check status
            return True
    
    def mark_completed_old(
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
        """
        Load completion data from disk.
        
        Returns the full completion.json structure. For new structure, returns
        data[processor_name] entry. For backward compatibility, also checks old structure.
        """
        if not self.completion_file.exists():
            return {}
        
        try:
            with open(self.completion_file, 'r') as f:
                data = json.load(f)
            
            # Check if this is new structure (processor_name as top-level key)
            if self.processor_name in data:
                return data[self.processor_name]

            # If other known processor keys exist, this is the new multi-processor
            # format and our processor simply hasn't been added yet
            known_processors = {
                "positional_lookup_processor", "positional_boundaries_processor",
                "topics_extraction_processor", "timestamp_alignment_processor",
                "image_topic_previews_processor", "embeddings_processor",
                "adjacent_similarities_postprocessor",
            }
            if any(k in known_processors for k in data):
                return {}

            # Old structure - return full data for backward compatibility
            return data
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _load_full(self) -> Dict:
        """Load the full completion.json file (for saving)."""
        if not self.completion_file.exists():
            return {}
        
        try:
            with open(self.completion_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save(self, processor_data: Dict):
        """
        Save completion data to disk atomically with numeric key sorting.
        
        Ensures proper field ordering and MCAP sorting before saving.
        
        Args:
            processor_data: Data for this processor (will be stored under processor_name key)
        """
        # Write to temporary file first
        temp_file = self.completion_file.with_suffix('.tmp')
        
        try:
            # Load full file to preserve other processors' data
            full_data = self._load_full()
            
            # Ensure proper field ordering and MCAP sorting for this processor's data
            processor_data = self._ensure_processor_data_order(processor_data)
            
            # Update this processor's entry
            full_data[self.processor_name] = processor_data
            
            with open(temp_file, 'w') as f:
                # Lock the file for atomic write
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                # Sort keys numerically before saving (only for numeric keys like MCAP IDs)
                sorted_data = self._sort_keys_numerically(full_data)
                json.dump(sorted_data, f, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic rename
            temp_file.replace(self.completion_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def _ensure_processor_data_order(self, processor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure proper field ordering for processor-level data.
        
        Order: status, output_files (if positional_lookup_processor), rosbags, models
        
        Also updates output_files for embeddings_processor rosbags if all MCAPs are completed.
        
        Args:
            processor_data: Processor-level data dictionary
        
        Returns:
            Processor data with proper field ordering and updated output_files
        """
        if not isinstance(processor_data, dict):
            return processor_data
        
        # First, update output_files for embeddings_processor rosbags
        if self.processor_name == "embeddings_processor" and "models" in processor_data:
            models = processor_data["models"]
            if isinstance(models, dict):
                for model_name, model_entry in models.items():
                    if isinstance(model_entry, dict) and "rosbags" in model_entry:
                        rosbags = model_entry["rosbags"]
                        if isinstance(rosbags, dict):
                            for rosbag_name, rosbag_entry in rosbags.items():
                                if isinstance(rosbag_entry, dict) and "mcaps" in rosbag_entry:
                                    # Update output_files if all MCAPs completed
                                    updated_entry = self._update_rosbag_output_files_if_completed(
                                        rosbag_entry, model_name, rosbag_name
                                    )
                                    rosbags[rosbag_name] = updated_entry
        
        ordered_data: Dict[str, Any] = {}
        
        # 1. status (always first)
        if "status" in processor_data:
            ordered_data["status"] = processor_data["status"]
        
        # 2. completed_at (if present)
        if "completed_at" in processor_data:
            ordered_data["completed_at"] = processor_data["completed_at"]
        
        # 3. output_files (for positional_lookup_processor, before rosbags)
        if "output_files" in processor_data:
            ordered_data["output_files"] = processor_data["output_files"]
        
        # 4. rosbags (if present, with proper ordering for each rosbag)
        if "rosbags" in processor_data:
            rosbags = processor_data["rosbags"]
            if isinstance(rosbags, dict):
                ordered_rosbags = {}
                for rosbag_name, rosbag_entry in rosbags.items():
                    ordered_rosbags[rosbag_name] = self._ensure_rosbag_entry_order(rosbag_entry)
                ordered_data["rosbags"] = ordered_rosbags
        
        # 5. models (if present)
        if "models" in processor_data:
            models = processor_data["models"]
            if isinstance(models, dict):
                ordered_models = {}
                for model_name, model_entry in models.items():
                    ordered_model_entry = {}
                    # Model-level: status, rosbags
                    if "status" in model_entry:
                        ordered_model_entry["status"] = model_entry["status"]
                    if "completed_at" in model_entry:
                        ordered_model_entry["completed_at"] = model_entry["completed_at"]
                    if "rosbags" in model_entry:
                        rosbags = model_entry["rosbags"]
                        if isinstance(rosbags, dict):
                            ordered_rosbags = {}
                            for rosbag_name, rosbag_entry in rosbags.items():
                                ordered_rosbags[rosbag_name] = self._ensure_rosbag_entry_order(rosbag_entry)
                            ordered_model_entry["rosbags"] = ordered_rosbags
                    ordered_models[model_name] = ordered_model_entry
                ordered_data["models"] = ordered_models
        
        # Include any other fields
        for key, value in processor_data.items():
            if key not in ordered_data:
                ordered_data[key] = value
        
        return ordered_data
    
    def _sort_keys_numerically(self, obj: Any) -> Any:
        """
        Recursively sort dictionary keys numerically where possible.
        
        Preserves field order for structured entries (status, completed_at, etc.)
        but sorts numeric keys (like MCAP IDs) numerically.
        
        Args:
            obj: Dictionary, list, or other value to sort
        
        Returns:
            Object with keys sorted numerically where applicable
        """
        if isinstance(obj, dict):
            # Check if this looks like a structured entry (has "status" or "mcaps")
            # In that case, preserve order but sort nested numeric keys
            if "status" in obj or "mcaps" in obj or "topics" in obj or "rosbags" in obj or "models" in obj:
                # This is a structured entry - preserve order, but recursively sort nested structures
                return {k: self._sort_keys_numerically(v) for k, v in obj.items()}
            
            # For other dictionaries, sort keys: try numeric sort first, fallback to lexicographic
            def sort_key(key: str) -> tuple:
                """Return tuple for sorting: (is_numeric, numeric_value, string_value)"""
                # Check if this is an MCAP name (ends with .mcap or is a numeric ID)
                key_str = str(key)
                if key_str.endswith(".mcap"):
                    # Extract numeric ID from MCAP name
                    stem = key_str[:-5]
                    parts = stem.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        return (0, int(parts[1]), '')
                
                try:
                    # Try to convert to int for numeric sorting
                    return (0, int(key_str), '')
                except ValueError:
                    # Not numeric, sort lexicographically
                    return (1, 0, key_str)
            
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
    
    # ========================================================================
    # NEW UNIFIED INTERFACE METHODS
    # ========================================================================
    
    def mark_completed(
        self,
        rosbag_name: Optional[str] = None,
        mcap_name: Optional[str] = None,
        model_name: Optional[str] = None,
        topic_name: Optional[str] = None,
        status: str = "completed",
        output_files: Optional[Union[Path, List[Path]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Unified method to mark completion at any level.
        
        The tracker automatically determines the structure based on provided parameters:
        - Only rosbag_name: creates processor -> rosbags[rosbag_name]
        - rosbag_name + mcap_name: creates processor -> rosbags[rosbag_name].mcaps[mcap_name]
        - model_name + rosbag_name + mcap_name: creates processor -> models[model_name].rosbags[rosbag_name].mcaps[mcap_name]
        - model_name + rosbag_name + topic_name: creates processor -> models[model_name].rosbags[rosbag_name].topics[topic_name]
        
        Ensures proper field ordering: status, completed_at, output_files, then mcaps/topics.
        For positional_lookup_processor, output_files are moved to processor level.
        
        Args:
            rosbag_name: Name of the rosbag (required for all processors)
            mcap_name: Name of the MCAP (optional, for MCAP-level processors)
            model_name: Name of the model (optional, for model-level processors)
            topic_name: Name of the topic (optional, for AdjacentSimilaritiesPostprocessor)
            status: "pending", "in_progress", or "completed"
            output_files: Single Path, list of Paths, or None
            metadata: Optional additional metadata
        """
        if rosbag_name is None:
            raise ValueError("rosbag_name is required")
        
        data = self._load()
        
        # Convert single Path to list
        if output_files is not None and isinstance(output_files, Path):
            output_files = [output_files]
        
        # Convert paths to relative paths and strings
        output_files_list = []
        if output_files:
            for path in output_files:
                path_obj = Path(path)
                if path_obj.is_absolute():
                    try:
                        rel_path = path_obj.relative_to(self.completion_file.parent)
                    except ValueError:
                        # Path is not relative to completion file parent, use as-is
                        rel_path = path_obj
                else:
                    rel_path = path_obj
                output_files_list.append(str(rel_path))
        
        # Special handling for processors where all rosbags write to the same file:
        # output_files go to processor level (written once, not per rosbag)
        uses_shared_output = self.processor_name in (
            "positional_lookup_processor",
            "positional_boundaries_processor",
        )
        processor_level_output_files = None
        if uses_shared_output and output_files_list and not mcap_name:
            processor_level_output_files = output_files_list
            output_files_list = []  # Don't put at rosbag level
        
        # Create entry dict with proper field ordering
        entry: Dict[str, Any] = {
            "status": status
        }
        
        # Only include completed_at and output_files when status="completed"
        if status == "completed":
            entry["completed_at"] = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S")
            if output_files_list:
                entry["output_files"] = output_files_list
        
        if metadata:
            entry.update(metadata)
        
        # Navigate/create structure based on parameters
        if topic_name is not None:
            # Structure: models[model_name].rosbags[rosbag_name].topics[topic_name]
            if model_name is None:
                raise ValueError("model_name is required when topic_name is provided")
            
            if "models" not in data:
                data["models"] = {}
            if model_name not in data["models"]:
                data["models"][model_name] = {}
            if "rosbags" not in data["models"][model_name]:
                data["models"][model_name]["rosbags"] = {}
            if rosbag_name not in data["models"][model_name]["rosbags"]:
                data["models"][model_name]["rosbags"][rosbag_name] = {}
            if "topics" not in data["models"][model_name]["rosbags"][rosbag_name]:
                data["models"][model_name]["rosbags"][rosbag_name]["topics"] = {}
            
            data["models"][model_name]["rosbags"][rosbag_name]["topics"][topic_name] = entry
            
        elif mcap_name is not None:
            if model_name is not None:
                # Structure: models[model_name].rosbags[rosbag_name].mcaps[mcap_name]
                if "models" not in data:
                    data["models"] = {}
                if model_name not in data["models"]:
                    data["models"][model_name] = {}
                if "rosbags" not in data["models"][model_name]:
                    data["models"][model_name]["rosbags"] = {}
                if rosbag_name not in data["models"][model_name]["rosbags"]:
                    data["models"][model_name]["rosbags"][rosbag_name] = {}
                if "mcaps" not in data["models"][model_name]["rosbags"][rosbag_name]:
                    data["models"][model_name]["rosbags"][rosbag_name]["mcaps"] = {}
                
                data["models"][model_name]["rosbags"][rosbag_name]["mcaps"][mcap_name] = entry
                
                # Check if all MCAPs are completed and update rosbag-level output_files
                rosbag_entry = data["models"][model_name]["rosbags"][rosbag_name]
                rosbag_entry = self._update_rosbag_output_files_if_completed(
                    rosbag_entry, model_name, rosbag_name
                )
                rosbag_entry = self._ensure_rosbag_entry_order(rosbag_entry)
                data["models"][model_name]["rosbags"][rosbag_name] = rosbag_entry
            else:
                # Structure: rosbags[rosbag_name].mcaps[mcap_name]
                if "rosbags" not in data:
                    data["rosbags"] = {}
                if rosbag_name not in data["rosbags"]:
                    data["rosbags"][rosbag_name] = {}
                if "mcaps" not in data["rosbags"][rosbag_name]:
                    data["rosbags"][rosbag_name]["mcaps"] = {}
                
                data["rosbags"][rosbag_name]["mcaps"][mcap_name] = entry
                
                # Ensure proper field ordering for rosbag entry with mcaps
                rosbag_entry = data["rosbags"][rosbag_name]
                rosbag_entry = self._ensure_rosbag_entry_order(rosbag_entry)
                data["rosbags"][rosbag_name] = rosbag_entry
                
        elif model_name is not None:
            # Structure: models[model_name].rosbags[rosbag_name]
            if "models" not in data:
                data["models"] = {}
            if model_name not in data["models"]:
                data["models"][model_name] = {}
            if "rosbags" not in data["models"][model_name]:
                data["models"][model_name]["rosbags"] = {}
            
            # Check if this rosbag entry already exists and has mcaps
            existing_entry = data["models"][model_name]["rosbags"].get(rosbag_name, {})
            if isinstance(existing_entry, dict) and "mcaps" in existing_entry:
                # Merge with existing entry, preserving mcaps
                existing_entry.update(entry)
                entry = existing_entry
                # Update output_files if all MCAPs are completed
                entry = self._update_rosbag_output_files_if_completed(
                    entry, model_name, rosbag_name
                )
            
            data["models"][model_name]["rosbags"][rosbag_name] = entry
            # Ensure proper field ordering
            if isinstance(entry, dict) and "mcaps" in entry:
                entry = self._ensure_rosbag_entry_order(entry)
                data["models"][model_name]["rosbags"][rosbag_name] = entry
            
        else:
            # Structure: rosbags[rosbag_name]
            if "rosbags" not in data:
                data["rosbags"] = {}
            
            data["rosbags"][rosbag_name] = entry
        
        # Handle processor-level output_files for positional_lookup_processor
        if processor_level_output_files:
            if "output_files" not in data or not isinstance(data.get("output_files"), list):
                data["output_files"] = []
            # Merge with existing output_files, avoiding duplicates
            existing_output_files = set(data.get("output_files", []))
            for output_file in processor_level_output_files:
                if output_file not in existing_output_files:
                    data["output_files"].append(output_file)
        
        self._save(data)
    
    def _update_rosbag_output_files_if_completed(
        self, rosbag_entry: Dict[str, Any], model_name: str, rosbag_name: str
    ) -> Dict[str, Any]:
        """
        Update rosbag-level output_files if all MCAPs are completed.
        For embeddings_processor, collects all shard files from the shards directory.
        
        Args:
            rosbag_entry: Rosbag entry dictionary
            model_name: Model name (for embeddings processor)
            rosbag_name: Rosbag name
        
        Returns:
            Updated rosbag entry with output_files if all MCAPs completed
        """
        if not isinstance(rosbag_entry, dict) or "mcaps" not in rosbag_entry:
            return rosbag_entry
        
        mcaps = rosbag_entry["mcaps"]
        if not isinstance(mcaps, dict) or not mcaps:
            return rosbag_entry
        
        # Check if all MCAPs are completed
        all_completed = all(
            isinstance(mcap_entry, dict) and mcap_entry.get("status") == "completed"
            for mcap_entry in mcaps.values()
        )
        
        if not all_completed:
            return rosbag_entry
        
        # Only update output_files if not already set or if we need to add shard files
        if self.processor_name == "embeddings_processor":
            # Collect shard files from shards directory
            # Shards are in: {completion_file.parent}/{model_name}/{rosbag_name}/shards/
            shards_dir = self.completion_file.parent / model_name / rosbag_name / "shards"
            
            output_files = rosbag_entry.get("output_files", [])
            output_files_set = set(output_files)
            
            # Add manifest.parquet if not already present
            manifest_path = self.completion_file.parent / model_name / rosbag_name / "manifest.parquet"
            if manifest_path.exists():
                rel_manifest = manifest_path.relative_to(self.completion_file.parent)
                if str(rel_manifest) not in output_files_set:
                    output_files.append(str(rel_manifest))
                    output_files_set.add(str(rel_manifest))
            
            # Add meta.json if manifest exists
            meta_path = self.completion_file.parent / model_name / rosbag_name / "meta.json"
            if meta_path.exists():
                rel_meta = meta_path.relative_to(self.completion_file.parent)
                if str(rel_meta) not in output_files_set:
                    output_files.append(str(rel_meta))
                    output_files_set.add(str(rel_meta))
            
            # Collect all shard files
            if shards_dir.exists() and shards_dir.is_dir():
                shard_files = sorted(shards_dir.glob("shard-*.npy"))
                for shard_file in shard_files:
                    rel_shard = shard_file.relative_to(self.completion_file.parent)
                    if str(rel_shard) not in output_files_set:
                        output_files.append(str(rel_shard))
                        output_files_set.add(str(rel_shard))
            
            # Update rosbag entry with output_files
            rosbag_entry["output_files"] = output_files
            
            # Update completed_at if not present (use latest from MCAPs)
            if "completed_at" not in rosbag_entry:
                latest_completed_at = None
                for mcap_entry in mcaps.values():
                    if isinstance(mcap_entry, dict) and "completed_at" in mcap_entry:
                        mcap_at = mcap_entry["completed_at"]
                        if latest_completed_at is None or mcap_at > latest_completed_at:
                            latest_completed_at = mcap_at
                if latest_completed_at:
                    rosbag_entry["completed_at"] = latest_completed_at
            
            # Update status to completed
            rosbag_entry["status"] = "completed"
        
        return rosbag_entry
    
    def _ensure_rosbag_entry_order(self, rosbag_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure proper field ordering for rosbag entries: status, completed_at, output_files, then mcaps/topics.
        Also ensures MCAPs are sorted numerically.
        
        Args:
            rosbag_entry: Rosbag entry dictionary (may have mcaps or topics)
        
        Returns:
            Rosbag entry with proper field ordering and sorted MCAPs
        """
        if not isinstance(rosbag_entry, dict):
            return rosbag_entry
        
        # Extract fields in correct order
        ordered_entry: Dict[str, Any] = {}
        
        # 1. status (always first)
        if "status" in rosbag_entry:
            ordered_entry["status"] = rosbag_entry["status"]
        
        # 2. completed_at (if present)
        if "completed_at" in rosbag_entry:
            ordered_entry["completed_at"] = rosbag_entry["completed_at"]
        
        # 3. output_files (if present)
        if "output_files" in rosbag_entry:
            ordered_entry["output_files"] = rosbag_entry["output_files"]
        
        # 4. mcaps or topics (always last, sorted numerically)
        if "mcaps" in rosbag_entry:
            # Sort MCAPs numerically
            mcaps = rosbag_entry["mcaps"]
            if isinstance(mcaps, dict):
                sorted_mcaps = self._sort_mcaps_numerically(mcaps)
                ordered_entry["mcaps"] = sorted_mcaps
        
        if "topics" in rosbag_entry:
            ordered_entry["topics"] = rosbag_entry["topics"]
        
        # Include any other fields (shouldn't happen, but be safe)
        for key, value in rosbag_entry.items():
            if key not in ordered_entry:
                ordered_entry[key] = value
        
        return ordered_entry
    
    def _sort_mcaps_numerically(self, mcaps: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sort MCAP dictionary keys numerically (0, 1, 2, ...) instead of lexicographically (0, 1, 10, ...).
        
        Args:
            mcaps: Dictionary of MCAP entries
        
        Returns:
            Dictionary with numerically sorted keys
        """
        if not isinstance(mcaps, dict):
            return mcaps
        
        def sort_key(key: str) -> tuple:
            """Return tuple for sorting: (is_numeric, numeric_value, string_value)"""
            # Try to extract numeric part from MCAP name (e.g., "rosbag2_xxx_0.mcap" -> 0)
            # First try to find trailing number before .mcap
            if key.endswith(".mcap"):
                base = key[:-5]  # Remove ".mcap"
                # Try to find trailing number
                parts = base.split("_")
                if parts:
                    try:
                        # Last part might be the number
                        num = int(parts[-1])
                        return (0, num, key)
                    except ValueError:
                        pass
            
            # Fallback: try to convert entire key to int
            try:
                return (0, int(key), '')
            except ValueError:
                # Not numeric, sort lexicographically
                return (1, 0, key)
        
        sorted_items = sorted(mcaps.items(), key=lambda x: sort_key(str(x[0])))
        return {k: self._sort_keys_numerically(v) for k, v in sorted_items}
    
    def is_completed(
        self,
        rosbag_name: Optional[str] = None,
        mcap_name: Optional[str] = None,
        model_name: Optional[str] = None,
        topic_name: Optional[str] = None
    ) -> bool:
        """
        Unified method to check completion at any level.
        
        Returns True if the item is marked as "completed" in the structure.
        Automatically navigates the structure based on provided parameters.
        
        Args:
            rosbag_name: Name of the rosbag (required for all processors)
            mcap_name: Name of the MCAP (optional, for MCAP-level processors)
            model_name: Name of the model (optional, for model-level processors)
            topic_name: Name of the topic (optional, for AdjacentSimilaritiesPostprocessor)
        
        Returns:
            True if marked as "completed", False otherwise
        """
        if rosbag_name is None:
            return False
        
        data = self._load()
        
        # Navigate structure based on parameters
        try:
            if topic_name is not None:
                # Structure: models[model_name].rosbags[rosbag_name].topics[topic_name]
                if model_name is None:
                    return False
                entry = data.get("models", {}).get(model_name, {}).get("rosbags", {}).get(rosbag_name, {}).get("topics", {}).get(topic_name)
                
            elif mcap_name is not None:
                if model_name is not None:
                    # Structure: models[model_name].rosbags[rosbag_name].mcaps[mcap_name]
                    entry = data.get("models", {}).get(model_name, {}).get("rosbags", {}).get(rosbag_name, {}).get("mcaps", {}).get(mcap_name)
                else:
                    # Structure: rosbags[rosbag_name].mcaps[mcap_name]
                    entry = data.get("rosbags", {}).get(rosbag_name, {}).get("mcaps", {}).get(mcap_name)
                    
            elif model_name is not None:
                # Structure: models[model_name].rosbags[rosbag_name]
                entry = data.get("models", {}).get(model_name, {}).get("rosbags", {}).get(rosbag_name)
                
            else:
                # Structure: rosbags[rosbag_name]
                entry = data.get("rosbags", {}).get(rosbag_name)
            
            if not isinstance(entry, dict):
                return False
            
            status = entry.get("status")
            return status == "completed"
            
        except (AttributeError, KeyError, TypeError):
            return False
    
    def is_rosbag_completed(self, rosbag_name: str) -> bool:
        """Check if rosbag is completed (rosbag-level only processors)."""
        return self.is_completed(rosbag_name=rosbag_name)
    
    def is_mcap_completed(self, rosbag_name: str, mcap_name: str) -> bool:
        """Check if MCAP is completed (MCAP-level processors)."""
        return self.is_completed(rosbag_name=rosbag_name, mcap_name=mcap_name)
    
    def is_model_rosbag_completed(self, model_name: str, rosbag_name: str) -> bool:
        """Check if model+rosbag is completed."""
        return self.is_completed(model_name=model_name, rosbag_name=rosbag_name)
    
    def is_model_mcap_completed(self, model_name: str, rosbag_name: str, mcap_name: str) -> bool:
        """Check if model+MCAP is completed (embeddings processor)."""
        return self.is_completed(model_name=model_name, rosbag_name=rosbag_name, mcap_name=mcap_name)
    
    def is_model_topic_completed(self, model_name: str, rosbag_name: str, topic_name: str) -> bool:
        """Check if model+topic is completed (AdjacentSimilaritiesPostprocessor)."""
        return self.is_completed(model_name=model_name, rosbag_name=rosbag_name, topic_name=topic_name)
    
    def mark_processor_status(self, status: str) -> None:
        """Set the top-level processor status."""
        data = self._load()
        data["status"] = status
        if status == "completed":
            data["completed_at"] = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S")
        self._save(data)

