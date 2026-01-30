"""
Step 3: Positional Lookup

Builds positional lookup tables from position/GPS messages.
Hybrid processor that collects per MCAP and aggregates per rosbag.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from ..abstract import HybridProcessor
from ..core import McapProcessingContext, RosbagProcessingContext
from ..utils import CompletionTracker, PipelineLogger, get_logger


class PositionalLookupProcessor(HybridProcessor):
    """
    Build aggregated positional lookup table for a rosbag.
    
    Hybrid processor that:
    - Collects position messages during MCAP iteration
    - Aggregates positions per rosbag after all MCAPs
    - Finalizes by combining all rosbag JSONs into one big file
    """
    
    def __init__(self, output_dir: Path, positional_grid_resolution: float = 0.0001):
        """
        Initialize positional lookup step.
        
        Args:
            output_dir: Full path to positional lookup table JSON file
            positional_grid_resolution: Step size in degrees (for spatial indexing, ~11 meters)
        """
        super().__init__("positional_lookup_processor")
        self.output_dir = Path(output_dir)  # Now expects full file path
        self.positional_grid_resolution = positional_grid_resolution  # 0.0001 degrees (~11 meters)
        self.logger: PipelineLogger = get_logger()
        self.completion_tracker = CompletionTracker(self.output_dir.parent, processor_name="positional_lookup_processor")
        self.positions: List[Dict[str, Any]] = []  # Store collected position data (accumulated across MCAPs)
        self.current_mcap_id: Optional[str] = None  # Track which MCAP we're currently collecting from
        self.combined_data: Dict[str, Dict] = {}  # Accumulate all rosbag data in memory for single file output
        self.processed_rosbags: set = set()  # Track which rosbags have been processed
        self.mcap_message_counts: Dict[str, int] = defaultdict(int)  # Track message counts per MCAP
        self.collected_topic: Optional[str] = None  # Track which topic we're collecting

    def wants_message(self, topic: str, msg_type: str) -> bool:
        """Collect position messages."""
        if "bestpos" in topic.lower():
            if self.collected_topic is None:
                self.collected_topic = topic
            return True
        return False

    def collect_message(self, message: Any, channel: Any, schema: Any, ros2_msg: Any) -> None:
        """
        Collect a position message from bestpos topic.
        
        Stores position with MCAP ID to track which MCAP each position came from.
        
        Args:
            message: MCAP message (decoded ROS2 message)
            channel: MCAP channel info
            schema: MCAP schema info
            ros2_msg: Decoded ROS2 message
        """
        # Extract position data from message
        try:
            lat = ros2_msg.lat
            lon = ros2_msg.lon

            if lat == 0.0 or lon == 0.0:
                return  
            
        except (AttributeError, ValueError) as e:
            self.logger.warning(f"Failed to extract position from message: {e}")
            return
        
        # Store position with MCAP ID for aggregation
        # Note: current_mcap_id should be set before collection via reset() or process_mcap()
        self.positions.append({
            'latitude': lat,
            'longitude': lon,
            'mcap_id': self.current_mcap_id  # Track which MCAP this came from
        })
        
        # Track message count per MCAP
        if self.current_mcap_id:
            self.mcap_message_counts[self.current_mcap_id] += 1
    
    def get_data(self) -> Dict[str, List]:
        """
        Get collected position messages.
        
        Returns:
            Dictionary with collected positions
        """
        return {"positions": self.positions}

    def process_rosbag_before_mcaps(self, context: RosbagProcessingContext) -> None:
        """
        Initialize processing for a new rosbag.
        
        Called before MCAP iteration starts. Prepares state for collection.
        Note: Completion is checked in process_rosbag_after_mcaps after verifying data exists.
        
        Args:
            context: RosbagProcessingContext
        """
        # Clear positions for this rosbag (will accumulate across MCAPs)
        self.positions = []
        self.current_mcap_id = None
        self.mcap_message_counts = defaultdict(int)
        self.collected_topic = None
        
        # Log that we're starting collection
        self.logger.info(f"Collecting bestpos messages for positional lookup...")
    
    def reset(self) -> None:
        """
        Reset collector state before each MCAP iteration.
        
        Note: We DON'T clear positions - we want to accumulate across all MCAPs
        for rosbag-level aggregation. Only reset the current MCAP ID.
        """
        # Don't clear self.positions - accumulate across MCAPs
        # Just reset the current MCAP ID (will be set via context in collect_message or process_mcap)
        self.current_mcap_id = None
    
    def process_mcap(self, context: McapProcessingContext) -> None:
        """
        Mark MCAP as completed after message collection.
        
        Called after messages from this MCAP have been collected.
        Marks completion per MCAP while still aggregating into rosbag-level JSON.
        
        Args:
            context: MCAP processing context
        """
        # Get rosbag name and mcap name
        rosbag_name = str(context.get_relative_path())
        mcap_name = context.get_mcap_name()
        
        # Mark this MCAP as completed (no output_files at MCAP level - they contribute to rosbag-level file)
        self.completion_tracker.mark_completed(
            rosbag_name=rosbag_name,
            mcap_name=mcap_name,
            status="completed"
        )
    
    def is_mcap_completed(self, context: McapProcessingContext) -> bool:
        """
        Check if a specific MCAP is completed by checking completion.json.
        
        For PositionalLookupProcessor, completion.json is the source of truth since
        we cannot reliably determine if a specific MCAP was processed by looking at
        the positional_lookup_table.json (all MCAPs write to the same aggregated file).
        
        Args:
            context: MCAP processing context
            
        Returns:
            True if this MCAP is marked as completed in completion.json, False otherwise
        """
        rosbag_name = str(context.get_relative_path())
        mcap_name = context.get_mcap_name()
        
        # Use new unified interface
        return self.completion_tracker.is_mcap_completed(rosbag_name, mcap_name)
    
    def _verify_mcap_data_in_json(self, rosbag_name: str, mcap_id: str) -> bool:
        """
        Verify that a specific MCAP's data exists in the JSON file.
        
        Args:
            rosbag_name: Name of the rosbag
            mcap_id: ID of the MCAP (e.g., "0", "1", "669")
            
        Returns:
            True if this MCAP's data is present in the JSON file
        """
        if not self.output_dir.exists():
            return False
        
        try:
            with open(self.output_dir, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if rosbag entry exists
            if rosbag_name not in data:
                return False
            
            rosbag_data = data[rosbag_name]
            if not isinstance(rosbag_data, dict):
                return False
            
            # Check if any location entry has this MCAP ID in its mcaps dict
            for location_data in rosbag_data.values():
                if isinstance(location_data, dict) and "mcaps" in location_data:
                    mcaps = location_data["mcaps"]
                    if isinstance(mcaps, dict) and mcap_id in mcaps:
                        # Found this MCAP's data
                        return True
            
            return False
        except (json.JSONDecodeError, IOError):
            return False
    
    def get_output_path(self, context: Union[RosbagProcessingContext, McapProcessingContext]) -> Path:
        """Get the expected output path for this context (combined file)."""
        return self.output_dir
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected messages per MCAP.
        
        Returns:
            Dictionary with total positions, counts per MCAP, and topic name
        """
        return {
            "total_positions": len(self.positions),
            "mcap_counts": dict(self.mcap_message_counts),
            "topic": self.collected_topic or "bestpos (unknown)"
        }
    
    def process_rosbag_after_mcaps(self, context: RosbagProcessingContext) -> Dict:
        """
        Process collected position messages from ALL MCAPs and build aggregated lookup.
        
        Aggregates positions from all MCAPs into one rosbag-level lookup table.
        Accumulates data in memory instead of writing per-rosbag files.
        Structure: {"lat,lon": {"total": X, "mcaps": {"0": Y, "4": Z}}}
        
        Args:
            context: Rosbag processing context
        
        Returns:
            Aggregated positional lookup table with grid-based location counts
        """
        # Get rosbag name for storing in combined data
        rosbag_name = context.get_relative_path().as_posix()  # Use relative path as key
        
        # Read existing data from JSON file
        existing_data = {}
        if self.output_dir.exists():
            try:
                with open(self.output_dir, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Error reading combined file: {e}")
                # Continue with empty dict - will create new file
        
        # If no positions collected, skip processing
        if len(self.positions) == 0:
            self.logger.warning(f"No positions collected for {rosbag_name}, skipping aggregation")
            return {}
        
        # Calculate breakdown by MCAP
        mcap_breakdown = {}
        for pos in self.positions:
            mcap_id = pos.get('mcap_id')
            if mcap_id is None:
                mcap_id = 'unknown'
            else:
                mcap_id = str(mcap_id)  # Convert to string for consistency
            mcap_breakdown[mcap_id] = mcap_breakdown.get(mcap_id, 0) + 1
        
        # Log detailed summary
        topic_name = self.collected_topic or "bestpos"
        self.logger.info(f"Building aggregated positional lookup for {context.get_relative_path()}...")
        self.logger.info(f"  Collected {len(self.positions)} positions from {len(mcap_breakdown)} MCAP(s) via {topic_name}")
        if len(mcap_breakdown) > 0:
            breakdown_str = ", ".join([f"MCAP {mcap_id}: {count}" for mcap_id, count in sorted(mcap_breakdown.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)])
            self.logger.info(f"  Breakdown: {breakdown_str}")
        
        # Aggregate counts per location per MCAP
        # Structure: {lat_lon_key: {mcap_id: count}}
        location_counts = defaultdict(lambda: defaultdict(int))
        
        for pos in self.positions:
            lat_grid, lon_grid = self._round_to_grid(pos['latitude'], pos['longitude'])
            lat_lon_key = f"{lat_grid:.6f},{lon_grid:.6f}"
            # Ensure mcap_id is always a string (convert None to 'unknown')
            mcap_id = pos.get('mcap_id')
            if mcap_id is None:
                mcap_id = 'unknown'
            else:
                mcap_id = str(mcap_id)  # Convert to string
            location_counts[lat_lon_key][mcap_id] += 1
        
        # Convert to final structure with total and mcaps dict
        final_location_data = {}
        for lat_lon_key, mcap_counts in location_counts.items():
            total_count = sum(mcap_counts.values())
            final_location_data[lat_lon_key] = {
                "total": total_count,
                "mcaps": dict(mcap_counts)  # Convert defaultdict to regular dict
            }
        
        # Add new rosbag data
        existing_data[rosbag_name] = final_location_data
        
        # Write back immediately (atomic write)
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        # Verify the write succeeded by reading it back
        try:
            with open(self.output_dir, 'r', encoding='utf-8') as f:
                verify_data = json.load(f)
            if rosbag_name not in verify_data or not verify_data[rosbag_name]:
                raise ValueError(f"Data verification failed for {rosbag_name}")
        except Exception as e:
            self.logger.error(f"Failed to verify written data for {rosbag_name}: {e}")
            return {}
        
        # Also keep in memory for tracking (optional, for finalize summary)
        self.combined_data[rosbag_name] = final_location_data
        self.processed_rosbags.add(rosbag_name)
        
        self.logger.success(f"Built aggregated lookup with {len(location_counts)} grid cells from {len(self.positions)} positions")
        self.logger.info(f"  Incrementally written to {self.output_dir}")
        
        # Mark rosbag as completed (output file is at rosbag level)
        self.completion_tracker.mark_completed(
            rosbag_name=rosbag_name,
            status="completed",
            output_files=[self.output_dir]
        )
        
        # Clear positions after processing this rosbag (prepare for next rosbag)
        self.positions = []
        self.current_mcap_id = None
        self.mcap_message_counts = defaultdict(int)
        self.collected_topic = None
        
        return final_location_data
    
    def finalize(self) -> None:
        """
        Finalize positional lookup processing.
        
        Called after all rosbags have been processed.
        Since data is written incrementally after each rosbag, this method
        just provides a summary of what was processed.
        """
        if not self.output_dir.exists():
            self.logger.warning("No positional lookup file found to finalize")
            return
        
        # Read the file to get final count
        try:
            with open(self.output_dir, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
            
            total_rosbags = len(final_data)
            total_locations = sum(len(rosbag_data) for rosbag_data in final_data.values())
            
            self.logger.info(f"Finalizing positional lookup: {total_rosbags} rosbag(s) with {total_locations} total location entries in {self.output_dir}")
            self.logger.success(f"Positional lookup complete: {self.output_dir}")
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error reading combined file for finalization summary: {e}")


    def _round_to_grid(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Round GPS coordinates to grid cells.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
        
        Returns:
            Tuple of (rounded_latitude, rounded_longitude)
        """
        lat_grid = round(lat / self.positional_grid_resolution) * self.positional_grid_resolution
        lon_grid = round(lon / self.positional_grid_resolution) * self.positional_grid_resolution
        return lat_grid, lon_grid