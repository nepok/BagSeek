"""
MCAP-level processor for building timestamp alignment tables.
"""
import json
from pathlib import Path
from typing import Dict, Any
from ...core import Processor, ProcessingLevel, McapProcessingContext, CompletionTracker
from ...collectors import TimestampsCollector


class TimestampAlignmentBuilder(Processor):
    """
    Build timestamp lookup table for an MCAP.
    
    Uses TimestampsCollector to gather timestamp data.
    
    Operates at MCAP level - runs once per mcap file.
    """
    
    def __init__(self):
        super().__init__("timestamp_alignment", ProcessingLevel.MCAP)
        self.required_collectors = [TimestampsCollector]
    
    def process(self, context: McapProcessingContext, data: Any) -> Dict:
        """
        Collect all timestamps from all topics in the mcap.
        
        Args:
            context: Processing context with mcap_path set
            data: Dictionary containing collector results
        
        Returns:
            Timestamp lookup table
        """
        # Check completion
        output_dir = context.config.lookup_tables_dir / context.get_rosbag_name()
        completion_file = context.config.lookup_tables_dir / "completion.json"
        completion_tracker = CompletionTracker(completion_file)
        
        if completion_tracker.is_completed(context.get_rosbag_name(), context.get_mcap_name()):
            print(f"    ✓ Timestamps already built for {context.get_mcap_name()}, skipping")
            return {}
        
        print(f"    Building timestamp alignment for {context.get_mcap_name()}...")
        
        # Get timestamps from collector
        timestamps = data.get("TimestampsCollector", {})
        
        # TODO: Process/transform timestamps if needed
        # The timestamps dict is already in the right format:
        # {"topic_name": [timestamp1, timestamp2, ...]}
        # 
        # You might want to add additional processing here:
        # - Sort timestamps
        # - Calculate time ranges
        # - Build lookup indices
        # - etc.
        
        # Write to rosbag-specific folder
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{context.get_mcap_name()}.json"
        
        with open(output_file, 'w') as f:
            json.dump(timestamps, f, indent=2)
        
        # Mark as completed
        completion_tracker.mark_completed(
            context.get_rosbag_name(),
            output_file,
            mcap_name=context.get_mcap_name(),
            metadata={"topic_count": len(timestamps)}
        )
        
        print(f"    ✓ Built timestamps for {len(timestamps)} topics")
        return timestamps

