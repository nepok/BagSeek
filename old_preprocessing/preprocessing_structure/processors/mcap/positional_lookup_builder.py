"""
MCAP-level processor for building positional lookup tables.
"""
import json
from pathlib import Path
from typing import Dict, Any
from ...core import Processor, ProcessingLevel, McapProcessingContext, CompletionTracker
from ...collectors import PositionMessagesCollector


class PositionalLookupBuilder(Processor):
    """
    Build positional lookup table from bestpos messages.
    
    Uses PositionMessagesCollector to gather position data.
    Creates grid-based lookup for this MCAP (later aggregated by postprocessor).
    
    Operates at MCAP level - runs once per mcap file.
    """
    
    def __init__(self, grid_size: float = 10.0):
        super().__init__("positional_lookup", ProcessingLevel.MCAP)
        self.grid_size = grid_size
        self.required_collectors = [PositionMessagesCollector]
    
    def process(self, context: McapProcessingContext, data: Any) -> Dict:
        """
        Process bestpos messages and build grid-based lookup.
        
        Args:
            context: Processing context with mcap_path set
            data: Dictionary containing collector results
        
        Returns:
            Position counts for this mcap
        """
        # Check completion
        output_dir = context.config.lookup_tables_dir / "positional_lookup" / context.get_rosbag_name()
        completion_file = context.config.lookup_tables_dir / "positional_lookup" / "completion.json"
        completion_tracker = CompletionTracker(completion_file)
        
        if completion_tracker.is_completed(context.get_rosbag_name(), context.get_mcap_name()):
            print(f"    ✓ Positional lookup already built for {context.get_mcap_name()}, skipping")
            return {}
        
        print(f"    Building positional lookup for {context.get_mcap_name()}...")
        
        # Get positions from collector
        positions = data.get("PositionMessagesCollector", [])
        
        position_counts = {}
        
        # TODO: Implement grid-based position counting
        # 
        # Algorithm:
        # 1. For each position message:
        #    - Calculate grid cell: grid_key = (round(lat/grid_size), round(lon/grid_size))
        #    - Increment count for that grid cell
        # 
        # Example:
        # for pos in positions:
        #     # Calculate grid cell
        #     grid_lat = round(pos.latitude / self.grid_size) * self.grid_size
        #     grid_lon = round(pos.longitude / self.grid_size) * self.grid_size
        #     grid_key = f"{grid_lat:.4f},{grid_lon:.4f}"
        #     
        #     # Increment count
        #     position_counts[grid_key] = position_counts.get(grid_key, 0) + 1
        
        # Write intermediate result for this mcap
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{context.get_mcap_name()}.json"
        
        with open(output_file, 'w') as f:
            json.dump(position_counts, f, indent=2)
        
        # Mark as completed
        completion_tracker.mark_completed(
            context.get_rosbag_name(),
            output_file,
            mcap_name=context.get_mcap_name(),
            metadata={"grid_cells": len(position_counts), "total_positions": len(positions)}
        )
        
        print(f"    ✓ Built lookup with {len(position_counts)} grid cells from {len(positions)} positions")
        return position_counts

