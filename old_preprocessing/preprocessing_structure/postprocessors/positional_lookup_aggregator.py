"""
Postprocessor for aggregating positional lookups across mcaps.
"""
import json
from pathlib import Path
from typing import Dict
from ..core import CompletionTracker


class PositionalLookupAggregator:
    """
    Aggregate positional lookups across mcaps.
    
    Combines individual mcap-level position data into
    final rosbag-level lookup tables.
    
    Runs after main pipeline.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.positional_dir = output_dir / "positional_lookup"
    
    def run(self):
        """
        Combine all mcap-level positional data into final rosbag-level structure.
        """
        print("\nAggregating positional lookups...")
        
        # Check completion
        completion_tracker = CompletionTracker(
            self.output_dir / "positional_lookup_aggregated" / "completion.json"
        )
        
        # Get all rosbag directories
        if not self.positional_dir.exists():
            print("  No positional data found, skipping")
            return
        
        rosbag_dirs = [d for d in self.positional_dir.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
        
        for rosbag_dir in rosbag_dirs:
            rosbag_name = rosbag_dir.name
            
            if completion_tracker.is_completed(rosbag_name):
                print(f"  ✓ Lookup already aggregated for {rosbag_name}, skipping")
                continue
            
            print(f"  Aggregating lookup for {rosbag_name}...")
            
            # Aggregate all mcap files for this rosbag
            aggregated_lookup = {}
            
            # TODO: Implement aggregation logic
            # 
            # Algorithm:
            # 1. Read all mcap-level position files for this rosbag
            # 2. Merge grid cell counts (sum counts for same cells)
            # 3. Create final lookup structure
            # 
            # Example structure:
            # mcap_files = sorted(rosbag_dir.glob("*.json"))
            # 
            # for mcap_file in mcap_files:
            #     with open(mcap_file, 'r') as f:
            #         mcap_data = json.load(f)
            #     
            #     # Merge counts
            #     for grid_key, count in mcap_data.items():
            #         aggregated_lookup[grid_key] = aggregated_lookup.get(grid_key, 0) + count
            # 
            # # Optionally add metadata
            # final_lookup = {
            #     "rosbag": rosbag_name,
            #     "total_cells": len(aggregated_lookup),
            #     "total_positions": sum(aggregated_lookup.values()),
            #     "grid_cells": aggregated_lookup
            # }
            
            # Write aggregated result
            output_file = self.output_dir / "positional_lookup_aggregated" / f"{rosbag_name}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(aggregated_lookup, f, indent=2)
            
            # Mark as completed
            completion_tracker.mark_completed(rosbag_name, output_file)
            
            print(f"  ✓ Aggregated {len(aggregated_lookup)} grid cells for {rosbag_name}")
        
        print("✓ All positional lookups aggregated!")

