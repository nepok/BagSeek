"""
Rosbag-level processor for calculating fencepost positions.
"""
from pathlib import Path
from typing import Dict, Any, List
from ...core import Processor, ProcessingLevel, RosbagProcessingContext, CompletionTracker


class FencepostCalculator(Processor):
    """
    Calculate fencepost mapping for distributing representative images across a rosbag.
    
    Calculates which MCAPs should contain which of the 7 representative images
    at fractional positions (1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8) and at what
    percentage through each MCAP to extract them.
    
    This processor ONLY calculates the mapping. Actual image collection is done
    by MCAP-level processors that read this mapping.
    
    Operates at ROSBAG level - runs once per rosbag.
    """
    
    def __init__(self):
        super().__init__("fencepost_calculator", ProcessingLevel.ROSBAG)
        # This processor only calculates mapping, no collectors needed
        self.required_collectors = []
        self.last_mapping = {}  # Store the most recent mapping for collectors
    
    def process(self, context: RosbagProcessingContext, data: Any) -> Dict:
        """
        Calculate the fencepost mapping for the rosbag.
        
        This processor ONLY calculates which MCAPs contain which fencepost parts
        at what percentages. The actual image collection is done by other processors.
        Completion is checked but not marked - only the stitcher marks completion.
        
        Args:
            context: Processing context
            data: Collector data (not used)
        
        Returns:
            Dict with fencepost mapping information
        """
        # Check if already completed by stitcher
        output_dir = context.config.representative_previews_dir
        completion_tracker = CompletionTracker(output_dir / "completion.json")
        
        if completion_tracker.is_completed(context):
            print(f"  ✓ Previews already complete for {context.get_relative_path()}, skipping")
            return {}
        
        print(f"  📊 Calculating fencepost mapping for {context.get_relative_path()}...")
        
        # Get sorted MCAP files
        mcap_files = self._get_sorted_mcap_files(context.rosbag_path)
        
        if not mcap_files:
            print(f"  ⚠ No MCAP files found in {context.rosbag_path}")
            return {}
        
        # Calculate fencepost mapping
        fencepost_mapping = self._calculate_fencepost_mapping(len(mcap_files))
        
        # Print mapping details
        print(f"     {len(mcap_files)} MCAPs → 7 parts distributed:")
        for mcap_idx, parts in sorted(fencepost_mapping.items(), key=lambda x: int(x[0])):
            parts_str = ", ".join([f"part {p[0]} at {p[1]:.1%}" for p in parts])
            print(f"       MCAP {mcap_idx}: {parts_str}")
        
        print(f"  ✓ Fencepost mapping calculated")
        
        # Store mapping as instance variable for collectors to access
        self.last_mapping = fencepost_mapping
        
        return {
            "rosbag": str(context.get_relative_path()),
            "num_mcaps": len(mcap_files),
            "mapping": fencepost_mapping
        }
    
    def _get_sorted_mcap_files(self, rosbag_path: Path) -> List[Path]:
        """
        Get all MCAP files sorted by numeric ID.
        
        Args:
            rosbag_path: Path to rosbag directory
        
        Returns:
            List of MCAP file paths sorted by numeric ID
        """
        mcap_files = list(rosbag_path.glob("*.mcap"))
        
        # Sort by numeric ID at the end of filename
        mcap_files.sort(key=lambda x: int(x.stem.rsplit('_', 1)[-1]))
        
        return mcap_files
    
    def _calculate_fencepost_mapping(self, num_mcaps: int) -> Dict[str, List[tuple]]:
        """
        Calculate fencepost mapping for distributing parts across MCAPs.
        
        Args:
            num_mcaps: Number of MCAP files
        
        Returns:
            Dict mapping mcap_idx (as string) -> [(part_idx, percentage), ...]
        """
        num_parts = 8  # 7 fencepost images (parts 1-7)
        step_length = num_mcaps / num_parts
        
        mcap_to_parts = {}
        
        for part_idx in range(1, num_parts):  # parts 1 to 7
            position = step_length * part_idx
            mcap_idx = int(position)  # Floor to get MCAP index
            percentage = position - mcap_idx  # Fractional part is the percentage
            
            mcap_key = str(mcap_idx)
            if mcap_key not in mcap_to_parts:
                mcap_to_parts[mcap_key] = []
            
            mcap_to_parts[mcap_key].append([part_idx, percentage])
        
        return mcap_to_parts

