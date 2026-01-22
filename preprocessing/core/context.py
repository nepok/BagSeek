"""
Core context classes for the preprocessing pipeline.

Provides context objects that preserve relative paths for proper handling
of multi-part rosbags and nested directory structures.
"""
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..config import Config


@dataclass
class RosbagProcessingContext:
    """
    Shared context passed through the pipeline for rosbag-level processing.
    
    Preserves relative paths to handle multi-part rosbags correctly.
    """
    rosbag_path: Path
    config: "Config"
    mcap_files: List[Path] = field(default_factory=list)
    
    def get_relative_path(self) -> Path:
        """
        Get path relative to rosbags directory.
        
        This preserves the directory structure from the rosbags folder,
        allowing all output directories to mirror the input structure.
        Automatically handles regular rosbags, multi-part rosbags, and
        any future nested directory structures.
        
        Returns:
            Path relative to config.rosbags_dir
            
        Examples:
            - Regular: rosbag2_2025_07_25-10_14_58
            - Multi-part: rosbag2_2025_07_25-15_37_32_multi_parts/Part_1
            - Future nested: 2025/july/rosbag2_xxx
        """
        return self.rosbag_path.relative_to(self.config.rosbags_dir)
    
    def get_rosbag_name(self) -> str:
        """
        Get the rosbag name (for backward compatibility).
        
        Returns:
            Name of the rosbag directory
        """
        return self.rosbag_path.name


@dataclass
class McapProcessingContext:
    """
    Shared context passed through the pipeline for MCAP-level processing.
    
    Contains both rosbag and MCAP path information.
    """
    rosbag_path: Path
    config: "Config"
    mcap_path: Path
    mcap_files: List[Path] = field(default_factory=list)
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    
    def get_relative_path(self) -> Path:
        """
        Get rosbag path relative to rosbags directory.
        
        Returns:
            Path relative to config.rosbags_dir
        """
        return self.rosbag_path.relative_to(self.config.rosbags_dir)
    
    def get_rosbag_name(self) -> str:
        """
        Get the rosbag name.
        
        Returns:
            Name of the rosbag directory
        """
        return self.rosbag_path.name
    
    def get_mcap_name(self) -> str:
        """
        Get the MCAP file name.
        
        Returns:
            Name of the MCAP file (without path)
        """
        return self.mcap_path.name

    def get_mcap_id(self) -> str:
        """
        Get the MCAP file id.
        
        Returns:
            Number of the MCAP file
        """
        return self.mcap_path.stem.split("_")[-1]


# Type alias for convenience
ProcessingContext = RosbagProcessingContext | McapProcessingContext

