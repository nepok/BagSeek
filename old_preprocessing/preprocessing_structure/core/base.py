"""
Core base classes for the rosbag processing pipeline.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from ..config import Config


class ProcessingLevel(Enum):
    """Defines at what level a processor operates"""
    ROSBAG = "rosbag"  # Processes entire rosbag
    MCAP = "mcap"      # Processes each mcap in a rosbag
    MESSAGE = "message" # Processes individual messages


@dataclass
class RosbagProcessingContext:
    """Shared context passed through the pipeline"""
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

@dataclass 
class McapProcessingContext:
    """Shared context passed through the pipeline"""
    rosbag_path: Path
    config: "Config"
    mcap_path: Path
    mcap_files: List[Path] = field(default_factory=list)
    cache_dir: Path = field(default_factory=lambda: Path("cache"))


# Type alias for backward compatibility
ProcessingContext = Union[RosbagProcessingContext, McapProcessingContext]


class Processor(ABC):
    """Base class for all processors"""
    
    def __init__(self, name: str, level: ProcessingLevel):
        self.name = name
        self.level = level
        self.enabled = True
        self.required_collectors = []  # List of collector classes this processor needs
    
    @abstractmethod
    def process(self, context: ProcessingContext, data: Any) -> Any:
        """
        Process data at the appropriate level.
        
        Args:
            context: Current processing context
            data: Input data (varies by level, may include collector results)
        
        Returns:
            Processed data or None
        """
        pass
    
    def should_process(self, context: ProcessingContext) -> bool:
        """Override to add conditional processing logic"""
        return self.enabled

