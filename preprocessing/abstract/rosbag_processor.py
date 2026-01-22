"""
Base class for processors that operate purely at rosbag level.

These processors run once per rosbag, typically before MCAP iteration.
They don't need to collect messages from MCAPs.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from ..core import RosbagProcessingContext


class RosbagProcessor(ABC):
    """
    Base class for processors that operate purely at rosbag level.
    
    These processors run once per rosbag, typically before MCAP iteration.
    They don't need to collect messages from MCAPs.
    
    Examples: TopicsExtractionProcessor
    """
    
    def __init__(self, name: str):
        """
        Initialize rosbag processor.
        
        Args:
            name: Processor name/identifier
        """
        self.name = name
        self.enabled = True
    
    @abstractmethod
    def process_rosbag(self, context: RosbagProcessingContext) -> Any:
        """
        Process rosbag-level data.
        
        Called once per rosbag. This is the main processing method.
        
        Args:
            context: RosbagProcessingContext
        
        Returns:
            Processed data or None
        """
        pass
    
    def should_process(self, context: RosbagProcessingContext) -> bool:
        """
        Check if this processor should run.
        
        Override to add conditional processing logic.
        
        Args:
            context: RosbagProcessingContext
        
        Returns:
            True if processor should run, False otherwise
        """
        return self.enabled
    
    def get_output_path(self, context: RosbagProcessingContext) -> Optional[Path]:
        """
        Get the expected output path for a given context.
        
        Override this method in processors to return the path where
        output would be written for the given context.
        This is used by the completion tracker to check completion status.
        
        Args:
            context: RosbagProcessingContext
        
        Returns:
            Path to expected output file, or None if not applicable
        """
        return None
