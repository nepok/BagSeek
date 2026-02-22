"""
Base class for postprocessors that use outputs from other processors.

These processors don't collect messages themselves, but instead
read and process outputs from other processors.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional
from ..core import RosbagProcessingContext


class PostProcessor(ABC):
    """
    Base class for postprocessors that use outputs from other processors.
    
    These processors don't collect messages themselves, but instead
    read and process outputs from other processors.
    
    Examples: AdjacentSimilaritiesPostprocessor
    """
    
    def __init__(self, name: str):
        """
        Initialize postprocessor.
        
        Args:
            name: Postprocessor name/identifier
        """
        self.name = name
        self.enabled = True
    
    @abstractmethod
    def process_rosbag(self, context: RosbagProcessingContext) -> Any:
        """
        Process rosbag-level data using outputs from other processors.
        
        Called once per rosbag. Reads outputs from other processors
        and generates one central output.
        
        Args:
            context: RosbagProcessingContext
        
        Returns:
            Processed data or None
        """
        pass
    
    def finalize(self) -> None:
        """
        Finalize processing after all rosbags (optional).
        
        Called once after all rosbags are done.
        Override if you need global aggregation.
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

    def is_rosbag_complete(self, rosbag_name: str, mcap_names: List[str]) -> bool:
        """
        Check if this postprocessor has fully completed for a given rosbag.

        Default delegates to the rosbag-level completion tracker entry.
        Override for postprocessors with custom completion semantics.

        Args:
            rosbag_name: Rosbag relative path string
            mcap_names: List of MCAP filenames (unused by default)

        Returns:
            True if the rosbag is marked complete for this postprocessor
        """
        tracker = getattr(self, 'completion_tracker', None)
        if tracker is None:
            return False
        return tracker.is_rosbag_completed(rosbag_name)

    def on_rosbag_complete(self, context: RosbagProcessingContext) -> None:
        """
        Hook called at the end of each rosbag's processing loop.

        No-op by default. Override to perform idempotent post-loop work.

        Args:
            context: RosbagProcessingContext
        """
        pass

