"""
Base class for processors that operate purely at MCAP level.

These processors collect messages during MCAP iteration and process per MCAP.
They don't need rosbag-level aggregation.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional
from ..core import McapProcessingContext, RosbagProcessingContext


class McapProcessor(ABC):
    """
    Base class for processors that operate purely at MCAP level.
    
    These processors collect messages during MCAP iteration and process per MCAP.
    They don't need rosbag-level aggregation.
    
    Examples: TimestampAlignmentProcessor
    """
    
    def __init__(self, name: str):
        """
        Initialize MCAP processor.
        
        Args:
            name: Processor name/identifier
        """
        self.name = name
        self.enabled = True
    
    def reset(self) -> None:
        """
        Reset collector state before each MCAP iteration.
        
        Called before each MCAP to clear MCAP-specific state.
        Override if you need to reset state between MCAPs.
        """
        pass
    
    def wants_message(self, topic: str, msg_type: str) -> bool:
        """
        Filter which messages to collect.
        
        Override this if the processor needs to collect messages.
        Returns False by default.
        
        Args:
            topic: Topic name
            msg_type: Message type string
        
        Returns:
            True if this message should be collected, False otherwise
        """
        return False
    
    def collect_message(self, message: Any, channel: Any, schema: Any, ros2_msg: Any) -> None:
        """
        Collect a single message.
        
        Override this if the processor needs to collect messages.
        Called during MCAP message iteration.
        
        Args:
            message: MCAP message
            channel: MCAP channel info
            schema: MCAP schema info
            ros2_msg: Decoded ROS2 message
        """
        pass
    
    @abstractmethod
    def process_mcap(self, context: McapProcessingContext) -> Any:
        """
        Process MCAP-level data after message collection.
        
        Called after all messages from one MCAP have been collected.
        This is the main processing method.
        
        Args:
            context: McapProcessingContext
        
        Returns:
            Processed data or None
        """
        pass
    
    def should_process(self, context: McapProcessingContext) -> bool:
        """
        Check if this processor should run.
        
        Override to add conditional processing logic.
        
        Args:
            context: McapProcessingContext
        
        Returns:
            True if processor should run, False otherwise
        """
        return self.enabled
    
    def get_output_path(self, context: McapProcessingContext) -> Optional[Path]:
        """
        Get the expected output path for a given context.

        Override this method in processors to return the path where
        output would be written for the given context.
        This is used by the completion tracker to check completion status.

        Args:
            context: McapProcessingContext

        Returns:
            Path to expected output file, or None if not applicable
        """
        return None

    def is_rosbag_complete(self, rosbag_name: str, mcap_names: List[str]) -> bool:
        """
        Check if this processor has fully completed for a given rosbag.

        Default: fast-path rosbag-level check first, then per-MCAP fallback.
        Override for processors with custom completion semantics (e.g. multi-model).

        Args:
            rosbag_name: Rosbag relative path string
            mcap_names: List of MCAP filenames to check individually on slow path

        Returns:
            True if all MCAPs are complete for this processor
        """
        if not mcap_names:
            return True
        tracker = getattr(self, 'completion_tracker', None)
        if tracker is None:
            return False
        if tracker.is_rosbag_completed(rosbag_name):
            return True
        return all(tracker.is_mcap_completed(rosbag_name, mcap) for mcap in mcap_names)

    def is_mcap_complete(self, context: McapProcessingContext) -> bool:
        """
        Check if this processor has completed for a specific MCAP.

        Default delegates to the MCAP-level completion tracker entry.
        Override for processors with custom MCAP completion logic.

        Args:
            context: McapProcessingContext for the MCAP to check

        Returns:
            True if this MCAP is marked complete for this processor
        """
        tracker = getattr(self, 'completion_tracker', None)
        if tracker is None:
            return False
        rosbag_name = str(context.get_relative_path())
        return tracker.is_mcap_completed(rosbag_name, context.get_mcap_name())

    def on_rosbag_complete(self, context: RosbagProcessingContext) -> None:
        """
        Hook called at the end of each rosbag's processing loop.

        No-op by default. Override to perform idempotent post-loop work.

        Args:
            context: RosbagProcessingContext
        """
        pass
