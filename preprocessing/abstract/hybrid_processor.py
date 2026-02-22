"""
Base class for processors that operate at both rosbag and MCAP levels.

These processors:
- May compute rosbag-level data before MCAP iteration (e.g., fenceposts)
- Collect messages during MCAP iteration
- Process/aggregate after all MCAPs of a rosbag
- May finalize across all rosbags
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional
from ..core import RosbagProcessingContext, McapProcessingContext


class HybridProcessor(ABC):
    """
    Base class for processors that operate at both rosbag and MCAP levels.
    
    These processors:
    - May compute rosbag-level data before MCAP iteration (e.g., fenceposts)
    - Collect messages during MCAP iteration
    - Process/aggregate after all MCAPs of a rosbag
    - May finalize across all rosbags
    
    Examples: PositionalLookupProcessor, ImagePreviewsProcessor, EmbeddingsProcessor
    """
    
    def __init__(self, name: str):
        """
        Initialize hybrid processor.
        
        Args:
            name: Processor name/identifier
        """
        self.name = name
        self.enabled = True
    
    # ====================================================================
    # ROSBAG-LEVEL METHODS
    # ====================================================================
    
    def process_rosbag_before_mcaps(self, context: RosbagProcessingContext) -> Any:
        """
        Process rosbag-level data BEFORE MCAP iteration.
        
        Called once per rosbag, before the MCAP loop starts.
        Use this for rosbag-level computations that inform MCAP collection
        (e.g., computing fencepost positions for image previews).
        
        Args:
            context: RosbagProcessingContext
        
        Returns:
            Processed data or None
        """
        return None
    
    @abstractmethod
    def process_rosbag_after_mcaps(self, context: RosbagProcessingContext) -> Any:
        """
        Process rosbag-level data AFTER all MCAPs have been processed.
        
        Called once per rosbag, after all MCAPs have been iterated.
        Use this to aggregate MCAP-level data into rosbag-level outputs
        (e.g., stitching images, aggregating positions, finalizing shards).
        
        Args:
            context: RosbagProcessingContext
        
        Returns:
            Processed data or None
        """
        pass
    
    def finalize(self) -> None:
        """
        Finalize processing after all rosbags have been processed.
        
        Called once after all rosbags are done.
        Use this to combine all rosbag-level outputs into a single global file
        (e.g., combining all per-rosbag JSONs into one big lookup table).
        
        This is optional - only override if you need global aggregation.
        """
        pass
    
    # ====================================================================
    # MCAP-LEVEL METHODS
    # ====================================================================
    
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
    
    def process_mcap(self, context: McapProcessingContext) -> Any:
        """
        Process MCAP-level data after message collection (optional).
        
        Called after all messages from one MCAP have been collected.
        Override if you need to process data immediately after each MCAP
        (e.g., if you can't accumulate everything in memory).
        
        Most hybrid processors don't need this - they accumulate during
        collection and process in process_rosbag_after_mcaps().
        
        Args:
            context: McapProcessingContext
        
        Returns:
            Processed data or None
        """
        return None
    
    def should_process(self, context: Any) -> bool:
        """
        Check if this processor should run.
        
        Override to add conditional processing logic.
        
        Args:
            context: Processing context (rosbag or mcap)
        
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
        Check if this processor has fully completed for a given rosbag.

        Default: fast-path rosbag-level check first, then per-MCAP fallback.
        Override for processors with custom completion semantics.

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
