"""
Collector for gathering image messages with fencepost tracking.
"""
from typing import Dict, List, Any
from dataclasses import dataclass
from .base_collector import BaseCollector


@dataclass
class ImageMessage:
    """Container for image message data"""
    topic: str
    timestamp: int
    data: Any  # Raw image data or PIL Image
    is_fencepost: bool = False  # True if this is a fencepost image
    fencepost_index: int = -1  # Which fencepost (1-7)


class ImageMessagesCollector(BaseCollector):
    """
    Collects all image messages with fencepost tracking.
    
    Tracks progress through messages (0-100%) and flags messages
    at fractional positions: 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8
    
    Used by: EmbeddingGenerator, FencepostCalculator
    """
    
    # Fencepost positions (7 preview images)
    FENCEPOST_FRACTIONS = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8]
    
    def __init__(self, cache_enabled: bool = False, track_fenceposts: bool = True):
        super().__init__(cache_enabled)
        self.track_fenceposts = track_fenceposts
        self.messages_by_topic = {}
        self.topic_indices = {}
        self.topic_counts = {}
        self.fencepost_indices = {}
    
    def set_topic_counts(self, topic_counts: Dict[str, int]):
        """
        Set expected message counts for fencepost calculation.
        Call this before collecting messages if using fencepost tracking.
        
        Args:
            topic_counts: Dictionary of topic -> message count
        """
        self.topic_counts = topic_counts
        self.topic_indices = {topic: 0 for topic in topic_counts}
        self.fencepost_indices = {topic: 0 for topic in topic_counts}
    
    def collect_message(self, message, channel, schema):
        """
        Collect a single image message with fencepost tracking.
        
        Args:
            message: MCAP message
            channel: MCAP channel info
            schema: MCAP schema info
        """
        topic = channel.topic
        
        # Check if it's an image topic
        if not self._is_image_topic(topic):
            return
        
        if topic not in self.messages_by_topic:
            self.messages_by_topic[topic] = []
        
        if topic not in self.topic_indices:
            self.topic_indices[topic] = 0
        
        # TODO: Implement fencepost tracking and image extraction
        # 
        # Calculate progress percentage if tracking fenceposts:
        # current_idx = self.topic_indices[topic]
        # total_count = self.topic_counts.get(topic, 0)
        # progress = current_idx / total_count if total_count > 0 else 0
        # 
        # Check if we've crossed a fencepost:
        # is_fencepost = False
        # fencepost_idx = -1
        # if self.track_fenceposts and topic in self.fencepost_indices:
        #     if self.fencepost_indices[topic] < len(self.FENCEPOST_FRACTIONS):
        #         target_fraction = self.FENCEPOST_FRACTIONS[self.fencepost_indices[topic]]
        #         if progress >= target_fraction:
        #             is_fencepost = True
        #             fencepost_idx = self.fencepost_indices[topic] + 1
        #             self.fencepost_indices[topic] += 1
        # 
        # Extract image data:
        # image_data = self._decode_image(message)
        # 
        # Create ImageMessage:
        # img_msg = ImageMessage(
        #     topic=topic,
        #     timestamp=message.log_time,
        #     data=image_data,
        #     is_fencepost=is_fencepost,
        #     fencepost_index=fencepost_idx
        # )
        # self.messages_by_topic[topic].append(img_msg)
        
        self.topic_indices[topic] += 1
    
    def get_data(self) -> Dict[str, List[ImageMessage]]:
        """
        Get collected image messages.
        
        Returns:
            Dictionary mapping topic names to lists of ImageMessage objects
            Format: {"image_topic": [ImageMessage(...), ...]}
        """
        return self.messages_by_topic
    
    def _is_image_topic(self, topic: str) -> bool:
        """
        Check if a topic is an image topic.
        
        Args:
            topic: Topic name
        
        Returns:
            True if it's an image topic
        """
        # TODO: Implement image topic detection
        # Common patterns: /camera, /image, message type sensor_msgs/Image
        return "image" in topic.lower() or "camera" in topic.lower()
    
    def _decode_image(self, message: Any) -> Any:
        """
        Decode ROS image message to usable format.
        
        Args:
            message: ROS message
        
        Returns:
            PIL Image or numpy array
        """
        # TODO: Implement image decoding
        # from PIL import Image
        # import numpy as np
        # 
        # # Extract image data from ROS message
        # # Convert based on encoding (bgr8, rgb8, mono8, etc.)
        pass

