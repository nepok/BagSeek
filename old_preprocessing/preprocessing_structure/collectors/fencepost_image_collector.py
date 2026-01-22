"""
Collector for gathering fencepost images at specific indices.
"""
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from .base_collector import BaseCollector


@dataclass
class FencepostImage:
    """Container for fencepost image data"""
    topic: str
    timestamp: int
    part_index: int  # Which fencepost part (1-7)
    mcap_index: int  # Which MCAP this came from
    message_index: int  # Index within the topic
    data: Any  # Raw ROS2 message


class FencepostImageCollector(BaseCollector):
    """
    Collects images at specific indices based on fencepost mapping.
    
    Given a fencepost mapping that specifies which parts to extract from
    a specific MCAP at which percentages, this collector:
    1. Calculates target message indices for each topic
    2. Tracks current index while iterating
    3. Collects only messages that match target indices
    
    Example:
        If MCAP 1 should extract part 2 at 25% and part 3 at 87.5%,
        and a topic has 100 messages, this collector will extract
        messages at indices 25 and 87.
    """
    
    def __init__(self, mcap_context, fencepost_calculator, topic_info: Dict[str, Dict[str, int]], cache_enabled: bool = False):
        """
        Initialize fencepost image collector.
        
        Args:
            mcap_context: Processing context for extracting mcap_index
            fencepost_calculator: FencepostCalculator instance to get mapping from
            topic_info: Dict mapping topic -> {"topic_id": id, "message_count": count}
            cache_enabled: Whether to cache results
        """
        super().__init__(cache_enabled)
        
        # Extract mcap_index from context
        self.mcap_index = int(mcap_context.mcap_path.stem.split('_')[-1])
        
        # Get fencepost_mapping directly from calculator
        self.fencepost_mapping = fencepost_calculator.last_mapping
        
        # Filter topic_info for image topics only
        self.topic_info = {
            topic: info for topic, info in topic_info.items()
            if self._is_image_topic(topic)
        }
        
        # Calculate target indices for this MCAP
        self.target_indices = self._calculate_target_indices()
        
        # Track current index per topic (initialize for image topics only)
        self.topic_index = {topic: 0 for topic in self.topic_info.keys()}
        
        # Store collected images
        self.collected_images: List[FencepostImage] = []
        
        # Log what we're collecting
        print(f"      FencepostImageCollector: {len(self.topic_info)} image topic(s), {len(self.target_indices)} with target indices")
    
    def _calculate_target_indices(self) -> Dict[str, Dict[int, int]]:
        """
        Calculate target message indices for each topic and part.
        
        Returns:
            Dict mapping topic -> {part_idx: target_message_index}
            Example: {"/camera/front": {1: 45, 2: 90}}
        """
        target_indices = {}
        
        # Check if this MCAP has parts to extract (handle both string and int keys)
        mcap_key = str(self.mcap_index)
        if mcap_key not in self.fencepost_mapping and self.mcap_index not in self.fencepost_mapping:
            return target_indices
        
        parts = self.fencepost_mapping.get(mcap_key, self.fencepost_mapping.get(self.mcap_index, []))
        
        for part_idx, percentage in parts:
            for topic, info in self.topic_info.items():
                total_messages = info["message_count"]
                target_idx = int(total_messages * percentage)
                
                if topic not in target_indices:
                    target_indices[topic] = {}
                target_indices[topic][part_idx] = target_idx
        
        return target_indices
    
    def collect_message(self, message, channel, schema):
        """
        Collect a single message if it matches a target index.
        
        Args:
            message: MCAP message
            channel: MCAP channel info
            schema: MCAP schema info
        """
        topic = channel.topic
        
        # Skip topics we don't care about (non-image topics)
        if topic not in self.topic_info:
            return
        
        # Get current index for this topic
        current_idx = self.topic_index[topic]
        
        # Check if this message index matches any target index for this topic
        if topic in self.target_indices:
            for part_idx, target_idx in self.target_indices[topic].items():
                if current_idx == target_idx:
                    # This is a fencepost image!
                    fencepost_img = FencepostImage(
                        topic=topic,
                        timestamp=message.log_time,
                        part_index=part_idx,
                        mcap_index=self.mcap_index,
                        message_index=current_idx,
                        data=message
                    )
                    self.collected_images.append(fencepost_img)
        
        # Increment index for this topic
        self.topic_index[topic] += 1
    
    def get_data(self) -> List[FencepostImage]:
        """
        Get collected fencepost images.
        
        Returns:
            List of FencepostImage objects
        """
        return self.collected_images
    
    def _is_image_topic(self, topic: str) -> bool:
        """
        Check if a topic is an image topic.
        
        Args:
            topic: Topic name
        
        Returns:
            True if it's an image topic
        """
        return "image" in topic.lower() or "camera" in topic.lower()
