"""
Collector for gathering timestamps from all topics.
"""
from typing import Dict, List
from .base_collector import BaseCollector


class TimestampsCollector(BaseCollector):
    """
    Collects all timestamps from all topics in an MCAP.
    
    Used by: TimestampAlignmentBuilder
    """
    
    def __init__(self, cache_enabled: bool = False):
        super().__init__(cache_enabled)
        self.timestamps = {}
    
    def collect_message(self, message, channel, schema):
        """
        Collect a timestamp from a message.
        
        Args:
            message: MCAP message
            channel: MCAP channel info
            schema: MCAP schema info
        """
        # TODO: Implement timestamp extraction
        # Extract topic and timestamp from message
        # 
        # topic = channel.topic
        # timestamp = message.log_time  # or message.publish_time
        # 
        # if topic not in self.timestamps:
        #     self.timestamps[topic] = []
        # self.timestamps[topic].append(timestamp)
        pass
    
    def get_data(self) -> Dict[str, List[int]]:
        """
        Get collected timestamps.
        
        Returns:
            Dictionary mapping topic names to lists of timestamps
            Format: {"topic_name": [timestamp1, timestamp2, ...]}
        """
        return self.timestamps

