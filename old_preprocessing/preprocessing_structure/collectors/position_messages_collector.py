"""
Collector for gathering position messages (bestpos topic).
"""
from typing import List
from dataclasses import dataclass
from .base_collector import BaseCollector


@dataclass
class PositionMessage:
    """Container for position message data"""
    timestamp: int
    latitude: float
    longitude: float
    altitude: float
    # Add other bestpos fields as needed


class PositionMessagesCollector(BaseCollector):
    """
    Collects all messages from bestpos topic.
    
    Used by: PositionalLookupBuilder
    """
    
    def __init__(self, cache_enabled: bool = False, bestpos_topic: str = "/bestpos"):
        super().__init__(cache_enabled)
        self.bestpos_topic = bestpos_topic
        self.positions = []
    
    def collect_message(self, message, channel, schema):
        """
        Collect a position message from bestpos topic.
        
        Args:
            message: MCAP message
            channel: MCAP channel info
            schema: MCAP schema info
        """
        # Only collect from bestpos topic
        if channel.topic != self.bestpos_topic:
            return
        
        # TODO: Implement position data extraction
        # 
        # Decode message and extract position data:
        # pos_data = decode_bestpos_message(message)
        # 
        # pos_msg = PositionMessage(
        #     timestamp=message.log_time,
        #     latitude=pos_data['latitude'],
        #     longitude=pos_data['longitude'],
        #     altitude=pos_data['altitude']
        # )
        # self.positions.append(pos_msg)
        pass
    
    def get_data(self) -> List[PositionMessage]:
        """
        Get collected position messages.
        
        Returns:
            List of PositionMessage objects
        """
        return self.positions

