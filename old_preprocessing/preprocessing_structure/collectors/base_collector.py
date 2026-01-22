"""
Base collector class for gathering message-level data.
"""
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path


class BaseCollector(ABC):
    """
    Base class for all collectors.
    
    Collectors iterate through MCAP messages once and gather data
    that can be used by multiple processors.
    """
    
    def __init__(self, cache_enabled: bool = False):
        """
        Initialize collector.
        
        Args:
            cache_enabled: Whether to cache results in memory (default: False)
        """
        self.cache_enabled = cache_enabled
    
    @abstractmethod
    def collect_message(self, message: Any, channel: Any, schema: Any):
        """
        Collect a single message.
        
        Args:
            message: MCAP message
            channel: MCAP channel info
            schema: MCAP schema info
        """
        pass
    
    @abstractmethod
    def get_data(self) -> Any:
        """
        Get the collected data.
        
        Returns:
            Collected data (format depends on collector type)
        """
        pass
    
    def should_cache(self) -> bool:
        """
        Whether this collector's results should be cached.
        
        Override to implement custom caching logic.
        
        Returns:
            True if results should be cached, False otherwise
        """
        return self.cache_enabled

