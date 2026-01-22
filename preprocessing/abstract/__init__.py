"""
Abstract base classes for preprocessing.
"""
from .rosbag_processor import RosbagProcessor
from .mcap_processor import McapProcessor
from .hybrid_processor import HybridProcessor
from .postprocessor import PostProcessor

__all__ = [
    "RosbagProcessor",
    "McapProcessor",
    "HybridProcessor",
    "PostProcessor",
]

# Backward compatibility alias
# BaseProcessor is kept as a legacy class in processor.py
# This alias allows gradual migration
# BaseProcessor = RosbagProcessor  # Not needed since BaseProcessor still exists
