"""
Preprocessing - Simplified preprocessing pipeline structure.

Provides abstract base classes for collectors, processors, and postprocessors,
along with essential utilities for logging and completion tracking.
"""
from .abstract import RosbagProcessor, McapProcessor, HybridProcessor, PostProcessor
from .utils import CompletionTracker, PipelineLogger, get_logger, setup_logging

__all__ = [
    # Abstract classes
    "RosbagProcessor",
    "McapProcessor",
    "HybridProcessor",
    "PostProcessor",
    # Utilities
    "CompletionTracker",
    "PipelineLogger",
    "get_logger",
    "setup_logging",
]

