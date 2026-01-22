"""
Rosbag Processing Pipeline

A modular, resumable pipeline for multi-level rosbag processing with:
- Core infrastructure (base classes, pipeline orchestrator, completion tracking)
- Collectors (reusable message-level data gatherers)
- Processors (rosbag-level and mcap-level)
- Postprocessors (aggregation and analysis)
"""

# Configuration
from .config import Config

# Core
from .core import (
    ProcessingLevel,
    RosbagProcessingContext,
    McapProcessingContext,
    Processor,
    CompletionTracker,
)

# Collectors
from .collectors import (
    BaseCollector,
    TimestampsCollector,
    ImageMessagesCollector,
    ImageMessage,
    PositionMessagesCollector,
    PositionMessage,
)

# Processors
from .processors import (
    TopicsExtractor,
    FencepostCalculator,
    TimestampAlignmentBuilder,
    PositionalLookupBuilder,
    EmbeddingGenerator,
)

# Postprocessors
from .postprocessors import (
    AdjacentSimilarityAnalyzer,
    PositionalLookupAggregator,
)

# Utils
from .utils import (
    PipelineLogger,
    get_logger,
    setup_logging,
    get_all_rosbags,
    get_all_mcaps,
    validate_rosbag,
    validate_mcap_file,
    get_rosbag_info,
)

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "Config",
    # Core
    "ProcessingLevel",
    "ProcessingContext",
    "Processor",
    "CompletionTracker",
    # Collectors
    "BaseCollector",
    "TimestampsCollector",
    "ImageMessagesCollector",
    "ImageMessage",
    "PositionMessagesCollector",
    "PositionMessage",
    # Processors
    "TopicsExtractor",
    "FencepostCalculator",
    "TimestampAlignmentBuilder",
    "PositionalLookupBuilder",
    "EmbeddingGenerator",
    # Postprocessors
    "AdjacentSimilarityAnalyzer",
    "PositionalLookupAggregator",
    # Utils
    "PipelineLogger",
    "get_logger",
    "setup_logging",
    "get_all_rosbags",
    "get_all_mcaps",
    "validate_rosbag",
    "validate_mcap_file",
    "get_rosbag_info",
]

