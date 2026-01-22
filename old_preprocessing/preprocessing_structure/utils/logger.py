"""
Logging utilities for the rosbag processing pipeline.

Provides nice, colorful, and structured logging output.
"""
import logging
import sys
from typing import Optional


class PipelineLogger:
    """
    Custom logger for the rosbag processing pipeline.
    
    Provides structured, color-coded logging with different levels
    and nice formatting for pipeline stages.
    """
    
    def __init__(self, name: str = "pipeline", level: int = logging.INFO):
        """
        Initialize the pipeline logger.
        
        Args:
            name: Logger name
            level: Logging level (default: INFO)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Only add handlers if none exist (avoid duplicate handlers)
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def section(self, title: str, char: str = "=", width: int = 70):
        """
        Log a section header.
        
        Args:
            title: Section title
            char: Character to use for border
            width: Width of the section
        """
        self.info(char * width)
        self.info(title)
        self.info(char * width)
    
    def subsection(self, title: str, char: str = "-", width: int = 60):
        """
        Log a subsection header.
        
        Args:
            title: Subsection title
            char: Character to use for border
            width: Width of the subsection
        """
        self.info("")
        self.info(char * width)
        self.info(title)
        self.info(char * width)
    
    def success(self, message: str):
        """
        Log a success message with special formatting.
        
        Args:
            message: Success message
        """
        # TODO: Add green checkmark or special formatting
        self.info(f"✓ {message}")
    
    def progress(self, current: int, total: int, item_name: str = "item"):
        """
        Log progress information.
        
        Args:
            current: Current item number
            total: Total number of items
            item_name: Name of the item being processed
        """
        percentage = (current / total * 100) if total > 0 else 0
        self.info(f"Progress: {current}/{total} {item_name}s ({percentage:.1f}%)")
    
    def processor_start(self, processor_name: str):
        """Log processor start"""
        self.info(f"→ Running: {processor_name}")
    
    def processor_skip(self, processor_name: str, reason: str = "already completed"):
        """Log processor skip"""
        self.info(f"⊘ Skipping: {processor_name} ({reason})")
    
    def processor_complete(self, processor_name: str):
        """Log processor completion"""
        self.success(f"Completed: {processor_name}")


# Global logger instance
_global_logger: Optional[PipelineLogger] = None


def get_logger(name: str = "pipeline", level: int = logging.INFO) -> PipelineLogger:
    """
    Get or create the global logger instance.
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        PipelineLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = PipelineLogger(name, level)
    
    return _global_logger


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Setup logging configuration for the pipeline.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    # TODO: Implement global logging setup
    # This can be called at the start of main.py to configure logging
    pass

