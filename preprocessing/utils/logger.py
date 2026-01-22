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
    Automatically handles indentation based on context.
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
        
        # Automatic indentation - default indent level
        self._default_indent = 2  # Default indent in spaces
    
    def set_default_indent(self, spaces: int):
        """
        Set the default automatic indent level.
        
        Args:
            spaces: Number of spaces for default indent
        """
        self._default_indent = spaces
    
    def _format_message(self, message: str, indent: Optional[int] = None) -> str:
        """
        Format message with indentation.
        
        Args:
            message: Message to format
            indent: Optional indent level (in spaces). If None, uses default indent.
        
        Returns:
            Formatted message with indentation
        """
        if indent is None:
            indent = self._default_indent
        
        if indent > 0:
            return " " * indent + message
        return message
    
    def info(self, message: str, indent: Optional[int] = 4):
        """
        Log info message.
        
        Args:
            message: Message to log
            indent: Optional indent level (in spaces). If None, uses current indent level.
        """
        self.logger.info(self._format_message(message, indent))
    
    def debug(self, message: str, indent: Optional[int] = 12):
        """
        Log debug message.
        
        Args:
            message: Message to log
            indent: Optional indent level (in spaces). If None, uses current indent level.
        """
        self.logger.debug(self._format_message(message, indent))
    
    def warning(self, message: str, indent: Optional[int] = 0):
        """
        Log warning message.
        
        Args:
            message: Message to log
            indent: Optional indent level (in spaces). If None, uses current indent level.
        """
        self.logger.warning(self._format_message(message, indent))
    
    def error(self, message: str, indent: Optional[int] = 0):
        """
        Log error message.
        
        Args:
            message: Message to log
            indent: Optional indent level (in spaces). If None, uses current indent level.
        """
        self.logger.error(self._format_message(message, indent))
    
    def critical(self, message: str, indent: Optional[int] = 0):
        """
        Log critical message.
        
        Args:
            message: Message to log
            indent: Optional indent level (in spaces). If None, uses current indent level.
        """
        self.logger.critical(self._format_message(message, indent))
    
    def section(self, title: str, char: str = "=", width: int = 70, indent: Optional[int] = 0):
        """
        Log a section header.
        
        Args:
            title: Section title
            char: Character to use for border
            width: Width of the section
        """
        self.info(char * width, indent=indent)
        self.info(title, indent=indent)
        self.info(char * width, indent=indent)
    
    def subsection(self, title: str, char: str = "-", width: int = 60, indent: Optional[int] = 0):
        """
        Log a subsection header.
        
        Args:
            title: Subsection title
            char: Character to use for border
            width: Width of the subsection
        """
        self.info("", indent=indent)
        self.info(char * width, indent=indent)
        self.info(title, indent=indent)
        self.info(char * width, indent=indent)
    
    def success(self, message: str, indent: Optional[int] = None):
        """
        Log a success message with special formatting.
        
        Args:
            message: Success message
            indent: Optional indent level (in spaces). If None, uses current indent level.
        """
        # TODO: Add green checkmark or special formatting
        self.info(f"✓ {message}", indent=indent)
    
    def progress(self, current: int, total: int, item_name: str = "item", indent: Optional[int] = 0):
        """
        Log progress information.
        
        Args:
            current: Current item number
            total: Total number of items
            item_name: Name of the item being processed
            indent: Optional indent level (in spaces). If None, uses current indent level.
        """
        percentage = (current / total * 100) if total > 0 else 0
        self.info(f"Progress: {current}/{total} {item_name}s ({percentage:.1f}%)", indent=indent)
    
    def processor_start(self, processor_name: str, indent: Optional[int] = None):
        """
        Log processor start.
        
        Args:
            processor_name: Name of the processor
            indent: Optional indent level (in spaces). If None, uses current indent level.
        """
        self.info(f"→ Running: {processor_name}", indent=indent)
    
    def processor_skip(self, processor_name: str, reason: str = "already completed", indent: Optional[int] = None):
        """
        Log processor skip.
        
        Args:
            processor_name: Name of the processor
            reason: Reason for skipping
            indent: Optional indent level (in spaces). If None, uses current indent level.
        """
        self.info(f"⊘ Skipping: {processor_name} ({reason})", indent=indent)
    
    def processor_complete(self, processor_name: str, indent: Optional[int] = None):
        """
        Log processor completion.
        
        Args:
            processor_name: Name of the processor
            indent: Optional indent level (in spaces). If None, uses current indent level.
        """
        self.success(f"Completed: {processor_name}", indent=indent)


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

