"""
Logger configuration utility for Collector-Ansatz.

Provides a function to set up loggers for collectors.
Each collector should create its own logger instance using this function.
"""
import logging
import sys

class FixedWidthFormatter(logging.Formatter):
    """Custom formatter that ensures fixed-width alignment for level and name."""
    
    def format(self, record):
        # Format levelname with fixed width (8 chars, right-aligned)
        levelname = f"[{record.levelname}]"
        record.levelname = f"{levelname:>8}"
        # Format name with fixed width (25 chars, left-aligned)
        record.name = f"{record.name:^19}"
        return super().format(record)

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a configured logger for a collector.
    
    Args:
        name: Logger name (typically the collector class name)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = FixedWidthFormatter(
            ' %(levelname)s | %(asctime)s | %(name)s |    %(message)s',
            datefmt='%d-%m-%Y %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
    
    return logger

