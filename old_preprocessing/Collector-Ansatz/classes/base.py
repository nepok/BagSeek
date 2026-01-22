# collectors/base.py
from abc import ABC, abstractmethod
from pathlib import Path
import sys
import logging

# Add parent directory to path to import logger utility
sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import setup_logger

class Preprocessor(ABC):
    """
    Stateless per-message, stateful over one MCAP.
    """

    def __init__(self, *, enabled: bool = True, log_level: int = logging.DEBUG):
        self.enabled = enabled
        # Set up logger using the class name
        logger_name = self.__class__.__name__
        self.logger = setup_logger(logger_name, level=log_level)

    def wants(self, topic: str, msg_type: str) -> bool:
        """
        Fast filter: should this collector see this message?
        """
        return False

    @abstractmethod
    def on_message(self, *, topic: str, msg, timestamp_ns: int):
        """
        Called for every message that passes wants().
        """
        pass

    def finalize(self):
        """
        Called once after MCAP iteration.
        """
        pass