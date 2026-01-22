# collectors/topics.py
from .base import Collector
from pathlib import Path
import sys
import json
import subprocess
import re
import logging

# Add parent directory to path to import logger utility
sys.path.insert(0, str(Path(__file__).parent.parent))
from classes.completion_tracker import CompletionTracker


class TopicsCollector(Collector):
    """
    Collector that extracts topic names and types from rosbag metadata
    and writes them to a JSON file.
    
    This collector doesn't need to iterate messages - it reads metadata directly.
    """
    
    def __init__(self, input_path: Path, output_dir: Path):
        """
        Initialize Topics collector.
        
        Args:
            input_path: Path to the rosbag directory (input source)
            output_dir: Base directory where topics JSON files should be written
        """
        super().__init__()
        self.input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        # Construct output path based on whether it's a multi-part rosbag
        if self.input_path.name.startswith("Part_"):
            # Multi-part rosbag: output_dir / base_name / Part_x.json
            self.output_path = output_dir / self.input_path.parent.name / f"{self.input_path.name}.json"
        else:
            # Normal rosbag: output_dir / rosbag_name.json
            self.output_path = output_dir / f"{self.input_path.name}.json"
        
        self.topics = []
        self.types = {}
        
        # Initialize completion tracker
        self.completion_tracker = CompletionTracker(output_dir=output_dir)
        
        self.logger.info(f"Topics collector initialized for {self.input_path}")
        self.logger.info("--------------------------------")
        
        # Extract topics and types
        self._extract_topics_and_types()
    
    def _extract_topics_and_types(self):
        """Extract topics and types from metadata.yaml or ros2 bag info."""
        # Try metadata.yaml first
        metadata_path = self.input_path / "metadata.yaml"
        if metadata_path.exists():
            try:
                import yaml
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = yaml.safe_load(f)
                
                # Navigate to topics_with_message_count
                bag_info = metadata.get('rosbag2_bagfile_information', {})
                topics_with_count = bag_info.get('topics_with_message_count', [])
                
                if topics_with_count:
                    # Extract topic names and types
                    for topic_entry in topics_with_count:
                        topic_metadata = topic_entry.get('topic_metadata', {})
                        topic_name = topic_metadata.get('name')
                        topic_type = topic_metadata.get('type')
                        
                        if topic_name:
                            self.topics.append(topic_name)
                            if topic_type:
                                self.types[topic_name] = topic_type
                    
                    if self.topics:
                        self.topics = sorted(self.topics)
                        self.logger.info(f"Extracted {len(self.topics)} topics from metadata.yaml")
                        self.logger.debug(f"Found {len(self.types)} topic types")
                        return
            except Exception as e:
                self.logger.warning(f"Failed to parse metadata.yaml: {e}")
                # Fall through to ros2 bag info fallback
        
        # Fallback to ros2 bag info command
        self.logger.info("Falling back to ros2 bag info command")
        try:
            result = subprocess.run(
                ['ros2', 'bag', 'info', str(self.input_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse topic names from output
                topics_set = set()
                for line in result.stdout.split('\n'):
                    # Look for topic names (typically start with "/" and contain alphanumeric, underscores, slashes)
                    matches = re.findall(r'(\/[a-zA-Z0-9_][a-zA-Z0-9_/]*)', line)
                    for match in matches:
                        # Filter out common false positives
                        if match.startswith('/') and len(match) > 1 and not match.startswith('//'):
                            topics_set.add(match)
                
                if topics_set:
                    self.topics = sorted(list(topics_set))
                    self.logger.info(f"Extracted {len(self.topics)} topics from ros2 bag info")
                    # Note: types not available from ros2 bag info output
                    self.logger.warning("Topic types not available from ros2 bag info")
                    return
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"ros2 bag info command failed: {e}")
        except Exception as e:
            self.logger.error(f"Error running ros2 bag info: {e}")
        
        # If we get here, extraction failed
        if not self.topics:
            self.logger.warning("No topics extracted from metadata.yaml or ros2 bag info")
    
    def wants(self, topic: str, msg_type: str) -> bool:
        """
        This collector doesn't need to process messages.
        
        Returns:
            False (doesn't need messages)
        """
        return False
    
    def on_message(self, *, topic: str, msg, timestamp_ns: int):
        """
        Not used for this collector (doesn't iterate messages).
        """
        pass
        
    def finalize(self):
        """Write topics and types to JSON file."""
        if not self.topics:
            self.logger.warning("No topics to write, skipping JSON file creation")
            return
        
        # Check if already completed
        if self.completion_tracker.is_completed(
            rosbag_path=self.input_path,
            output_file=self.output_path
        ):
            self.logger.info(f"Topics JSON already exists for {self.input_path.name}, skipping")
            return
        
        self.logger.info(f"Writing topics JSON to {self.output_path}")
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare JSON data
            json_data = {
                "topics": self.topics,
                "types": self.types if self.types else {}
            }
            
            # Write JSON file
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            # Mark as completed
            self.completion_tracker.mark_completed(
                rosbag_path=self.input_path,
                output_file=self.output_path
            )
            
            self.logger.info(f"Successfully wrote topics JSON: {len(self.topics)} topics, {len(self.types)} types")
            self.logger.debug(f"JSON file size: {self.output_path.stat().st_size} bytes")
            self.logger.debug("--------------------------------")
        except Exception as e:
            self.logger.error(f"Failed to write topics JSON file: {e}")
            raise

