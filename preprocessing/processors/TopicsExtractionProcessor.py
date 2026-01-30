"""
Step 1: Topics Extraction

Extracts topics and types from rosbags.
Operates at rosbag level.
"""
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict
from ..abstract import RosbagProcessor
from ..core import RosbagProcessingContext
from ..utils import CompletionTracker, PipelineLogger, get_logger

class TopicsExtractionProcessor(RosbagProcessor):
    """
    Extract topics and types from a rosbag.
    
    Operates at ROSBAG level - runs once per rosbag.
    """
    
    def __init__(self, output_dir: Path, rosbags_dir: Path):
        """
        Initialize topics extraction step.
        
        Args:
            output_dir: Directory to write topics.json files
            rosbags_dir: Base directory containing rosbags
        """
        super().__init__("topics_extractor")
        self.output_dir = Path(output_dir)
        self.rosbags_dir = Path(rosbags_dir)
        self.logger: PipelineLogger = get_logger()
        self.completion_tracker = CompletionTracker(self.output_dir, processor_name="topics_extraction_processor")
    
    def process_rosbag(self, context: RosbagProcessingContext) -> Dict:
        """
        Extract all topics and their types from rosbag.
        
        Args:
            context: Processing context
        
        Returns:
            Dict with topics and types
        """

        self.logger.info(f"Extracting topics from {context.get_relative_path()}...")
        
        # Get rosbag name (relative path as string)
        rosbag_name = str(context.get_relative_path())
        
        # Construct output file path
        output_file = self.output_dir / context.get_relative_path().with_suffix('.json')
        
        # Check completion using new unified interface
        if self.completion_tracker.is_rosbag_completed(rosbag_name):
            self.logger.processor_skip(f"topics extraction for {rosbag_name}", "already completed")
            return {}
        
        topics_data = {
            "rosbag": str(context.get_relative_path()),
            "topics": {}
        }
        
        # Try metadata.yaml first
        metadata_path = context.rosbag_path / "metadata.yaml"
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
                            topics_data["topics"][topic_name] = topic_type or "unknown"
                    
                    if topics_data["topics"]:
                        self.logger.info(f"Extracted {len(topics_data['topics'])} topics from metadata.yaml")
            except Exception as e:
                self.logger.warning(f"Failed to parse metadata.yaml: {e}")
                # Fall through to ros2 bag info fallback
        
        # Fallback to ros2 bag info if no topics found
        if not topics_data["topics"]:
            self.logger.info("Falling back to ros2 bag info command...")
            try:
                result = subprocess.run(
                    ['ros2', 'bag', 'info', str(context.rosbag_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Parse topic names from output using regex
                    topics_set = set()
                    for line in result.stdout.split('\n'):
                        matches = re.findall(r'(\/[a-zA-Z0-9_][a-zA-Z0-9_/]*)', line)
                        for match in matches:
                            # Filter out false positives
                            if match.startswith('/') and len(match) > 1 and not match.startswith('//'):
                                topics_set.add(match)
                    
                    if topics_set:
                        # Store with "unknown" type
                        for topic in sorted(topics_set):
                            topics_data["topics"][topic] = "unknown"
                        self.logger.info(f"Extracted {len(topics_data['topics'])} topics from ros2 bag info")
                        self.logger.warning("Topic types not available from ros2 bag info")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                self.logger.error(f"ros2 bag info command failed: {e}")
            except Exception as e:
                self.logger.error(f"Error running ros2 bag info: {e}")
        
        # If still no topics found, log warning
        if not topics_data["topics"]:
            self.logger.warning(f"No topics extracted from {context.get_relative_path()}")
        
        # Write output (automatically mirrors rosbags directory structure)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(topics_data, f, indent=2)
        
        # Mark as completed using new unified interface
        self.completion_tracker.mark_completed(
            rosbag_name=rosbag_name,
            status="completed",
            output_files=[output_file]
        )
        
        self.logger.success(f"Extracted {len(topics_data['topics'])} topics")
        return topics_data
    
    def get_output_path(self, context: RosbagProcessingContext) -> Path:
        """Get the expected output path for this context."""
        return self.output_dir / context.get_relative_path().with_suffix('.json')

