"""
Rosbag-level processor for extracting topic information.
"""
import json
from pathlib import Path
from typing import Dict, Any
import subprocess
import re
from ...core import Processor, ProcessingLevel, RosbagProcessingContext, CompletionTracker


class TopicsExtractor(Processor):
    """
    Extract topics and types from a rosbag.
    
    Operates at ROSBAG level - runs once per rosbag.
    """
    
    def __init__(self):
        super().__init__("topics_extractor", ProcessingLevel.ROSBAG)
        self.required_collectors = []  # No collectors needed
    
    def process(self, context: RosbagProcessingContext, data: Any) -> Dict:
        """
        Extract all topics and their types from rosbag.
        
        Args:
            context: Processing context
            data: Collector data (not used)
        
        Returns:
            Dict with topics and types
        """
        # Check completion
        output_dir = context.config.topics_dir
        completion_tracker = CompletionTracker(output_dir / "completion.json")
        
        if completion_tracker.is_completed(context):
            print(f"  ✓ Topics already extracted for {context.get_relative_path()}, skipping")
            return {}
        
        print(f"  Extracting topics from {context.get_relative_path()}...")
        
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
                        print(f"    Extracted {len(topics_data['topics'])} topics from metadata.yaml")
            except Exception as e:
                print(f"    Warning: Failed to parse metadata.yaml: {e}")
                # Fall through to ros2 bag info fallback
        
        # Fallback to ros2 bag info if no topics found
        if not topics_data["topics"]:
            print("    Falling back to ros2 bag info command...")
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
                        print(f"    Extracted {len(topics_data['topics'])} topics from ros2 bag info")
                        print(f"    Warning: Topic types not available from ros2 bag info")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"    Error: ros2 bag info command failed: {e}")
            except Exception as e:
                print(f"    Error running ros2 bag info: {e}")
        
        # If still no topics found, log warning
        if not topics_data["topics"]:
            print(f"    Warning: No topics extracted from {context.get_relative_path()}")
        
        # Write output (automatically mirrors rosbags directory structure)
        output_file = output_dir / context.get_relative_path().with_suffix('.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(topics_data, f, indent=2)
        
        # Mark as completed
        completion_tracker.mark_completed(
            context,
            output_file
        )
        
        print(f"  ✓ Extracted {len(topics_data['topics'])} topics")
        return topics_data

