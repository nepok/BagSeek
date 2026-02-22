"""
Step 4: Image Previews

Creates representative preview images from rosbags.
Contains rosbag processor, mcap processor, and postprocessor.
"""
import json
import io
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from ..abstract import HybridProcessor
from ..core import RosbagProcessingContext, McapProcessingContext
from ..utils import CompletionTracker, PipelineLogger, get_logger


class ImageTopicPreviewsProcessor(HybridProcessor):
    """
    Calculate fencepost mapping and collect representative images.
    
    Hybrid processor that:
    - Calculates fencepost mapping before MCAP iteration (which MCAPs contain which parts 1-7)
    - Extracts topic info from MCAP summary before message iteration
    - Collects images at target indices during MCAP iteration
    - Saves collected images after all MCAPs
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize image previews processor.
        
        Args:
            output_dir: Directory to write preview files
        """
        super().__init__("image_topic_previews_processor")
        self.output_dir = Path(output_dir)
        self.logger: PipelineLogger = get_logger()
        self.completion_tracker = CompletionTracker(self.output_dir, processor_name="image_topic_previews_processor")
        
        # Fencepost mapping: MCAP ID -> [(part_idx, percentage), ...]
        self.fencepost_mapping: Dict[str, List[Tuple[int, float]]] = {}
        
        # Topic info per MCAP: {topic: {"topic_id": id, "message_count": count}}
        self.topic_info: Dict[str, Dict[str, int]] = {}
        
        # Target indices: {topic: {part_idx: target_message_index}}
        self.target_indices: Dict[str, Dict[int, int]] = {}
        
        # Current message index per topic during iteration
        self.topic_index: Dict[str, int] = defaultdict(int)
        
        # Collected fencepost images
        self.collected_images: List[Dict[str, Any]] = []
        
        # Current MCAP being processed
        self.current_mcap_id: Optional[str] = None
        
        # Set of image topics
        self.image_topics: set = set()
        
        # Track actual collection counts per topic (only images actually collected, not all processed)
        self.actual_collection_counts: Dict[str, int] = defaultdict(int)
    
    def process_rosbag_before_mcaps(self, context: RosbagProcessingContext) -> None:
        """
        Calculate fencepost mapping before MCAP iteration.
        
        Args:
            context: RosbagProcessingContext
        """
        # Check if already completed - skip fencepost calculation if so
        rosbag_name = str(context.get_relative_path())
        if self.completion_tracker.is_rosbag_completed(rosbag_name):
            # Already completed, skip fencepost calculation
            self.fencepost_mapping = {}
            self.collected_images = []
            self.topic_index = defaultdict(int)
            self.current_mcap_id = None
            self.image_topics = set()
            return
        
        # Get MCAP files
        mcap_files = context.mcap_files
        if not mcap_files:
            self.logger.warning(f"No MCAP files found in {context.get_relative_path()}")
            return
        
        # Calculate fencepost mapping
        self.fencepost_mapping = self._calculate_fencepost_mapping(len(mcap_files))
        
        # Log mapping details
        self.logger.info(f"Calculating fencepost mapping for {context.get_relative_path()}...")
        self.logger.info(f"  {len(mcap_files)} MCAPs → 7 parts distributed:")
        for mcap_idx, parts in sorted(self.fencepost_mapping.items(), key=lambda x: int(x[0])):
            parts_str = ", ".join([f"part {p[0]} at {p[1]:.1%}" for p in parts])
            self.logger.info(f"    MCAP {mcap_idx}: {parts_str}")
        
        # Initialize state
        self.collected_images = []
        self.topic_index = defaultdict(int)
        self.current_mcap_id = None
        self.image_topics = set()
    
    def _calculate_fencepost_mapping(self, num_mcaps: int) -> Dict[str, List[Tuple[int, float]]]:
        """
        Calculate fencepost mapping for distributing parts across MCAPs.
        
        Args:
            num_mcaps: Number of MCAP files
        
        Returns:
            Dict mapping mcap_idx (as string) -> [(part_idx, percentage), ...]
        """
        num_parts = 8  # 7 fencepost images (parts 1-7)
        step_length = num_mcaps / num_parts
        
        mcap_to_parts = {}
        
        for part_idx in range(1, num_parts):  # parts 1 to 7
            position = step_length * part_idx
            mcap_idx = int(position)  # Floor to get MCAP index
            percentage = position - mcap_idx  # Fractional part is the percentage
            
            mcap_key = str(mcap_idx)
            if mcap_key not in mcap_to_parts:
                mcap_to_parts[mcap_key] = []
            
            mcap_to_parts[mcap_key].append((part_idx, percentage))
        
        return mcap_to_parts
    
    def is_mcap_skippable(self, mcap_id: str) -> bool:
        """
        Check if an MCAP can be skipped (doesn't contain any fencepost parts).

        Args:
            mcap_id: MCAP ID string

        Returns:
            True if MCAP can be skipped, False if it needs processing
        """
        return mcap_id not in self.fencepost_mapping

    def is_mcap_complete(self, context) -> bool:
        """Uniform interface — MCAP is complete if it's not in the fencepost mapping."""
        return self.is_mcap_skippable(context.get_mcap_id())
    
    def has_fencepost_parts(self) -> bool:
        """
        Check if the current MCAP has any fencepost parts to collect.
        
        Should be called after setup_fencepost_targets_for_mcap().
        
        Returns:
            True if there are target indices to collect, False otherwise
        """
        return len(self.target_indices) > 0
    
    def setup_fencepost_targets_for_mcap(self, reader: Any, mcap_context: McapProcessingContext, is_only_processor: bool) -> bool:
        """
        Extract topic info from MCAP summary and calculate target indices for fencepost collection.
        
        Called from main.py before message iteration.
        If MCAP is not in fencepost_mapping, target_indices will be empty,
        so no images will be collected for this MCAP.
        
        Args:
            reader: SeekingReader instance with MCAP file open
            mcap_context: McapProcessingContext for logging
            is_only_processor: True if this is the only active processor
        
        Returns:
            True if this MCAP should be processed, False if it can be skipped
        """
        # Extract topic info from summary
        channels = reader.get_summary().channels
        channel_message_counts = reader.get_summary().statistics.channel_message_counts
        
        # Create topic_info dict
        all_topic_info = {
            channel.topic: {
                "topic_id": channel.id,
                "message_count": channel_message_counts.get(channel.id, 0)
            }
            for channel_id, channel in channels.items()
        }
        
        # Filter for image topics only
        self.topic_info = {
            topic: info for topic, info in all_topic_info.items()
            if self._is_image_topic(topic)
        }
        
        # Update image_topics set
        self.image_topics = set(self.topic_info.keys())
        
        # Calculate target indices for this MCAP
        # If MCAP is not in fencepost_mapping, this will return empty dict
        self.target_indices = self._calculate_target_indices()
        
        # Reset topic_index for this MCAP
        self.topic_index = defaultdict(int)
        
        # Check if we should skip this MCAP entirely
        if is_only_processor and not self.has_fencepost_parts():
            self.logger.processor_skip(
                f"{self.name} for {mcap_context.get_mcap_name()}",
                "no fencepost parts"
            )
            return False
        
        if self.topic_info:
            num_targets = sum(len(indices) for indices in self.target_indices.values())
            if num_targets > 0:
                self.logger.info(f"  Found {len(self.topic_info)} image topic(s), {num_targets} target indices")
            else:
                self.logger.debug(f"  Found {len(self.topic_info)} image topic(s), but no fencepost parts in this MCAP")
        
        return True
    
    def _calculate_target_indices(self) -> Dict[str, Dict[int, int]]:
        """
        Calculate target message indices for each topic and part.
        
        Returns:
            Dict mapping topic -> {part_idx: target_message_index}
        """
        target_indices = {}
        
        # Check if this MCAP has parts to extract
        mcap_key = str(self.current_mcap_id)
        if mcap_key not in self.fencepost_mapping:
            return target_indices
        
        parts = self.fencepost_mapping[mcap_key]
        
        for part_idx, percentage in parts:
            for topic, info in self.topic_info.items():
                total_messages = info["message_count"]
                target_idx = int(total_messages * percentage)
                
                if topic not in target_indices:
                    target_indices[topic] = {}
                target_indices[topic][part_idx] = target_idx
        
        return target_indices
    
    def wants_message(self, topic: str, msg_type: str) -> bool:
        """
        Check if topic is an image topic and has target indices to collect.
        
        Returns False if there are no target indices for this MCAP (no fencepost parts),
        which allows skipping message processing entirely.
        """
        # If no target indices, skip all messages (MCAP has no fencepost parts)
        if not self.target_indices:
            return False
        
        # Check if this topic is an image topic and has target indices
        return self._is_image_topic(topic) and topic in self.target_indices
    
    def _is_image_topic(self, topic: str) -> bool:
        """Check if a topic is an image topic."""
        return "image" in topic.lower() or "camera" in topic.lower()
    
    def collect_message(self, message: Any, channel: Any, schema: Any, ros2_msg: Any) -> None:
        """
        Collect a message if it matches a target index.
        
        Args:
            message: MCAP message
            channel: MCAP channel info
            schema: MCAP schema info
            ros2_msg: Decoded ROS2 message
        """
        topic = channel.topic
        
        # Skip topics we don't care about
        if topic not in self.topic_info:
            return
        
        # Get current index for this topic
        current_idx = self.topic_index[topic]
        
        # Calculate progress percentage for logging
        message_count = self.topic_info[topic]["message_count"]
        if message_count > 0:
            progress = (current_idx / message_count) * 100
            # Log at key milestones (every 25%)
            if current_idx > 0 and current_idx % max(1, message_count // 4) == 0:
                self.logger.debug(f"    {topic}: {progress:.1f}% ({current_idx}/{message_count})")
        
        # Check if this message index matches any target index for this topic
        if topic in self.target_indices:
            for part_idx, target_idx in self.target_indices[topic].items():
                if current_idx == target_idx:
                    # This is a fencepost image! Convert to PIL Image
                    try:
                        pil_image = self._ros2_image_to_pil(ros2_msg)
                        if pil_image is not None:
                            fencepost_img = {
                                "topic": topic,
                                "timestamp": message.log_time,
                                "part_index": part_idx,
                                "mcap_id": self.current_mcap_id,
                                "message_index": current_idx,
                                "image": pil_image  # Store PIL Image
                            }
                            self.collected_images.append(fencepost_img)
                            self.actual_collection_counts[topic] += 1
                            self.logger.debug(f"    Collected part {part_idx} from {topic} at index {current_idx}")
                    except Exception as e:
                        self.logger.warning(f"    Failed to convert image from {topic} at index {current_idx}: {e}")
        
        # Increment index for this topic
        self.topic_index[topic] += 1
    
    def get_collection_counts(self) -> Dict[str, int]:
        """
        Get the actual number of images collected per topic for the current MCAP.
        
        Returns:
            Dictionary mapping topic names to count of actually collected images
        """
        return dict(self.actual_collection_counts)
    
    def reset_collection_counts(self) -> None:
        """Reset collection counts (called at start of each MCAP)."""
        self.actual_collection_counts.clear()
    
    def _ros2_image_to_pil(self, ros2_msg: Any) -> Optional[Image.Image]:
        """
        Convert ROS2 image message to PIL Image.
        
        Supports:
        - sensor_msgs/msg/CompressedImage
        - sensor_msgs/msg/Image
        
        Args:
            ros2_msg: Decoded ROS2 message
        
        Returns:
            PIL Image or None if conversion fails
        """
        msg_type = type(ros2_msg).__name__
        
        try:
            # Handle CompressedImage
            if "CompressedImage" in msg_type:
                # CompressedImage has data as bytes
                img_data = bytes(ros2_msg.data)
                pil_image = Image.open(io.BytesIO(img_data))
                return pil_image.convert('RGB')
            
            # Handle raw Image
            elif "Image" in msg_type and hasattr(ros2_msg, 'encoding'):
                if not all(hasattr(ros2_msg, attr) for attr in ['encoding', 'width', 'height', 'data']):
                    self.logger.warning(f"Image message missing required attributes")
                    return None
                
                encoding = ros2_msg.encoding
                width = ros2_msg.width
                height = ros2_msg.height
                data = bytes(ros2_msg.data)
                
                # Map encoding to channels
                channels_map = {
                    "mono8": 1,
                    "rgb8": 3,
                    "bgr8": 3,
                    "rgba8": 4,
                    "bgra8": 4
                }
                
                channels = channels_map.get(encoding)
                if channels is None:
                    self.logger.warning(f"Unsupported image encoding: {encoding}")
                    return None
                
                # Convert bytes to numpy array
                img_data = np.frombuffer(data, dtype=np.uint8)
                img_data = img_data.reshape((height, width, channels))
                
                # Convert BGR to RGB if needed
                if encoding == "bgr8":
                    img_data = img_data[:, :, ::-1]  # BGR to RGB
                elif encoding == "bgra8":
                    img_data = img_data[:, :, [2, 1, 0, 3]]  # BGRA to RGBA
                
                # Convert to PIL Image
                if channels == 1:
                    pil_image = Image.fromarray(img_data, mode='L')
                elif channels == 3:
                    pil_image = Image.fromarray(img_data, mode='RGB')
                elif channels == 4:
                    pil_image = Image.fromarray(img_data, mode='RGBA').convert('RGB')
                else:
                    return None
                
                return pil_image
            
            else:
                self.logger.warning(f"Unsupported message type: {msg_type}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to convert ROS2 image to PIL: {e}")
            return None
    
    def reset(self) -> None:
        """Reset state before each MCAP iteration."""
        self.topic_index = defaultdict(int)
        self.target_indices = {}
        self.topic_info = {}
        self.actual_collection_counts.clear()
    
    def get_collection_counts(self) -> Dict[str, int]:
        """
        Get the actual number of images collected per topic for the current MCAP.
        
        Returns:
            Dictionary mapping topic names to count of actually collected images
        """
        return dict(self.actual_collection_counts)
    
    def get_output_path(self, context: RosbagProcessingContext) -> Path:
        """Get the expected output path for this context."""
        # Return directory path where stitched images will be saved
        return self.output_dir / context.get_relative_path()
    
    def process_rosbag_after_mcaps(self, context: RosbagProcessingContext) -> Dict:
        """
        Stitch collected images per topic and save after all MCAPs.
        
        Args:
            context: RosbagProcessingContext
        
        Returns:
            Preview info dictionary
        """
        # Construct output directory
        output_dir = self.output_dir / context.get_relative_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check completion (using first stitched image as completion marker)
        if not self.collected_images:
            self.logger.warning(f"No images collected for {context.get_relative_path()}")
            # Mark as complete so this rosbag isn't retried on every run
            rosbag_name_early = str(context.get_relative_path())
            self.completion_tracker.mark_completed(
                rosbag_name=rosbag_name_early,
                status="completed",
            )
            return {}
        
        # Get rosbag name
        rosbag_name = str(context.get_relative_path())
        
        # Check completion using new unified interface
        if self.completion_tracker.is_rosbag_completed(rosbag_name):
            self.logger.processor_skip(f"image previews for {rosbag_name}", "already completed")
            return {}
        
        self.logger.info(f"Stitching image previews for {context.get_relative_path()}...")
        
        # Group images by topic
        images_by_topic = defaultdict(list)
        for img in self.collected_images:
            images_by_topic[img["topic"]].append(img)
        
        # Extract unique topics
        topics = sorted(images_by_topic.keys())
        
        # Stitch images per topic
        stitched_files = []
        for topic in topics:
            topic_images = sorted(images_by_topic[topic], key=lambda x: x["part_index"])
            
            if not topic_images:
                continue
            
            # Stitch images into grid
            stitched_image = self._stitch_images(topic_images)
            
            if stitched_image is None:
                self.logger.warning(f"Failed to stitch images for {topic}")
                continue
            
            # Save stitched image
            # Sanitize topic name for filename
            topic_filename = topic.replace('/', '__').replace('\\', '__')
            stitched_file = output_dir / f"{topic_filename}.jpg"
            stitched_image.save(stitched_file, 'JPEG', quality=95)
            stitched_files.append(stitched_file)
            
            self.logger.info(f"  Stitched {len(topic_images)} images for {topic} -> {stitched_file.name}")
        
        # Mark as completed with all stitched files using new unified interface
        if stitched_files:
            self.completion_tracker.mark_completed(
                rosbag_name=rosbag_name,
                status="completed",
                output_files=stitched_files
            )
        
        self.logger.success(f"Stitched {len(stitched_files)} preview(s) from {len(self.collected_images)} images across {len(topics)} topic(s)")
        
        # Clear collected images for next rosbag
        self.collected_images = []
        
        return {}
    
    def _stitch_images(self, images: List[Dict[str, Any]]) -> Optional[Image.Image]:
        """
        Stitch multiple images into a grid.
        
        Args:
            images: List of image dicts with "image" (PIL Image) and "part_index"
        
        Returns:
            Stitched PIL Image or None if stitching fails
        """
        if not images:
            return None
        
        # Extract PIL Images and sort by part_index
        pil_images = []
        for img_dict in sorted(images, key=lambda x: x["part_index"]):
            if "image" in img_dict and img_dict["image"] is not None:
                pil_images.append(img_dict["image"])
        
        if not pil_images:
            return None
        
        # Calculate grid dimensions
        num_images = len(pil_images)
        # For 7 images: 1 row of 7, or 2 rows (4+3)
        if num_images <= 7:
            cols = num_images
            rows = 1
        else:
            cols = 4  # Max 4 columns
            rows = (num_images + cols - 1) // cols
        
        # Get max dimensions
        max_width = max(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)
        
        # Resize all images to same size
        resized_images = []
        for img in pil_images:
            # Use LANCZOS for high-quality resampling (compatible with older PIL versions)
            resized = img.resize((max_width, max_height), Image.LANCZOS)
            resized_images.append(resized)
        
        # Create output image
        grid_width = cols * max_width
        grid_height = rows * max_height
        grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        # Paste images into grid
        for idx, img in enumerate(resized_images):
            row = idx // cols
            col = idx % cols
            x = col * max_width
            y = row * max_height
            grid.paste(img, (x, y))
        
        return grid

