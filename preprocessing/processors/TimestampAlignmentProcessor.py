"""
Step 2: Timestamp Lookup Tables

Builds timestamp alignment/lookup tables from MCAP files.
Operates at mcap level - combines collection and processing.
"""
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import csv

from ..abstract import McapProcessor
from ..core import McapProcessingContext
from ..utils import CompletionTracker, PipelineLogger, get_logger


class TimestampAlignmentProcessor(McapProcessor):
    """
    Build timestamp lookup table for an MCAP.
    
    Combines collection and processing in a single class.
    Implements collector methods (wants_message, collect_message, reset, get_data)
    to collect timestamps during single-pass MCAP iteration in main.py.
    Then processes collected data in process_mcap() method.
    
    Operates at MCAP level - runs once per mcap file.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize timestamp lookup step.
        
        Args:
            output_dir: Directory to write timestamp lookup files
        """
        super().__init__("timestamp_alignment_processor")
        self.output_dir = Path(output_dir)
        self.logger: PipelineLogger = get_logger()
        self.completion_tracker = CompletionTracker(self.output_dir, processor_name="timestamp_alignment_processor")
        # Internal state for collected timestamps
        self.timestamps = defaultdict(list)
    
    # Collector methods (called during MCAP iteration in main.py)
    def wants_message(self, topic: str, msg_type: str) -> bool:
        """Collect all messages."""
        return True
    
    def collect_message(self, message: Any, channel: Any, schema: Any, ros2_msg: Any):
        """
        Collect a timestamp from a message.
        
        Called during MCAP message iteration in main.py.
        
        Args:
            message: MCAP message
            channel: MCAP channel info
            schema: MCAP schema info
        """
        topic = channel.topic
        timestamp_ns = message.log_time  # timestamp in nanoseconds
        self.timestamps[channel.topic].append(timestamp_ns)
    
    def get_data(self) -> Dict[str, List[int]]:
        """
        Get collected timestamps.
        
        Returns:
            Dictionary mapping topic names to lists of timestamps
        """
        return self.timestamps
    
    def reset(self):
        """Reset collector state before each MCAP iteration."""
        self.timestamps = defaultdict(list)
    
    def get_output_path(self, context: McapProcessingContext) -> Path:
        """Get the expected output path for this context."""
        return self.output_dir / context.get_relative_path() / f"{context.get_mcap_id()}.csv"
    
    # Processor method (called after MCAP iteration in main.py)
    def process_mcap(self, context: McapProcessingContext, all_topics: Optional[list[str]] = None) -> Dict:
        """
        Process collected timestamps and write to CSV.
        
        Called after MCAP message iteration is complete.
        Uses self.timestamps that were collected during iteration.
        
        Args:
            context: Processing context (should contain mcap_path, rosbag_path)
            all_topics: Optional list of all expected topics (for CSV column ordering)
        
        Returns:
            Timestamp lookup table
        """
        self.all_topics = all_topics

        # Construct output file path before checking completion (for fallback check)
        output_file = self.output_dir / context.get_relative_path() / f"{context.get_mcap_id()}.csv"
        
        # Get rosbag name and mcap name
        rosbag_name = str(context.get_relative_path())
        mcap_name = context.get_mcap_name()
        
        # Check completion using new unified interface
        if self.completion_tracker.is_mcap_completed(rosbag_name, mcap_name):
            self.logger.processor_skip(f"timestamp lookup for {mcap_name}", "already completed")
            return {}
        
        self.logger.info(f"Starting timestamp alignment for {context.get_mcap_name()}")
        
        # Process collected timestamps (already collected during iteration)
        topic_data = self.get_data()
        
        if not topic_data:
            self.logger.warning(f"No topic data collected, skipping CSV write")
            return {}
        
        self.logger.info(f"Found {len(topic_data)} topics with data")
        
        # 1. sort (explicitly!)
        self.logger.debug("Sorting timestamps for all topics")
        for ts in topic_data.values():
            ts.sort()
        
        # 2. reference topic
        ref_topic = max(topic_data.items(), key=lambda x: len(x[1]))[0]
        ref_timestamps = np.array(topic_data[ref_topic], dtype=np.int64)
        self.logger.info(f"Selected reference topic: {ref_topic} ({len(ref_timestamps)} messages)")
        
        # 3. reference timeline
        if len(ref_timestamps) < 2:
            self.logger.warning(f"Reference topic has < 2 messages, using raw timestamps")
            ref_ts = ref_timestamps
        else:
            diffs = np.diff(ref_timestamps)
            mean_interval = np.mean(diffs)
            refined_interval = mean_interval / 2.0
            
            ref_start = ref_timestamps[0]
            ref_end = ref_timestamps[-1]
            ref_ts = np.arange(ref_start, ref_end, refined_interval).astype(np.int64)
            self.logger.debug(f"Created reference timeline: {len(ref_ts)} points (interval: {refined_interval/1e9:.3f}s)")
        
        # 4. alignment
        self.logger.debug(f"Aligning {len(topic_data)} topics to reference timeline")
        self.logger.debug("--------------------------------")
        aligned_data = {}
        alignment_stats = {}
        for topic, timestamps in topic_data.items():
            aligned = []
            aligned_count = 0
            for ref_time in ref_ts:
                closest = min(timestamps, key=lambda x: abs(x - ref_time))
                if abs(closest - ref_time) < int(1e8):  # 100ms threshold
                    aligned.append(closest)
                    aligned_count += 1
                else:
                    aligned.append(None)
            aligned_data[topic] = aligned
            alignment_stats[topic] = (aligned_count, len(ref_ts))
        
        # Log alignment statistics
        self.logger.debug("Alignment statistics:")
        for topic, (aligned, total) in alignment_stats.items():
            percentage = (aligned / total * 100) if total > 0 else 0
            self.logger.debug(f"{topic}: {aligned}/{total} aligned ({percentage:.1f}%)")
        self.logger.debug("--------------------------------")
        
        # 5. missing topics
        if self.all_topics:
            missing_topics = [t for t in self.all_topics if t not in aligned_data]
            if missing_topics:
                self.logger.warning(f"Adding {len(missing_topics)} missing topics with None values")
                for topic in missing_topics:
                    aligned_data[topic] = [None] * len(ref_ts)
        
        # 6. write csv
        #print("output_dir:", self.output_dir, "context:", context.get_relative_path(), "mcap_name:", context.get_mcap_name())
        self.logger.info(f"Writing CSV to {output_file}")
        self._write_csv(output_file, ref_ts, aligned_data)
        
        # Mark as completed using new unified interface
        self.completion_tracker.mark_completed(
            rosbag_name=rosbag_name,
            mcap_name=mcap_name,
            status="completed",
            output_files=[output_file]
        )
        
        self.logger.success(f"Built timestamp lookup for {context.get_mcap_name()} with {len(topic_data)} topics")
        return topic_data
    
    def _write_csv(self, csv_path: Path, ref_ts: np.ndarray, aligned_data: Dict[str, List[Optional[int]]]):
        """
        Write aligned timestamp data to CSV file.
        
        Args:
            csv_path: Path to output CSV file
            ref_ts: Reference timeline timestamps
            aligned_data: Dictionary mapping topics to aligned timestamp lists
        """
        topics = self.all_topics if self.all_topics else list(aligned_data.keys())
        header = ['Reference Timestamp'] + topics + ['Max Distance']
        
        # Ensure output directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                
                for i, ref_time in enumerate(ref_ts):
                    row = [int(ref_time)]
                    row_values = []
                    
                    for topic in topics:
                        aligned_ts = aligned_data[topic][i]
                        if aligned_ts is not None:
                            row.append(int(aligned_ts))
                            row_values.append(abs(aligned_ts - ref_time))
                        else:
                            row.append("")
                    
                    max_dist = max(row_values) if row_values else ""
                    row.append(int(max_dist) if max_dist != "" else "")
                    writer.writerow(row)
            
            self.logger.debug(f"CSV file written successfully: {csv_path.stat().st_size} bytes, {len(ref_ts)} rows")
        except Exception as e:
            self.logger.error(f"Failed to write CSV file: {e}")
            raise

