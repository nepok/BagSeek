"""
Step 2: Timestamp Lookup Tables

Builds timestamp alignment/lookup tables from MCAP files.
Operates at mcap level - combines collection and processing.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..abstract import McapProcessor
from ..core import McapProcessingContext, RosbagProcessingContext
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
        # Per-MCAP metadata accumulated across reset() calls for summary.json
        self.mcap_summaries: List[Dict] = []
    
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
        return self.output_dir / context.get_relative_path() / f"{context.get_mcap_id()}.parquet"
    
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
        output_file = self.output_dir / context.get_relative_path() / f"{context.get_mcap_id()}.parquet"
        
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
        
        # 6. write parquet
        self.logger.info(f"Writing Parquet to {output_file}")
        self._write_parquet(output_file, ref_ts, aligned_data)
        
        # Mark as completed using new unified interface
        self.completion_tracker.mark_completed(
            rosbag_name=rosbag_name,
            mcap_name=mcap_name,
            status="completed",
            output_files=[output_file]
        )
        
        # Accumulate per-MCAP metadata for summary.json
        self.mcap_summaries.append({
            "mcap_id": context.get_mcap_id(),
            "count": len(ref_ts),
            "first_timestamp_ns": int(ref_ts[0]),
            "last_timestamp_ns": int(ref_ts[-1]),
        })

        self.logger.success(f"Built timestamp lookup for {context.get_mcap_name()} with {len(topic_data)} topics")
        return topic_data
    
    def _write_parquet(self, parquet_path: Path, ref_ts: np.ndarray, aligned_data: Dict[str, List[Optional[int]]]):
        """
        Write aligned timestamp data to Parquet file.

        Args:
            parquet_path: Path to output Parquet file
            ref_ts: Reference timeline timestamps
            aligned_data: Dictionary mapping topics to aligned timestamp lists
        """
        topics = self.all_topics if self.all_topics else list(aligned_data.keys())

        # Ensure output directory exists
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Build DataFrame with proper types
            data = {'Reference Timestamp': ref_ts.astype(np.int64)}

            for topic in topics:
                # Convert to nullable Int64 to handle None values
                aligned_ts = aligned_data[topic]
                data[topic] = pd.array(aligned_ts, dtype=pd.Int64Dtype())

            # Compute Max Distance column
            max_distances = []
            for i, ref_time in enumerate(ref_ts):
                distances = []
                for topic in topics:
                    aligned_ts = aligned_data[topic][i]
                    if aligned_ts is not None:
                        distances.append(abs(aligned_ts - ref_time))
                max_distances.append(max(distances) if distances else None)

            data['Max Distance'] = pd.array(max_distances, dtype=pd.Int64Dtype())

            df = pd.DataFrame(data)
            df.to_parquet(parquet_path, engine='pyarrow', index=False)

            self.logger.debug(f"Parquet file written successfully: {parquet_path.stat().st_size} bytes, {len(ref_ts)} rows")
        except Exception as e:
            self.logger.error(f"Failed to write Parquet file: {e}")
            raise

    def finalize_rosbag(self, rosbag_context: RosbagProcessingContext) -> None:
        """Write summary.json for a rosbag after all its MCAPs have been processed.

        For MCAPs that were skipped (already complete), loads metadata from
        pyarrow parquet file footers (no row data read). Then aggregates all
        per-MCAP metadata into a single summary.json.

        Call this once per rosbag, after the MCAP loop. Resets internal state
        so the processor is ready for the next rosbag.
        """
        # Fill in metadata for any MCAPs not processed in this run
        collected_ids = {s["mcap_id"] for s in self.mcap_summaries}
        for mcap_path in rosbag_context.mcap_files:
            mcap_context = McapProcessingContext(
                rosbag_path=rosbag_context.rosbag_path,
                mcap_path=mcap_path,
                config=rosbag_context.config,
            )
            if mcap_context.get_mcap_id() not in collected_ids:
                self._load_mcap_summary_from_parquet(mcap_context)

        # Write summary.json
        if self.mcap_summaries:
            self._write_summary(rosbag_context)

        # Reset for the next rosbag
        self.mcap_summaries = []

    def _load_mcap_summary_from_parquet(self, context: McapProcessingContext) -> None:
        """Load summary metadata from an existing parquet file (for skipped MCAPs).

        Uses pyarrow parquet metadata (file footer only) to get row count and
        Reference Timestamp column statistics without reading any row data.
        """
        parquet_path = self.get_output_path(context)
        if not parquet_path.exists():
            return

        try:
            pf = pq.ParquetFile(parquet_path)
            metadata = pf.metadata
            row_count = metadata.num_rows
            if row_count == 0:
                return

            # Get min/max from row group statistics for Reference Timestamp
            schema = pf.schema_arrow
            col_idx = schema.get_field_index('Reference Timestamp')
            file_min = None
            file_max = None
            for rg_idx in range(metadata.num_row_groups):
                col_stats = metadata.row_group(rg_idx).column(col_idx).statistics
                if col_stats is not None and col_stats.has_min_max:
                    if file_min is None or col_stats.min < file_min:
                        file_min = col_stats.min
                    if file_max is None or col_stats.max > file_max:
                        file_max = col_stats.max

            # Fallback: read just the column if statistics aren't available
            if file_min is None or file_max is None:
                table = pq.read_table(parquet_path, columns=['Reference Timestamp'])
                ts = table.column('Reference Timestamp')
                file_min = ts[0].as_py()
                file_max = ts[-1].as_py()

            self.mcap_summaries.append({
                "mcap_id": context.get_mcap_id(),
                "count": row_count,
                "first_timestamp_ns": int(file_min),
                "last_timestamp_ns": int(file_max),
            })
        except Exception as e:
            self.logger.warning(f"Failed to read parquet metadata from {parquet_path}: {e}")

    def _write_summary(self, rosbag_context: RosbagProcessingContext) -> None:
        """Write summary.json from accumulated mcap_summaries."""
        sorted_summaries = sorted(self.mcap_summaries, key=lambda s: int(s["mcap_id"]))

        mcap_ranges = []
        cumulative_index = 0
        global_first = None
        global_last = None

        for summary in sorted_summaries:
            mcap_ranges.append({
                "mcap_id": summary["mcap_id"],
                "start_index": cumulative_index,
                "count": summary["count"],
                "first_timestamp_ns": summary["first_timestamp_ns"],
                "last_timestamp_ns": summary["last_timestamp_ns"],
            })
            first = summary["first_timestamp_ns"]
            last = summary["last_timestamp_ns"]
            if global_first is None or first < global_first:
                global_first = first
            if global_last is None or last > global_last:
                global_last = last
            cumulative_index += summary["count"]

        summary_data = {
            "total_count": cumulative_index,
            "first_timestamp_ns": global_first,
            "last_timestamp_ns": global_last,
            "mcap_ranges": mcap_ranges,
        }

        output_dir = self.output_dir / rosbag_context.get_relative_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        self.logger.success(f"Wrote timestamp summary to {summary_path}")

