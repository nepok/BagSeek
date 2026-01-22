# collectors/timestamp_alignment.py
from collections import defaultdict
from .base import Collector
from pathlib import Path

import numpy as np
import csv


class TimestampAlignment(Collector):
    def __init__(self, csv_path, all_topics=None):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.all_topics = all_topics

        self.topic_data = defaultdict(list)
        self.topic_types = {}

        self.logger.info(f"Timestamp Alignment initialized for {self.csv_path}")
        self.logger.info("--------------------------------")
        if all_topics:
            self.logger.debug(f"Expected topics: {len(all_topics)}")

    def wants(self, topic, msg_type):
        return True  # alle Messages relevant

    def on_message(self, *, topic, timestamp_ns):
        self.topic_data[topic].append(timestamp_ns)

    def finalize(self):
        self.logger.info(f"Starting timestamp alignment for {self.csv_path.name}")        
        if not self.topic_data:
            self.logger.warning(f"No topic data collected, skipping CSV write")
            return

        self.logger.info(f"Found {len(self.topic_data)} topics with data")

        # 1. sort (explicitly!)
        self.logger.debug("Sorting timestamps for all topics")
        for ts in self.topic_data.values():
            ts.sort()

        # 2. reference topic
        ref_topic = max(self.topic_data.items(), key=lambda x: len(x[1]))[0]
        ref_timestamps = np.array(self.topic_data[ref_topic], dtype=np.int64)
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
        self.logger.debug(f"Aligning {len(self.topic_data)} topics to reference timeline")
        self.logger.debug("--------------------------------")
        aligned_data = {}
        alignment_stats = {}
        for topic, timestamps in self.topic_data.items():
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
        self.logger.info(f"Writing CSV to {self.csv_path}")
        self._write_csv(ref_ts, aligned_data)

    def _write_csv(self, ref_ts, aligned_data):
        topics = self.all_topics if self.all_topics else list(aligned_data.keys())
        header = ['Reference Timestamp'] + topics + ['Max Distance']
        
        # Ensure output directory exists
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.csv_path, 'w', newline='') as csvfile:
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
            
            self.logger.debug(f"CSV file written successfully: {self.csv_path.stat().st_size} bytes, {len(ref_ts)} rows")
        except Exception as e:
            self.logger.error(f"Failed to write CSV file: {e}")
            raise