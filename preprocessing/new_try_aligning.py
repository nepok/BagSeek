

import os
import numpy as np
import json
from collections import defaultdict
from rosbags.rosbag2 import Reader  # type: ignore
from tqdm import tqdm

# Constants
BASE_DIR = "/mnt/data/bagseek/flask-backend/src"
ROSBAGS_DIR = "/mnt/data/rosbags"
ROSBAG_NAME = "output_bag"
LOOKUP_TABLES_DIR = os.path.join(BASE_DIR, "lookup_tables_new")
TOPICS_DIR = os.path.join(BASE_DIR, "topics_new")
MAX_TIME_DIFF_NS = int(1e8)  # Max allowable distance from reference timestamp (~100ms)

def load_rosbag_timestamps(rosbag_path):
    topic_timestamps = defaultdict(list)
    topic_types = {}
    with Reader(rosbag_path) as reader:
        for connection, timestamp, _ in tqdm(reader.messages(), desc="Reading messages"):
            topic_timestamps[connection.topic].append(timestamp)
            topic_types[connection.topic] = connection.msgtype
    return topic_timestamps, topic_types

def determine_reference_topic(topic_timestamps):
    return max(topic_timestamps.items(), key=lambda x: len(x[1]))[0]

def create_reference_timestamps(timestamps, factor=2):
    timestamps = sorted(set(timestamps))
    diffs = np.diff(timestamps)
    mean_interval = np.mean(diffs)
    refined_interval = mean_interval / factor
    ref_start = timestamps[0]
    ref_end = timestamps[-1]
    return np.arange(ref_start, ref_end, refined_interval).astype(np.int64)

def align_topic_to_reference(topic_ts, ref_ts, max_diff=MAX_TIME_DIFF_NS):
    aligned = []
    topic_idx = 0
    for rt in ref_ts:
        while topic_idx + 1 < len(topic_ts) and abs(topic_ts[topic_idx + 1] - rt) < abs(topic_ts[topic_idx] - rt):
            topic_idx += 1
        closest = topic_ts[topic_idx]
        if abs(closest - rt) <= max_diff:
            aligned.append(closest)
        else:
            aligned.append(None)
    return aligned

def process_rosbag():
    rosbag_path = os.path.join(ROSBAGS_DIR, ROSBAG_NAME)
    topic_timestamps, topic_types = load_rosbag_timestamps(rosbag_path)

    if not topic_timestamps:
        print("No messages found.")
        return

    # Save topic list and types
    os.makedirs(TOPICS_DIR, exist_ok=True)
    with open(os.path.join(TOPICS_DIR, f"{ROSBAG_NAME}.json"), "w") as f:
        json.dump({"topics": list(topic_timestamps.keys()), "types": topic_types}, f, indent=2)

    # Find reference topic and generate ref timestamps
    ref_topic = determine_reference_topic(topic_timestamps)
    ref_ts = create_reference_timestamps(topic_timestamps[ref_topic], factor=2)

    print(f"Using reference topic: {ref_topic}, total ref timestamps: {len(ref_ts)}")

    # Align other topics
    aligned_data = {}
    for topic, timestamps in tqdm(topic_timestamps.items(), desc="Aligning topics"):
        timestamps = sorted(timestamps)
        aligned = align_topic_to_reference(timestamps, ref_ts)
        aligned_data[topic] = aligned

    # Write alignment to CSV
    os.makedirs(LOOKUP_TABLES_DIR, exist_ok=True)
    csv_path = os.path.join(LOOKUP_TABLES_DIR, f"{ROSBAG_NAME}.csv")

    import csv
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        header = ["Reference Timestamp"] + list(aligned_data.keys())
        writer.writerow(header)
        for i, rt in enumerate(ref_ts):
            row = [rt] + [aligned_data[topic][i] if aligned_data[topic][i] is not None else "None" for topic in aligned_data]
            writer.writerow(row)
    print(f"Wrote aligned CSV to {csv_path}")

if __name__ == "__main__":
    process_rosbag()