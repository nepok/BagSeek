import os
import csv
import numpy as np
import json
from rosbags.rosbag2 import Reader  # type: ignore
from collections import defaultdict
from tqdm import tqdm

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

def align_topic_to_reference(topic_ts, ref_ts, max_diff=int(1e8)):
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

# Define constants for paths
BASE_DIR = "/mnt/data/bagseek/flask-backend/src"
ROSBAGS_DIR = "/mnt/data/rosbags"
LOOKUP_TABLES_DIR = os.path.join(BASE_DIR, "lookup_tables")
TOPICS_DIR = os.path.join(BASE_DIR, "topics")

def process_rosbag(rosbag_path, csv_path):
    topic_data = defaultdict(list)
    topic_types = {}
    try:
        with Reader(rosbag_path) as reader:
            for connection, timestamp, rawdata in tqdm(reader.messages(), desc="Reading rosbag"):
                topic_data[connection.topic].append(timestamp)
                topic_types[connection.topic] = connection.msgtype
    except Exception as e:
        print(f"Error reading .db3 bag: {e}")
        return
    
    if not topic_data:
        print(f"No topics found in {rosbag_path}")
        return
    
    # Save topics to a JSON file
    topics = sorted(topic_data.keys())
    os.makedirs(TOPICS_DIR, exist_ok=True)
    topics_json_path = os.path.join(TOPICS_DIR, f"{os.path.basename(rosbag_path)}.json")
    with open(topics_json_path, 'w') as jsonfile:
        json.dump({"topics": topics, "types": topic_types}, jsonfile, indent=2)
    print(f"Topics JSON file generated: {topics_json_path}")

    # Determine reference topic by message count
    ref_topic = determine_reference_topic(topic_data)
    ref_ts = create_reference_timestamps(topic_data[ref_topic])

    # Align each topic to the reference timeline
    aligned_data = {}
    for topic, timestamps in tqdm(topic_data.items(), desc="Aligning topics"):
        aligned_data[topic] = align_topic_to_reference(sorted(timestamps), ref_ts)

    # Prepare CSV data
    csv_data = []
    header = ['Reference Timestamp'] + list(aligned_data.keys()) + ['Max Distance']
    for i, ref_time in enumerate(ref_ts):
        row = [int(ref_time)]
        row_values = []

        for topic in aligned_data.keys():
            ts = aligned_data[topic][i]
            row.append(str(ts) if ts is not None else "None")
            if ts is not None:
                row_values.append(ts)

        max_distance = max(row_values) - min(row_values) if len(row_values) > 1 else None
        row.append(str(max_distance) if max_distance is not None else "None")
        csv_data.append(row)

    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(csv_data)

    print(f"CSV file generated: {csv_path}")

    # Generate image reference map
    image_reference_map = {}
    for topic in aligned_data.keys():
        image_reference_map[topic] = {}
        for i, ref_time in enumerate(ref_ts):
            msg_time = aligned_data[topic][i]
            if msg_time is not None:
                image_reference_map[topic][str(int(ref_time))] = str(msg_time)

    image_map_dir = os.path.join(BASE_DIR, "image_reference_maps")
    os.makedirs(image_map_dir, exist_ok=True)
    image_map_path = os.path.join(image_map_dir, f"{os.path.basename(rosbag_path)}.json")
    with open(image_map_path, "w") as f:
        json.dump(image_reference_map, f, indent=2)
    print(f"Image reference map saved: {image_map_path}")

def create_topics_json_from_csv(csv_path):
    try:
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            # Skip the first column (Reference Timestamp) and the last column (Max Distance)
            topics = header[1:-1]
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    if not topics:
        print(f"No topics found in {csv_path}")
        return

    topics = sorted(topics)
    os.makedirs(TOPICS_DIR, exist_ok=True)
    rosbag_name = os.path.splitext(os.path.basename(csv_path))[0]
    topics_json_path = os.path.join(TOPICS_DIR, f"{rosbag_name}.json")
    if not os.path.exists(topics_json_path):
        with open(topics_json_path, 'w') as jsonfile:
            json.dump({"topics": topics}, jsonfile, indent=2)
        print(f"Topics JSON file generated: {topics_json_path}")
    else:
        print(f"Skipping already existing topics JSON: {topics_json_path}")


def main():

    for root, dirs, files in os.walk(ROSBAGS_DIR):
        if "metadata.yaml" in files:
            rosbag_path = os.path.dirname(os.path.join(root, "metadata.yaml"))
            
            if "EXCLUDED" in rosbag_path:
                continue

            rosbag_name = os.path.basename(rosbag_path)

            csv_path = os.path.join(LOOKUP_TABLES_DIR, f"{rosbag_name}.csv")
            topics_json_path = os.path.join(TOPICS_DIR, f"{rosbag_name}.json")
            
            if not os.path.exists(csv_path):
                print(f"Processing rosbag to CSV: {rosbag_name}")
                process_rosbag(rosbag_path, csv_path)
            else:
                print(f"Skipping already processed rosbag CSV: {rosbag_name}")

            if not os.path.exists(topics_json_path):
                print(f"Generating topics JSON for: {rosbag_name}")
                create_topics_json_from_csv(csv_path)
            else:
                print(f"Skipping already existing topics JSON: {rosbag_name}")

if __name__ == "__main__":
    main()