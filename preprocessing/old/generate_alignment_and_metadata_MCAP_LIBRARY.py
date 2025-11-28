import os
import csv
from pathlib import Path
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
from mcap.reader import make_reader


SOFT_MCAP_ERRORS = {"EndOfFile", "RecordLengthLimitExceeded"}

PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Define constants for paths
ROSBAGS_DIR_MNT = os.getenv("ROSBAGS_DIR_MNT")
ROSBAGS_DIR = os.getenv("ROSBAGS_DIR")
LOOKUP_TABLES_DIR = os.getenv("LOOKUP_TABLES_DIR")
TOPICS_DIR = os.getenv("TOPICS_DIR")

# Determine the topic with the most messages to use as the reference timeline
def determine_reference_topic(topic_timestamps):
    return max(topic_timestamps.items(), key=lambda x: len(x[1]))[0]

# Create a refined reference timeline with smaller intervals to improve alignment accuracy
# The factor reduces the mean interval to create a denser timeline for better matching
def create_reference_timestamps(timestamps, factor=2):
    timestamps = sorted(set(timestamps))
    if len(timestamps) < 2:
        return np.array(timestamps, dtype=np.int64)

    diffs = np.diff(timestamps)
    mean_interval = np.mean(diffs)
    refined_interval = mean_interval / factor if mean_interval else 1
    ref_start = timestamps[0]
    ref_end = timestamps[-1]
    return np.arange(ref_start, ref_end, refined_interval).astype(np.int64)

# Align timestamps of a topic to the closest reference timestamps within max_diff
# For each reference timestamp, find the closest topic timestamp to align messages across topics
def align_topic_to_reference(topic_ts, ref_ts, max_diff=int(1e8)):
    if not topic_ts:
        return [None] * len(ref_ts)

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

def resolve_mcap_files(path: Path) -> list[Path]:
    """Return a list of MCAP files contained in the provided path."""
    if path.is_file() and path.suffix == ".mcap":
        return [path]

    if path.is_dir():
        return sorted(child for child in path.glob("*.mcap") if child.is_file())

    return []


def process_rosbag(rosbag_path, csv_path):
    rosbag_path = Path(rosbag_path)
    csv_path = Path(csv_path)

    topic_data = defaultdict(list)
    topic_types = {}

    mcap_files = resolve_mcap_files(rosbag_path)

    if not mcap_files:
        print(f"No .mcap files found for {rosbag_path}")
        return False

    try:
        with tqdm(desc=f"Reading {os.path.basename(rosbag_path)}", unit="msg") as pbar:
            for file_path in mcap_files:
                with file_path.open("rb") as handle:
                    reader = make_reader(handle)
                    for schema, channel, message in reader.iter_messages():
                        topic_data[channel.topic].append(int(message.log_time))
                        if channel.topic not in topic_types:
                            topic_types[channel.topic] = schema.name if schema else "<unknown>"
                        pbar.update()
    except Exception as e:
        error_name = e.__class__.__name__
        if error_name in SOFT_MCAP_ERRORS:
            if error_name == "EndOfFile":
                print(f"Reached end of MCAP data for {rosbag_path}: {e!r}")
            else:
                print(
                    f"Encountered recoverable MCAP error for {rosbag_path}: {e!r} — continuing with collected data"
                )
        else:
            print(f"Error reading MCAP data at {rosbag_path}: {e!r}")
            return False

    if not topic_data:
        print(f"No topics found in {rosbag_path}")
        return False

    # Save topics to a JSON file without indentation
    topics = sorted(topic_data.keys())
    topics_root = Path(TOPICS_DIR)
    topics_root.mkdir(parents=True, exist_ok=True)
    topics_json_path = topics_root / f"{rosbag_path.name}.json"
    with open(topics_json_path, 'w') as jsonfile:
        json.dump({"topics": topics, "types": topic_types}, jsonfile, indent=2)
    print(f"Topics JSON file generated: {topics_json_path}")

    # Determine reference topic by message count
    ref_topic = determine_reference_topic(topic_data)
    ref_ts = create_reference_timestamps(topic_data[ref_topic])

    # Align each topic to the reference timeline to synchronize timestamps across topics
    aligned_data = {}
    for topic, timestamps in tqdm(topic_data.items(), desc="Aligning topics"):
        aligned_data[topic] = align_topic_to_reference(sorted(timestamps), ref_ts)

    # Prepare CSV data
    # The CSV rows contain the reference timestamp, aligned timestamps for each topic, and the max timestamp distance across topics
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
    return True

def create_topics_json_from_csv(csv_path):
    csv_path = Path(csv_path)
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
    topics_root = Path(TOPICS_DIR)
    topics_root.mkdir(parents=True, exist_ok=True)
    rosbag_name = csv_path.stem
    topics_json_path = topics_root / f"{rosbag_name}.json"
    if not topics_json_path.exists():
        with open(topics_json_path, 'w') as jsonfile:
            json.dump({"topics": topics}, jsonfile, indent=2)
        print(f"Topics JSON file generated: {topics_json_path}")
    else:
        print(f"Skipping already existing topics JSON: {topics_json_path}")

def iter_rosbag_dirs(base_dir: Path):
    for dirpath, _dirnames, filenames in os.walk(base_dir):
        bag_dir = Path(dirpath)
        if "EXCLUDED" in str(bag_dir):
            continue

        if not any(name.endswith('.mcap') for name in filenames):
            continue

        yield bag_dir


def walk(rosbag_dir: Path):
    topics_root = Path(TOPICS_DIR)
    lookup_root = Path(LOOKUP_TABLES_DIR)
    lookup_root.mkdir(parents=True, exist_ok=True)
    topics_root.mkdir(parents=True, exist_ok=True)

    for bag_dir in iter_rosbag_dirs(rosbag_dir):
        rosbag_name = bag_dir.name

        csv_path = lookup_root / f"{rosbag_name}.csv"
        topics_json_path = topics_root / f"{rosbag_name}.json"

        processed = False
        if not csv_path.exists():
            print(f"Processing rosbag to CSV: {rosbag_name}")
            processed = process_rosbag(bag_dir, csv_path)
        else:
            print(f"Skipping already processed rosbag CSV: {rosbag_name}")
            processed = True

        if processed and csv_path.exists() and not topics_json_path.exists():
            print(f"Generating topics JSON for: {rosbag_name}")
            create_topics_json_from_csv(csv_path)
        elif topics_json_path.exists():
            print(f"Skipping already existing topics JSON: {rosbag_name}")
        else:
            print(f"Skipping topics JSON generation for {rosbag_name} because CSV is missing")


# Main function walks through rosbags directory and processes each rosbag only if it hasn't been processed before
# This avoids redundant computation by checking for existing CSV and JSON files before processing
def main():
    if not ROSBAGS_DIR:
        raise RuntimeError("ROSBAGS_DIR is not set in the environment")
    if not LOOKUP_TABLES_DIR:
        raise RuntimeError("LOOKUP_TABLES_DIR is not set in the environment")
    if not TOPICS_DIR:
        raise RuntimeError("TOPICS_DIR is not set in the environment")

    rosbag_root = Path(ROSBAGS_DIR)
    if not rosbag_root.exists():
        raise FileNotFoundError(f"ROSBAGS_DIR path does not exist: {rosbag_root}")

    walk(rosbag_root)

if __name__ == "__main__":
    main()
