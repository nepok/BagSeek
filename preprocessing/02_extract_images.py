import json
import os
import csv
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions  # type: ignore
from rclpy.serialization import deserialize_message  # type: ignore
from rosidl_runtime_py.utilities import get_message  # type: ignore
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures
from sensor_msgs.msg import CompressedImage, Image  # type: ignore
from pathlib import Path
from dotenv import load_dotenv

PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Define paths
ROSBAGS_DIR_NAS = os.getenv("ROSBAGS_DIR_NAS")
LOOKUP_TABLES_DIR = os.getenv("LOOKUP_TABLES_DIR")
IMAGES_PER_TOPIC_DIR = os.getenv("IMAGES_PER_TOPIC_DIR")
TOPICS_DIR = os.getenv("TOPICS_DIR")

# Create a typestore and get the Image message class
typestore = get_typestore(Stores.LATEST)

# Detects whether a bag file is in .db3 or .mcap format
def detect_bag_format(bag_path: Path):
    metadata_file = bag_path / "metadata.yaml"

    if not metadata_file.is_file():
        return None

    for entry in bag_path.iterdir():
        if entry.suffix == ".db3":
            return "db3"
        if entry.suffix == ".mcap":
            return "mcap"

    return None

# Load the lookup table from a CSV file into a dictionary
def load_lookup_table(csv_path: Path):
    lookup = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ref_timestamp = row['Reference Timestamp']
            topic_timestamps = {topic: row[topic] for topic in reader.fieldnames[1:] if row[topic] != 'None'}
            lookup[ref_timestamp] = topic_timestamps
    return lookup

# Reads messages from an .mcap rosbag
def read_mcap_messages(bag_path: Path, allowed_topics: set[str]):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}

    message_types = {}
    for topic in allowed_topics:
        type_name = topic_types.get(topic)
        if not type_name:
            continue
        try:
            message_types[topic] = get_message(type_name)
        except (AttributeError, ValueError) as exc:
            print(f"Warning: unable to load ROS type '{type_name}' for topic {topic}: {exc}")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_cls = message_types.get(topic)
        if not msg_cls:
            continue

        msg = deserialize_message(data, msg_cls)
        yield topic, msg, timestamp

    del reader

# Saves the image to a file
def save_image(img, img_filepath: Path):
    cv2.imwrite(str(img_filepath), img)

# Extracts and saves an image from a single message
def extract_image_from_message(msg, topic, timestamp, aligned_data, output_dir: Path):
    # Find the row(s) in aligned_data where the topic timestamp matches the current message timestamp
    matching_rows = aligned_data[topic] == str(timestamp)
    try:
        # Use the index of the first matching row to get the correct reference timestamp
        ref_row_index = matching_rows.idxmax()
        reference_timestamp = aligned_data.iloc[ref_row_index]["Reference Timestamp"]
    except Exception as e:
        print(f"Error: No reference timestamp found for {topic} at {timestamp}")
        return

    # Save image using the actual message timestamp (not the reference timestamp)
    img_filename = f"{topic.replace('/', '__')}-{timestamp}.webp"
    img_filepath = output_dir / img_filename

    if img_filepath.exists():
        print(f"Skipping already extracted image: {img_filepath}")
        return

    try:
        if isinstance(msg, CompressedImage) or msg.__class__.__name__ == "sensor_msgs__msg__CompressedImage":
            # For compressed image (e.g., JPEG)
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode CompressedImage")
        elif isinstance(msg, Image) or msg.__class__.__name__ == "sensor_msgs__msg__Image":
            # For raw image in ROS 2
            channels = {
                "mono8": 1,
                "rgb8": 3,
                "bgr8": 3,
                "rgba8": 4,
                "bgra8": 4
            }.get(msg.encoding)

            if channels is None:
                raise NotImplementedError(f"Unsupported encoding: {msg.encoding}")

            img_data = np.frombuffer(msg.data, dtype=np.uint8)
            img_data = img_data.reshape((msg.height, msg.width, channels))

            if msg.encoding == "rgb8":
                img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "rgba8":
                img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else:
                img = img_data
        else:
            raise TypeError(f"Unsupported message type: {type(msg)}")

        save_image(img, img_filepath)

    except Exception as e:
        print(f"Failed to extract image from {topic} at {timestamp}: {e}")

# Extract images from a single rosbag and save them to the output directory
def extract_images_from_rosbag(rosbag_path: Path, output_dir: Path, csv_path: Path, format: str):
    aligned_data = pd.read_csv(csv_path, dtype=str)
    output_dir.mkdir(parents=True, exist_ok=True)

    topics_root = Path(TOPICS_DIR)
    topic_json_path = topics_root / f"{rosbag_path.name}.json"
    if not topic_json_path.exists():
        raise FileNotFoundError(f"Missing topic metadata JSON: {topic_json_path}")

    with open(topic_json_path, "r") as f:
        topic_metadata = json.load(f)

    image_topics = [
        topic
        for topic, msg_type in topic_metadata.get("types", {}).items()
        if msg_type in ("sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image")
    ]

    if not image_topics:
        print(f"No image topics found for {rosbag_path.name}, skipping")
        return

    image_topic_set = set(image_topics)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        if format == 'db3':
            with Reader(str(rosbag_path)) as reader:
                connections = [x for x in reader.connections if x.topic in image_topic_set]
                for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections), desc=f"Extracting {rosbag_path.name}"):
                    try:
                        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                        if connection.topic not in aligned_data.columns:
                            print(f"\nWarning: Topic {connection.topic} not found in aligned data — skipping message at {timestamp}")
                            continue

                        matching_rows = aligned_data[connection.topic] == str(timestamp)
                        if not matching_rows.any():
                            print(f"\nWarning: No reference timestamp found for {connection.topic} at {timestamp} — skipping")
                            continue

                        topic_dir = output_dir / connection.topic.replace('/', '__')
                        topic_dir.mkdir(parents=True, exist_ok=True)
                        future = executor.submit(
                            extract_image_from_message,
                            msg,
                            connection.topic,
                            timestamp,
                            aligned_data,
                            topic_dir,
                        )
                        futures.append(future)
                    except Exception as e:
                        print(f"\nError processing message from topic {connection.topic} at {timestamp}: {e}")

        elif format == 'mcap':
            message_reader = read_mcap_messages(rosbag_path, image_topic_set)
            for topic, msg, timestamp in tqdm(message_reader, desc=f"Extracting {rosbag_path.name}"):
                try:
                    if topic not in aligned_data.columns:
                        print(f"\nWarning: Topic {topic} not found in aligned data — skipping message at {timestamp}")
                        continue

                    matching_rows = aligned_data[topic] == str(timestamp)
                    if not matching_rows.any():
                        continue

                    topic_dir = output_dir / topic.replace('/', '__')
                    topic_dir.mkdir(parents=True, exist_ok=True)
                    future = executor.submit(
                        extract_image_from_message,
                        msg,
                        topic,
                        timestamp,
                        aligned_data,
                        topic_dir,
                    )
                    futures.append(future)
                except Exception as e:
                    print(f"\nError processing message from topic {topic} at {timestamp}: {e}")

        # Wait for all threads to finish
        for future in futures:
            future.result()

def iter_rosbag_dirs(base_dir: Path):
    for dirpath, _dirnames, filenames in os.walk(base_dir):
        if "metadata.yaml" not in filenames:
            continue

        bag_dir = Path(dirpath)
        if "EXCLUDED" in str(bag_dir):
            continue

        yield bag_dir


# Main function to iterate over all rosbags and extract images
def main():
    required_env = {
        "ROSBAGS_DIR_NAS": ROSBAGS_DIR_NAS,
        "LOOKUP_TABLES_DIR": LOOKUP_TABLES_DIR,
        "IMAGES_PER_TOPIC_DIR": IMAGES_PER_TOPIC_DIR,
        "TOPICS_DIR": TOPICS_DIR,
    }

    missing = [key for key, value in required_env.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    rosbag_root = Path(ROSBAGS_DIR_NAS)
    if not rosbag_root.exists():
        raise FileNotFoundError(f"ROSBAGS_DIR_NAS path does not exist: {rosbag_root}")

    lookup_root = Path(LOOKUP_TABLES_DIR)
    images_root = Path(IMAGES_PER_TOPIC_DIR)
    lookup_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    for bag_dir in iter_rosbag_dirs(rosbag_root):
        rosbag_name = bag_dir.name
        output_dir = images_root / rosbag_name
        csv_path = lookup_root / f"{rosbag_name}.csv"

        bag_format = detect_bag_format(bag_dir)
        if not bag_format:
            print(f"Skipping unknown format: {rosbag_name}")
            continue

        if not csv_path.exists():
            print(f"Skipping {rosbag_name}: missing lookup table CSV at {csv_path}")
            continue

        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"Skipping already processed rosbag: {rosbag_name}")
            continue

        print(f"Processing rosbag: {rosbag_name}")
        extract_images_from_rosbag(bag_dir, output_dir, csv_path, bag_format)

if __name__ == "__main__":
    main()
