import json
import os
import csv
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions # type: ignore
from rclpy.serialization import deserialize_message # type: ignore
from rosidl_runtime_py.utilities import get_message # type: ignore
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures
from sensor_msgs.msg import CompressedImage, Image # type: ignore

# Define paths
BASE_DIR = "/mnt/data/bagseek/flask-backend/src"
ROSBAGS_DIR = "/mnt/data/rosbags"
CSV_DIR = os.path.join(BASE_DIR, 'lookup_tables')
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'extracted_images')

# Create a typestore and get the Image message class.
typestore = get_typestore(Stores.LATEST)

def detect_bag_format(bag_path):
    """Detects whether a bag file is in .db3 or .mcap format."""
    metadata_file = os.path.join(bag_path, 'metadata.yaml')
    
    if not os.path.isfile(metadata_file):
        return None
    
    for f in os.listdir(bag_path):
        if f.endswith('.db3'):
            return 'db3'
        elif f.endswith('.yaml'):
            continue
        elif f.endswith('.mcap'):
            return 'mcap'
    
    return None

def load_lookup_table(csv_path):
    """Load the lookup table from a CSV file into a dictionary."""
    lookup = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ref_timestamp = row['Reference Timestamp']
            topic_timestamps = {topic: row[topic] for topic in reader.fieldnames[1:] if row[topic] != 'None'}
            lookup[ref_timestamp] = topic_timestamps
    return lookup

def read_mcap_messages(bag_path):
    """Reads messages from an .mcap rosbag."""
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id="mcap"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp

def save_image(img, img_filepath):
    """Saves the image to a file."""
    cv2.imwrite(img_filepath, img)

def extract_image_from_message(msg, topic, timestamp, aligned_data, output_dir):
    """Extracts and saves an image from a single message."""
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
    img_filepath = os.path.join(output_dir, img_filename)

    if os.path.exists(img_filepath):
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

def extract_images_from_rosbag(rosbag_path: str, output_dir: str, csv_path: str, format: str):
    """Extract images from a single rosbag and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    aligned_data = pd.read_csv(csv_path, dtype=str)

    # Detect rosbag format
    format = detect_bag_format(rosbag_path)

    # Load image topics from the topic JSON file
    rosbag_name = os.path.basename(rosbag_path)
    topic_json_path = os.path.join(BASE_DIR, "topics", rosbag_name + ".json")
    with open(topic_json_path, "r") as f:
        topic_metadata = json.load(f)

    image_topics = [
        topic for topic, msg_type in topic_metadata["types"].items()
        if msg_type in ("sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image")
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        if format == 'db3':
            with Reader(rosbag_path) as reader:
                connections = [x for x in reader.connections if x.topic in image_topics]
                for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections), desc=f"Extracting {os.path.basename(rosbag_path)}"):
                    try:
                        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                        # Robust timestamp lookup and topic existence
                        # Ensure the topic exists in the aligned data
                        if connection.topic not in aligned_data.columns:
                            print(f"Warning: Topic {connection.topic} not found in aligned data — skipping message at {timestamp}")
                            continue

                        # Check if the current message timestamp is referenced
                        matching_rows = aligned_data[connection.topic] == str(timestamp)
                        if not matching_rows.any():
                            print(f"\nWarning: No reference timestamp found for {connection.topic} at {timestamp} — skipping.")
                            continue

                        future = executor.submit(extract_image_from_message, msg, connection.topic, timestamp, aligned_data, output_dir)
                        futures.append(future)
                    except Exception as e:
                        print(f"Error processing message from topic {connection.topic} at {timestamp}: {e}")

        elif format == 'mcap':
            message_reader = read_mcap_messages(rosbag_path)
            for message in tqdm(message_reader, desc=f"Extracting {os.path.basename(rosbag_path)}"):
                try:
                    topic, msg, timestamp = message
                    if not ('image' in topic.lower() or 'camera' in topic.lower()):
                        continue
                    if msg is None:
                        print(f"Skipping {topic} at {timestamp}: No valid message data.")
                        continue
                    # Robust timestamp lookup and topic existence
                    if topic not in aligned_data.columns:
                        print(f"Warning: Topic {topic} not found in aligned data — skipping message at {timestamp}")
                        continue

                    matching_rows = aligned_data[topic] == str(timestamp)
                    if not matching_rows.any():
                        print(f"Warning: No reference timestamp found for {topic} at {timestamp} — skipping.")
                        continue

                    future = executor.submit(extract_image_from_message, msg, topic, timestamp, aligned_data, output_dir)
                    futures.append(future)
                except Exception as e:
                    print(f"Error processing message from topic {topic} at {timestamp}: {e}")

        # Wait for all threads to finish
        for future in futures:
            future.result()

def main():
    """Main function to iterate over all rosbags and extract images."""

    for root, dirs, files in os.walk(ROSBAGS_DIR):
        if "metadata.yaml" in files:
            rosbag_path = os.path.dirname(os.path.join(root, "metadata.yaml"))

            if "EXCLUDED" in rosbag_path:
                continue
            rosbag_name = os.path.basename(rosbag_path).replace(".db3", "")

            output_dir = os.path.join(OUTPUT_BASE_DIR, rosbag_name)
            csv_path = os.path.join(CSV_DIR, rosbag_name + '.csv')

            format = detect_bag_format(rosbag_path)
            if not format:
                print(f"Skipping unknown format: {rosbag_name}")
                continue

            if os.path.exists(output_dir) and os.listdir(output_dir):
                print(f"Skipping already processed rosbag: {rosbag_name}")
                continue

            print(f"Processing rosbag: {rosbag_name}")
            #extract_images_from_rosbag(rosbag_path, output_dir, csv_path, format)

if __name__ == "__main__":
    main()