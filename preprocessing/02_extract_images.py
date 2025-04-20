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
    row = aligned_data[aligned_data[topic] == str(timestamp)]

    try:
        reference_timestamp = row['Reference Timestamp'].iloc[0]
    except:
        print(f"Error: No reference timestamp found for {topic} at {timestamp}")
        return

    img_filename = f"{topic.replace('/', '__')}-{reference_timestamp}.webp"
    img_filepath = os.path.join(output_dir, img_filename)

    if os.path.exists(img_filepath):
        print(f"Skipping already extracted image: {img_filepath}")
        return

    try:
        # Check for ROS 2 message types
        msg_type = type(msg).__name__
        
        if msg_type == 'sensor_msgs__msg__CompressedImage':
            # For compressed image (e.g., JPEG)
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode CompressedImage")
        elif msg_type == 'sensor_msgs__msg__Image':
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
            raise TypeError(f"Unsupported message type: {msg_type}")

        save_image(img, img_filepath)

    except Exception as e:
        print(f"Failed to extract image from {topic} at {timestamp}: {e}")

def extract_images_from_rosbag(rosbag_path: str, output_dir: str, csv_path: str, format: str):
    """Extract images from a single rosbag and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    aligned_data = pd.read_csv(csv_path, dtype=str)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        if format == 'db3':
            with Reader(rosbag_path) as reader:
                connections = [x for x in reader.connections if 'image' in x.topic.lower() and x.topic != '/camera_image/Cam_MR']
                for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections), desc=f"Extracting {os.path.basename(rosbag_path)}"):
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    if connection.topic not in aligned_data.columns:
                        print(f'Error: Topic {connection.topic} not found in aligned_data')
                        continue

                    future = executor.submit(extract_image_from_message, msg, connection.topic, timestamp, aligned_data, output_dir)
                    futures.append(future)

        elif format == 'mcap':
            message_reader = read_mcap_messages(rosbag_path)
            for message in tqdm(message_reader, desc=f"Extracting {os.path.basename(rosbag_path)}"):
                topic, msg, timestamp = message
                if not ('image' in topic.lower() or 'camera' in topic.lower()):
                    continue
                if msg is None:
                    print(f"Skipping {topic} at {timestamp}: No valid message data.")
                    continue
                if topic not in aligned_data.columns:
                    print(f'Error: Topic {topic} not found in aligned_data')
                    continue
                
                future = executor.submit(extract_image_from_message, msg, topic, timestamp, aligned_data, output_dir)
                futures.append(future)

        # Wait for all threads to finish
        for future in futures:
            future.result()

def main():
    """Main function to iterate over all rosbags and extract images."""
    for rosbag_file in tqdm(os.listdir(ROSBAGS_DIR), desc="Processing rosbags"):
        print(rosbag_file)
        
        if not (rosbag_file == '2011_09_29_drive_0071_sync_bag'):
            continue
        rosbag_path = os.path.join(ROSBAGS_DIR, rosbag_file)
        output_dir = os.path.join(OUTPUT_BASE_DIR, rosbag_file) 
        csv_dir = os.path.join(CSV_DIR, rosbag_file + '.csv')

        format = detect_bag_format(rosbag_path)
        if not format:
            print(f"Skipping unknown format: {rosbag_file}")
            continue

        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"Skipping already processed rosbag: {rosbag_file}")
            continue
        
        print(f"Processing rosbag: {rosbag_file}")
        extract_images_from_rosbag(rosbag_path, output_dir, csv_dir, format)

if __name__ == "__main__":
    main()