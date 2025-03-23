import os
import csv
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import cv2
from PIL import Image  # Import Pillow
from io import BytesIO  # For handling byte streams
import numpy as np
from tqdm import tqdm

# Define paths
BASE_PATH = '/home/ubuntu/Documents/Bachelor/bagseek/flask-backend/src'
ROSBAG_DIR = os.path.join(BASE_PATH, 'rosbags')
CSV_DIR = os.path.join(BASE_PATH, 'lookup_tables')
OUTPUT_BASE_DIR = os.path.join(BASE_PATH, 'extracted_images')

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
            break
        elif f.endswith('.yaml'):
            continue
        elif f.endswith('.mcap'):
            return 'mcap'
            break
    
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

def extract_images_from_rosbag(rosbag_path: str, output_dir: str, csv_path: str, format: str):
    """Extract images from a single rosbag and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    aligned_data = pd.read_csv(csv_path, dtype=str)

    if format == 'db3':
    
        with Reader(rosbag_path) as reader:
            connections = [x for x in reader.connections if 'image' in x.topic.lower() and x.topic != '/camera_image/Cam_MR']
            #connections = [x for x in reader.connections if 'image' in x.topic.lower()]                
            for connection, timestamp, rawdata in tqdm(reader.messages(connections=connections), desc=f"Extracting {os.path.basename(rosbag_path)}"):
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                
                if connection.topic not in aligned_data.columns:
                    print(f'error: Topic {connection.topic} not found in aligned_data')
                

                row = aligned_data[aligned_data[connection.topic] == str(timestamp)]

                try:
                    reference_timestamp = row['Reference Timestamp'].iloc[0]
                except:
                    print(f"Error: No reference timestamp found for {connection.topic} at {timestamp}")
                    continue

                # Save the image
                img_filename = f"{connection.topic.replace('/', '__')}-{reference_timestamp}.webp"
                img_filepath = os.path.join(output_dir, img_filename)


                if os.path.exists(img_filepath):
                    print(f"Skipping already extracted image: {img_filepath}")
                    continue

                if hasattr(msg, 'encoding'):
                    try:
                        # Extract the image data from the message.
                        img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                        
                        if rosbag_path == '/home/ubuntu/Documents/Bachelor/bagseek/flask-backend/src/rosbags/rosbag2_2024_08_01-16_00_23':
                            img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                        else:
                            img = img_data
                        


                        cv2.imwrite(img_filepath, img)

                        #print(f"Extracted image: {img_filepath}")
                    except Exception as e:
                        print(f"Failed to extract image from {connection.topic} at {timestamp}: {e}")
    elif format == 'mcap':

        message_reader = read_mcap_messages(rosbag_path)
        for message in tqdm(message_reader, desc=f"Extracting {os.path.basename(rosbag_path)}"):
            msg = None  # Ensure msg always exists
            
            if format == 'db3':
                try:
                    topic, timestamp, rawdata, msgtype = message
                    msg = typestore.deserialize_cdr(rawdata, msgtype)
                except Exception as e:
                    print(f"Deserialization failed for {topic} at {timestamp}: {e}")
                    continue
            else:  # mcap
                topic, msg, timestamp = message

            if not ('image' in topic.lower() or 'camera' in topic.lower()):
                continue
            
            if msg is None:
                print(f"Skipping {topic} at {timestamp}: No valid message data.")
                continue

            if topic not in aligned_data.columns:
                print(f'Error: Topic {topic} not found in aligned_data')
                continue
            
            row = aligned_data[aligned_data[topic] == str(timestamp)]
            
            try:
                reference_timestamp = row['Reference Timestamp'].iloc[0]
            except:
                print(f"Error: No reference timestamp found for {topic} at {timestamp}")
                continue
            
            img_filename = f"{topic.replace('/', '__')}-{reference_timestamp}.webp"
            img_filepath = os.path.join(output_dir, img_filename)
            
            if os.path.exists(img_filepath):
                print(f"Skipping already extracted image: {img_filepath}")
                continue
            
            try:
                # Decode the JPEG image using Pillow
                img_data = np.frombuffer(msg.data, dtype=np.uint8)
                img = Image.open(BytesIO(img_data))  # Decode JPEG using Pillow
                img = img.convert('RGB')  # Ensure the image is in RGB format
                img.save(img_filepath, 'WEBP')  # Save as WebP format
            except Exception as e:
                print(f"Failed to extract image from {topic} at {timestamp}: {e}")

def main():
    """Main function to iterate over all rosbags and extract images."""
    for rosbag_file in tqdm(os.listdir(ROSBAG_DIR), desc="Processing rosbags"):
        rosbag_path = os.path.join(ROSBAG_DIR, rosbag_file)
        output_dir = os.path.join(OUTPUT_BASE_DIR, rosbag_file) 
        csv_dir = os.path.join(CSV_DIR, rosbag_file + '.csv')

        format = detect_bag_format(rosbag_path)
        if not format:
            print(f"Skipping unknown format: {rosbag_file}")
            continue

        # Skip processing if the output directory already exists
        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"Skipping already processed rosbag: {rosbag_file}")
            continue
        
        print(f"Processing rosbag: {rosbag_file}")
        extract_images_from_rosbag(rosbag_path, output_dir, csv_dir, format)

if __name__ == "__main__":
    main()