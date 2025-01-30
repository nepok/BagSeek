import os
import csv
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
BASE_PATH = '/home/ubuntu/Documents/Bachelor/bagseek/flask-backend/src'
ROSBAG_DIR = os.path.join(BASE_PATH, 'rosbags')
CSV_DIR = os.path.join(BASE_PATH, 'lookup_tables')
OUTPUT_BASE_DIR = os.path.join(BASE_PATH, 'extracted_images')

# Create a typestore and get the Image message class.
typestore = get_typestore(Stores.LATEST)

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

def extract_images_from_rosbag(rosbag_path: str, output_dir: str, csv_path: str):
    """Extract images from a single rosbag and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    aligned_data = pd.read_csv(csv_path, dtype=str)
    
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

def main():
    """Main function to iterate over all rosbags and extract images."""
    for rosbag_file in tqdm(os.listdir(ROSBAG_DIR), desc="Processing rosbags"):
        rosbag_path = os.path.join(ROSBAG_DIR, rosbag_file)
        output_dir = os.path.join(OUTPUT_BASE_DIR, rosbag_file + '_referenced') 
        csv_dir = os.path.join(CSV_DIR, rosbag_file + '.csv')

        # Skip processing if the output directory already exists
        #if os.path.exists(output_dir) and os.listdir(output_dir):
        #    print(f"Skipping already processed rosbag: {rosbag_file}")
        #    continue
        
        print(f"Processing rosbag: {rosbag_file}")
        extract_images_from_rosbag(rosbag_path, output_dir, csv_dir)

if __name__ == "__main__":
    main()