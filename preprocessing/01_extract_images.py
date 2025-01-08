import os
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import cv2
import numpy as np

# Define paths
BASE_PATH = '/home/ubuntu/Documents/Bachelor/bagseek/flask-backend/src'
ROSBAG_DIR = os.path.join(BASE_PATH, 'rosbags')
OUTPUT_BASE_DIR = os.path.join(BASE_PATH, 'extracted_images')

# Create a typestore and get the Image message class.
typestore = get_typestore(Stores.LATEST)

def extract_images_from_rosbag(rosbag_path: str, output_dir: str):
    """Extract images from a single rosbag and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    with Reader(rosbag_path) as reader:
        connections = [x for x in reader.connections if 'image' in x.topic.lower() and x.topic != '/camera_image/Cam_MR']        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            
            if hasattr(msg, 'encoding'):
                try:
                    # Extract the image data from the message.
                    img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                    img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    
                    # Save the image
                    img_filename = f"{connection.topic.replace('/', '__')}-{timestamp}.png"
                    img_filepath = os.path.join(output_dir, img_filename)
                    cv2.imwrite(img_filepath, img)

                    print(f"Extracted image: {img_filepath}")
                except Exception as e:
                    print(f"Failed to extract image from {connection.topic} at {timestamp}: {e}")

def main():
    """Main function to iterate over all rosbags and extract images."""
    for rosbag_file in os.listdir(ROSBAG_DIR):
        rosbag_path = os.path.join(ROSBAG_DIR, rosbag_file)
        output_dir = os.path.join(OUTPUT_BASE_DIR, os.path.splitext(rosbag_file)[0])
        
        # Skip processing if the output directory already exists
        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"Skipping already processed rosbag: {rosbag_file}")
            continue
        
        print(f"Processing rosbag: {rosbag_file}")
        extract_images_from_rosbag(rosbag_path, output_dir)

if __name__ == "__main__":
    main()