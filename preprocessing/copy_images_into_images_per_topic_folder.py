

import os
import shutil

BASE_DIR = '/mnt/data/bagseek/flask-backend/src/extracted_images'
NEW_DIR = os.path.join(BASE_DIR, "NEW")

def organize_images():
    if not os.path.exists(NEW_DIR):
        os.makedirs(NEW_DIR)

    for rosbag_folder in os.listdir(BASE_DIR):
        rosbag_path = os.path.join(BASE_DIR, rosbag_folder)
        if not os.path.isdir(rosbag_path) or rosbag_folder == "NEW":
            continue

        rosbag_output_dir = os.path.join(NEW_DIR, rosbag_folder)
        os.makedirs(rosbag_output_dir, exist_ok=True)

        for filename in os.listdir(rosbag_path):
            if not filename.endswith(".webp"):
                continue

            if '-' not in filename:
                continue

            topic_part, timestamp = filename.rsplit('-', 1)
            timestamp = timestamp.replace(".webp", "")

            topic_dir = os.path.join(rosbag_output_dir, topic_part)
            os.makedirs(topic_dir, exist_ok=True)

            src_path = os.path.join(rosbag_path, filename)
            dst_path = os.path.join(topic_dir, f"{timestamp}.webp")
            shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    organize_images()