

import os
import shutil

BASE_DIR = '/mnt/data/bagseek/flask-backend/src'
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings')
EMBEDDINGS_PER_TOPIC_DIR = os.path.join(BASE_DIR, 'embeddings_per_topic')

def organize_images():
    if not os.path.exists(EMBEDDINGS_PER_TOPIC_DIR):
        os.makedirs(EMBEDDINGS_PER_TOPIC_DIR)
    for model_folder in os.listdir(EMBEDDINGS_DIR):
        model_path = os.path.join(EMBEDDINGS_DIR, model_folder)
        for rosbag_folder in os.listdir(model_path):
            rosbag_path = os.path.join(model_path, rosbag_folder)
            if not os.path.isdir(rosbag_path):
                continue

            output_dir = os.path.join(EMBEDDINGS_PER_TOPIC_DIR, model_folder, rosbag_folder)
            os.makedirs(output_dir, exist_ok=True)

            for filename in os.listdir(rosbag_path):
                if not filename.endswith("_embedding.pt"):
                    continue

                if '-' not in filename:
                    continue

                topic_part, timestamp = filename.rsplit('-', 1)
                timestamp = timestamp.replace("_embedding.pt", "")

                topic_dir = os.path.join(output_dir, topic_part)
                os.makedirs(topic_dir, exist_ok=True)

                src_path = os.path.join(rosbag_path, filename)
                dst_path = os.path.join(topic_dir, f"{timestamp}_embedding.pt")
                shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    organize_images()