

import os
from PIL import Image

BASE_DIR = '/mnt/data/bagseek/flask-backend/src'
PER_TOPIC_DIR = os.path.join(BASE_DIR, "extracted_images_per_topic")
REPRESENTATIVE_PREVIEWS_DIR = os.path.join(BASE_DIR, "representative_previews")

def create_collage(image_paths, output_path):
    images = [Image.open(p) for p in image_paths if os.path.exists(p)]
    if not images:
        print(f"No valid images for collage at {output_path}")
        return

    heights = [img.height for img in images]
    min_height = min(heights)
    resized_images = [img.resize((int(img.width * min_height / img.height), min_height), Image.LANCZOS) for img in images]

    total_width = sum(img.width for img in resized_images)
    collage = Image.new('RGB', (total_width, min_height))

    x_offset = 0
    for img in resized_images:
        collage.paste(img, (x_offset, 0))
        x_offset += img.width

    collage.save(output_path)

def process_representative_images():
    for rosbag in os.listdir(PER_TOPIC_DIR):
        rosbag_path = os.path.join(PER_TOPIC_DIR, rosbag)
        if not os.path.isdir(rosbag_path):
            continue

        output_dir = os.path.join(REPRESENTATIVE_PREVIEWS_DIR, rosbag)
        os.makedirs(output_dir, exist_ok=True)

        for topic in os.listdir(rosbag_path):
            topic_path = os.path.join(rosbag_path, topic)
            if not os.path.isdir(topic_path):
                continue

            image_files = sorted([f for f in os.listdir(topic_path) if f.endswith(".webp")])
            if len(image_files) < 7:
                print(f"Not enough images in {rosbag}/{topic} for collage")
                continue

            step = len(image_files) // 7
            selected_files = [image_files[i * step] for i in range(7)]
            image_paths = [os.path.join(topic_path, f) for f in selected_files]

            collage_path = os.path.join(output_dir, f"{topic}_collage.webp")
            create_collage(image_paths, collage_path)

if __name__ == "__main__":
    process_representative_images()