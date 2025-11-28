import os
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
from typing import Iterable
import concurrent.futures

PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Define paths
ROSBAGS_DIR_NAS = os.getenv("ROSBAGS_DIR_NAS")
LOOKUP_TABLES_DIR = os.getenv("LOOKUP_TABLES_DIR")
IMAGES_PER_TOPIC_DIR = os.getenv("IMAGES_PER_TOPIC_DIR")
REPRESENTATIVE_PREVIEWS_DIR = os.getenv("REPRESENTATIVE_IMAGES_DIR")
TOPICS_DIR = os.getenv("TOPICS_DIR")

def create_collage(image_paths, output_path):
    """Create a collage from multiple images and save to output_path."""
    try:
        images = [Image.open(p) for p in image_paths if os.path.exists(p)]
        if not images:
            print(f"No valid images for collage at {output_path}")
            return False

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
        return True
    except Exception as e:
        print(f"Error creating collage {output_path}: {e}")
        return False

def process_single_rosbag(rosbag_info):
    """Process a single rosbag to create a representative collage from MCAP folders."""
    rosbag, rosbag_path, output_dir = rosbag_info
    
    try:
        # Find all unique MCAP folders across all topics in this rosbag
        # Use a dict to store the first image found for each MCAP number
        mcap_folders_dict = {}
        
        # Iterate through all topics in the rosbag
        for topic in os.listdir(rosbag_path):
            topic_path = os.path.join(rosbag_path, topic)
            if not os.path.isdir(topic_path):
                continue
            
            # Check all subdirectories (MCAP folders)
            for item in os.listdir(topic_path):
                subdir_path = os.path.join(topic_path, item)
                if os.path.isdir(subdir_path):
                    # Try to parse as numeric MCAP folder (0, 1, 2, ...)
                    try:
                        mcap_num = int(item)
                        # Only add if we haven't seen this MCAP number yet
                        if mcap_num not in mcap_folders_dict:
                            # Find first PNG image in this MCAP folder
                            png_files = [f for f in os.listdir(subdir_path) if f.endswith(".png") and os.path.isfile(os.path.join(subdir_path, f))]
                            if png_files:
                                # Sort and take the first image
                                first_image = sorted(png_files)[0]
                                mcap_folders_dict[mcap_num] = os.path.join(subdir_path, first_image)
                    except ValueError:
                        # Not a numeric folder, skip
                        continue
        
        if not mcap_folders_dict:
            print(f"No MCAP folders with images found in {rosbag}")
            return False, rosbag
        
        # Convert to sorted list of (mcap_num, image_path) tuples
        mcap_folders = sorted(mcap_folders_dict.items())
        
        if len(mcap_folders) < 7:
            print(f"Not enough MCAP folders in {rosbag} for collage (found {len(mcap_folders)})")
            return False, rosbag

        # Select 7 images by splitting into 8 parts and taking the middle of each part
        # For example, with 100 folders: split into 8 parts (12.5 each), take middle at 6.25, 18.75, 31.25, etc.
        num_folders = len(mcap_folders)
        part_size = num_folders / 8.0
        selected_indices = []
        for i in range(7):
            # Middle of part i+1 (parts are 1-indexed)
            middle_position = (i + 1) * part_size - part_size / 2.0
            selected_index = int(round(middle_position))
            # Clamp to valid range
            selected_index = max(0, min(selected_index, num_folders - 1))
            selected_indices.append(selected_index)
        
        selected_images = [mcap_folders[idx][1] for idx in selected_indices]

        collage_path = os.path.join(output_dir, f"{rosbag}_collage.webp")
        success = create_collage(selected_images, collage_path)
        
        if success:
            print(f"Created collage for {rosbag} (from {len(mcap_folders)} MCAP folders)")
        
        return success, rosbag
    except Exception as e:
        print(f"Error processing {rosbag}: {e}")
        return False, rosbag


def collect_tasks():
    """Collect all rosbags that need processing."""
    tasks = []
    
    for rosbag in os.listdir(IMAGES_PER_TOPIC_DIR):
        rosbag_path = os.path.join(IMAGES_PER_TOPIC_DIR, rosbag)
        if not os.path.isdir(rosbag_path):
            continue
        
        output_dir = os.path.join(REPRESENTATIVE_PREVIEWS_DIR, rosbag)
        os.makedirs(output_dir, exist_ok=True)
        
        tasks.append((rosbag, rosbag_path, output_dir))
    
    return tasks


def process_representative_images():
    """Process all rosbags in parallel to create representative collages."""
    print("Starting Process...")
    
    # Collect all tasks
    tasks = collect_tasks()
    
    if not tasks:
        print("No tasks found to process")
        return
    
    print(f"\nFound {len(tasks)} rosbags to process")
    print(f"Using {os.cpu_count()} CPU cores for parallel processing\n")
    
    # Process all tasks in parallel
    succeeded = 0
    failed = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_rosbag, task): task for task in tasks}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                success, task_name = future.result()
                if success:
                    succeeded += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Exception processing task: {e}")
                failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tasks)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_representative_images()