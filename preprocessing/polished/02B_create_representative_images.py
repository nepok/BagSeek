import os
import json
from collections import defaultdict
from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
from typing import Iterable, Dict, List, Optional
from datetime import datetime
import concurrent.futures

PARENT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Define paths
ROSBAGS = Path(os.getenv("ROSBAGS"))
LOOKUP_TABLES = Path(os.getenv("LOOKUP_TABLES"))
IMAGES = Path(os.getenv("IMAGES"))
REPRESENTATIVE_PREVIEWS = Path(os.getenv("REPRESENTATIVE_PREVIEWS"))
TOPICS = Path(os.getenv("TOPICS"))

# Configuration
MAX_WORKERS = 16
SKIP_COMPLETED = True  # Set to False to reprocess completed rosbags


# =========================
# Completion Tracking Functions
# =========================

def get_completion_file() -> Path:
    """Get the completion file path for representative image generation."""
    return REPRESENTATIVE_PREVIEWS / "completion.json"


def load_completion() -> Dict[str, Dict]:
    """Load the dictionary of completed rosbag data from the completion file.
    
    Returns:
        Dict mapping rosbag_name to dict with keys: completed_at, errors (optional)
    """
    completion_file = get_completion_file()
    if not completion_file.exists():
        return {}
    
    try:
        with open(completion_file, 'r') as f:
            data = json.load(f)
            completed_list = data.get('completed', [])
            result = {}
            
            for item in completed_list:
                if isinstance(item, str):
                    # Old format - just rosbag name
                    result[item] = {"completed_at": "unknown"}
                elif isinstance(item, dict):
                    # New format - dict with rosbag name and optional fields
                    rosbag_name = item.get("rosbag")
                    if rosbag_name:
                        result[rosbag_name] = {
                            "completed_at": item.get("completed_at", "unknown"),
                            "errors": item.get("errors", [])
                        }
            return result
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return {}


def save_completed_rosbag(rosbag_name: str, errors: Optional[List[str]] = None):
    """Mark a rosbag as completed and save to the completion file with timestamp and optional errors.
    
    Args:
        rosbag_name: Name of the rosbag
        errors: Optional list of error messages
    """
    completed = load_completion()
    timestamp = datetime.now().isoformat()
    
    # Update or create entry for this rosbag
    if rosbag_name not in completed:
        completed[rosbag_name] = {
            "completed_at": timestamp
        }
    else:
        completed[rosbag_name]["completed_at"] = timestamp
    
    if errors:
        completed[rosbag_name]["errors"] = errors
    
    # Save to file
    save_completion(completed)


def save_completion(completed: Dict[str, Dict]):
    """Save completed rosbags dictionary to completion file.
    
    Args:
        completed: Dictionary of completed rosbags to save
    """
    completion_file = get_completion_file()
    completion_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file with proper format
    data = {
        'completed': [
            {
                "rosbag": name,
                "completed_at": info.get("completed_at", "unknown"),
                **({"errors": info["errors"]} if info.get("errors") else {})
            }
            for name, info in sorted(completed.items())
        ],
    }
    
    with open(completion_file, 'w') as f:
        json.dump(data, f, indent=2)


def is_rosbag_completed(rosbag_name: str) -> bool:
    """Check if a rosbag has been completed by verifying all topics have collages.
    
    Returns:
        True if rosbag is marked as completed and all topics have collages
    """
    if not SKIP_COMPLETED:
        return False
    
    completed = load_completion()
    if rosbag_name not in completed:
        return False
    
    # Verify by checking if all topics have collages
    rosbag_dir = IMAGES / rosbag_name
    if not rosbag_dir.exists():
        return False
    
    output_dir = REPRESENTATIVE_PREVIEWS / rosbag_name
    if not output_dir.exists():
        return False
    
    # Check if all topics have collages
    topics = [d for d in rosbag_dir.iterdir() if d.is_dir()]
    for topic_dir in topics:
        topic_name = topic_dir.name.replace("/", "__")
        collage_path = output_dir / f"{topic_name}_collage.webp"
        if not collage_path.exists():
            return False
    
    return True


def create_collage(image_paths, output_path):
    """Create a collage from multiple images and save to output_path."""
    try:
        images = [Image.open(p) for p in image_paths if Path(p).exists()]
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

def process_single_topic(topic_info):
    """Process a single topic to create a representative collage from MCAP folders.
    
    Args:
        topic_info: Tuple of (topic_path, rosbag_name, output_dir)
        
    Returns:
        Dict with keys: rosbag_name, topic_name, status, num_images_selected, num_folders, num_images_total
        status can be: 'skipped', 'created', 'failed', 'no_images'
    """
    topic_path, rosbag_name, output_dir = topic_info
    topic_name = topic_path.name
    
    # Create output filename: topic_name_collage.webp
    # Replace slashes in topic name with underscores for filesystem safety
    topic_safe = topic_name.replace("/", "__")
    collage_path = output_dir / f"{topic_safe}_collage.webp"
    
    # Skip if collage already exists
    if collage_path.exists():
        return {
            'rosbag_name': rosbag_name,
            'topic_name': topic_name,
            'status': 'skipped',
            'num_images_selected': None,
            'num_folders': None,
            'num_images_total': None
        }
    
    try:
        # Find all MCAP folders (numeric subdirectories: 0, 1, 2, ...)
        mcap_folders = []
        for item in topic_path.iterdir():
            if item.is_dir():
                # Try to parse as numeric MCAP folder
                try:
                    mcap_num = int(item.name)
                    # Find first PNG image in this MCAP folder
                    png_files = sorted([f for f in item.iterdir() if f.is_file() and f.suffix == ".png"])
                    if png_files:
                        mcap_folders.append((mcap_num, png_files[0]))
                except ValueError:
                    # Not a numeric folder, skip
                    continue
        
        # Sort by MCAP number
        mcap_folders = sorted(mcap_folders, key=lambda x: x[0])
        
        if not mcap_folders:
            return {
                'rosbag_name': rosbag_name,
                'topic_name': topic_name,
                'status': 'no_images',
                'num_images_selected': None,
                'num_folders': 0,
                'num_images_total': None
            }
        
        num_folders = len(mcap_folders)
        num_images = None  # Will be set in fallback path
        
        if num_folders >= 8:
            # Fast path: use folder-based method when we have enough folders
            # Fence post logic: divide by 8, take indices 1-7 (1-indexed)
            # This gives 7 evenly spaced points
            part_size = num_folders / 8.0
            selected_indices = []
            for i in range(1, 8):  # 1 to 7 (1-indexed)
                position = i * part_size
                index = int(round(position))
                # Clamp to valid range
                index = max(0, min(index, num_folders - 1))
                selected_indices.append(index)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in selected_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            
            # If we have fewer than 7 unique indices, pad with last folder
            while len(unique_indices) < 7 and num_folders > 0:
                if num_folders - 1 not in unique_indices:
                    unique_indices.append(num_folders - 1)
                else:
                    break
            
            selected_images = [mcap_folders[idx][1] for idx in unique_indices[:7]]
        else:
            # Fallback: scan all images when we have fewer than 8 folders
            # Collect all PNG images from all MCAP folders
            all_images = []
            for item in topic_path.iterdir():
                if item.is_dir():
                    # Try to parse as numeric MCAP folder
                    try:
                        mcap_num = int(item.name)
                        # Collect all PNG images in this MCAP folder
                        png_files = sorted([f for f in item.iterdir() if f.is_file() and f.suffix == ".png"])
                        all_images.extend(png_files)
                    except ValueError:
                        # Not a numeric folder, skip
                        continue
            
            # Sort all images by path (which should maintain chronological order)
            all_images = sorted(all_images)
            
            if not all_images:
                return {
                    'rosbag_name': rosbag_name,
                    'topic_name': topic_name,
                    'status': 'no_images',
                    'num_images_selected': None,
                    'num_folders': num_folders,
                    'num_images_total': 0
                }
            
            num_images = len(all_images)
            
            # Fence post logic: divide by 8, take indices 1-7 (1-indexed)
            # This gives 7 evenly spaced points across all images
            target_count = min(7, num_images)
            if target_count == 0:
                return {
                    'rosbag_name': rosbag_name,
                    'topic_name': topic_name,
                    'status': 'no_images',
                    'num_images_selected': None,
                    'num_folders': num_folders,
                    'num_images_total': num_images
                }
            
            # Calculate spacing for fence posts
            if num_images == 1:
                selected_indices = [0]
            else:
                part_size = num_images / 8.0
                selected_indices = []
                for i in range(1, 8):  # 1 to 7 (1-indexed)
                    position = i * part_size
                    index = int(round(position))
                    # Clamp to valid range
                    index = max(0, min(index, num_images - 1))
                    selected_indices.append(index)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in selected_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            
            # If we have fewer than target_count unique indices, pad with last image
            while len(unique_indices) < target_count and num_images > 0:
                if num_images - 1 not in unique_indices:
                    unique_indices.append(num_images - 1)
                else:
                    break
            
            selected_images = [all_images[idx] for idx in unique_indices[:target_count]]
        
        success = create_collage([str(img) for img in selected_images], str(collage_path))
        
        if success:
            return {
                'rosbag_name': rosbag_name,
                'topic_name': topic_name,
                'status': 'created',
                'num_images_selected': len(selected_images),
                'num_folders': num_folders,
                'num_images_total': num_images if num_images is not None else None
            }
        else:
            return {
                'rosbag_name': rosbag_name,
                'topic_name': topic_name,
                'status': 'failed',
                'num_images_selected': len(selected_images),
                'num_folders': num_folders,
                'num_images_total': num_images if num_images is not None else None
            }
    except Exception as e:
        return {
            'rosbag_name': rosbag_name,
            'topic_name': topic_name,
            'status': 'failed',
            'num_images_selected': None,
            'num_folders': None,
            'num_images_total': None,
            'error': str(e)
        }


def collect_tasks():
    """Collect all topics that need processing.
    
    Returns:
        List of tuples: (topic_path, rosbag_name, output_dir)
    """
    tasks = []
    skipped_rosbags = []
    
    # Iterate through all rosbags in IMAGES
    for rosbag in IMAGES.iterdir():
        if not rosbag.is_dir():
            continue
        
        rosbag_name = rosbag.name
        
        # Check if rosbag is in EXCLUDED folder by checking ROSBAGS directory
        if ROSBAGS is not None:
            rosbag_path_in_rosbags = ROSBAGS / rosbag_name
            # Also check parent directories in case rosbag is in a subdirectory
            if rosbag_path_in_rosbags.exists():
                if "EXCLUDED" in str(rosbag_path_in_rosbags):
                    continue
            else:
                # Search for rosbag in ROSBAGS (might be in subdirectory)
                found_rosbag = None
                for rosbag_path in ROSBAGS.rglob(rosbag_name):
                    if rosbag_path.is_dir() and (rosbag_path / "metadata.yaml").exists():
                        found_rosbag = rosbag_path
                        break
                if found_rosbag and "EXCLUDED" in str(found_rosbag):
                    continue
        
        # Check if already completed
        if SKIP_COMPLETED and is_rosbag_completed(rosbag_name):
            skipped_rosbags.append(rosbag_name)
            continue
        
        output_dir = REPRESENTATIVE_PREVIEWS / rosbag_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate through all topics in this rosbag
        for topic_path in rosbag.iterdir():
            if not topic_path.is_dir():
                continue
            
            tasks.append((topic_path, rosbag_name, output_dir))
    
    if skipped_rosbags:
        print(f"Skipping {len(skipped_rosbags)} already completed rosbag(s): {', '.join(skipped_rosbags[:5])}{'...' if len(skipped_rosbags) > 5 else ''}")
    
    return tasks


def process_representative_images():
    """Process all topics in parallel to create representative collages."""
    print("Starting Process...")
    
    # Collect all tasks
    tasks = collect_tasks()
    
    if not tasks:
        print("No tasks found to process")
        return
    
    print(f"\nFound {len(tasks)} topics to process")
    print(f"Using {MAX_WORKERS} CPU cores for parallel processing\n")
    
    # Process all tasks in parallel
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_topic, task): task for task in tasks}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Extract rosbag_name from task if possible
                rosbag_name = task[1] if len(task) > 1 else "unknown"
                results.append({
                    'rosbag_name': rosbag_name,
                    'topic_name': 'unknown',
                    'status': 'failed',
                    'num_images_selected': None,
                    'num_folders': None,
                    'num_images_total': None,
                    'error': str(e)
                })
    
    # Group results by rosbag
    rosbag_results = defaultdict(list)
    for result in results:
        rosbag_results[result['rosbag_name']].append(result)
    
    # Print summary per rosbag
    total_succeeded = 0
    total_failed = 0
    total_skipped = 0
    total_no_images = 0
    
    for rosbag_name in sorted(rosbag_results.keys()):
        rosbag_topics = rosbag_results[rosbag_name]
        created = [r for r in rosbag_topics if r['status'] == 'created']
        skipped = [r for r in rosbag_topics if r['status'] == 'skipped']
        failed = [r for r in rosbag_topics if r['status'] == 'failed']
        no_images = [r for r in rosbag_topics if r['status'] == 'no_images']
        
        total_succeeded += len(created)
        total_failed += len(failed)
        total_skipped += len(skipped)
        total_no_images += len(no_images)
        
        print(f"\n{rosbag_name}:")
        print(f"  Created: {len(created)}", end="")
        if created:
            # Show image count info for created collages
            image_counts = {}
            for r in created:
                count = r['num_images_selected']
                if count is not None:
                    image_counts[count] = image_counts.get(count, 0) + 1
            
            if image_counts:
                count_strs = []
                for count, num in sorted(image_counts.items()):
                    if num == 1:
                        count_strs.append(f"{count} image")
                    else:
                        count_strs.append(f"{num}×{count} images")
                print(f" ({', '.join(count_strs)})", end="")
        print()
        
        if skipped:
            print(f"  Skipped: {len(skipped)}")
        if failed:
            print(f"  Failed: {len(failed)}")
            # Show errors if any
            errors = [r.get('error') for r in failed if r.get('error')]
            if errors:
                for error in errors[:3]:  # Show first 3 errors
                    print(f"    - {error}")
                if len(errors) > 3:
                    print(f"    ... and {len(errors) - 3} more errors")
        if no_images:
            print(f"  No images: {len(no_images)}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Created: {total_succeeded}")
    print(f"Skipped: {total_skipped}")
    print(f"Failed: {total_failed}")
    print(f"No images: {total_no_images}")
    print(f"Total: {len(tasks)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_representative_images()