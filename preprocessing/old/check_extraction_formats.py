#!/usr/bin/env python3
"""
Utility script to check and display extraction formats from status.json files.
Can also update existing status.json files to include the new extraction_formats field.
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables
PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)


def resolve_env_path(var_name: str, must_exist: bool) -> Path:
    """Resolve an environment variable to a path."""
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"Environment variable '{var_name}' is not set")
    
    path = Path(value).expanduser()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path specified by '{var_name}' does not exist: {path}")
    return path


def topic_to_directory_name(topic: str) -> str:
    """Convert topic name to directory name."""
    return topic.replace("/", "__")


def detect_extraction_formats(output_dir: Path, topic_dir_map: dict[str, str]) -> dict[str, list[str]]:
    """Detect which image formats exist for each topic directory.
    
    Returns:
        dict mapping topic name to list of formats found (e.g., ["webp"], ["original"], or ["webp", "original"])
    """
    format_map: dict[str, list[str]] = {}
    
    for topic, dir_name in topic_dir_map.items():
        topic_dir = output_dir / dir_name
        if not topic_dir.exists():
            format_map[topic] = []
            continue
        
        formats_found = set()
        
        # Check for webp files (from 02A script)
        if any(topic_dir.glob("*.webp")):
            formats_found.add("webp")
        
        # Check for original format files (from 02C script - jpg, png, etc.)
        # Check both direct topic directory and mcap subdirectories
        original_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for ext in original_exts:
            # Check direct topic directory
            if any(topic_dir.glob(f"*{ext}")):
                formats_found.add("original")
                break
            # Check mcap subdirectories
            for mcap_subdir in topic_dir.iterdir():
                if mcap_subdir.is_dir() and any(mcap_subdir.glob(f"*{ext}")):
                    formats_found.add("original")
                    break
            if "original" in formats_found:
                break
        
        format_map[topic] = sorted(formats_found)
    
    return format_map


def load_status_json(status_path: Path) -> dict[str, Any] | None:
    """Load status.json file."""
    if not status_path.exists():
        return None
    
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {status_path}: {e}")
        return None


def save_status_json(status_path: Path, data: dict[str, Any]) -> None:
    """Save status.json file."""
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def check_and_display_formats(images_dir: Path, update_files: bool = False) -> None:
    """Check extraction formats for all rosbags and optionally update status.json files."""
    
    if not images_dir.exists():
        print(f"Images directory does not exist: {images_dir}")
        return
    
    print("=" * 80)
    print("EXTRACTION FORMAT REPORT")
    print("=" * 80)
    print()
    
    rosbag_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    
    total_bags = 0
    bags_with_webp = 0
    bags_with_original = 0
    bags_with_both = 0
    bags_updated = 0
    
    for rosbag_dir in rosbag_dirs:
        status_path = rosbag_dir / "status.json"
        status_data = load_status_json(status_path)
        
        if not status_data:
            continue
        
        total_bags += 1
        rosbag_name = rosbag_dir.name
        
        # Get image topics from existing status
        existing_counts = status_data.get("existing_counts", {})
        image_topics = list(existing_counts.keys())
        
        if not image_topics:
            continue
        
        # Build topic directory map
        topic_dir_map = {topic: topic_to_directory_name(topic) for topic in image_topics}
        
        # Detect formats
        extraction_formats = detect_extraction_formats(rosbag_dir, topic_dir_map)
        
        # Count bags by format type
        has_webp = any("webp" in formats for formats in extraction_formats.values())
        has_original = any("original" in formats for formats in extraction_formats.values())
        
        if has_webp and has_original:
            bags_with_both += 1
            format_summary = "BOTH"
        elif has_webp:
            bags_with_webp += 1
            format_summary = "WEBP"
        elif has_original:
            bags_with_original += 1
            format_summary = "ORIGINAL"
        else:
            format_summary = "NONE"
        
        # Display information
        print(f"Rosbag: {rosbag_name}")
        print(f"  Overall Formats: {format_summary}")
        print(f"  State: {status_data.get('state', 'unknown')}")
        
        # Show per-topic breakdown if multiple formats exist
        if len(set(str(v) for v in extraction_formats.values())) > 1:
            print(f"  Per-Topic Formats:")
            for topic, formats in extraction_formats.items():
                if formats:
                    print(f"    {topic}: {', '.join(formats)}")
        
        # Update status.json if requested and field is missing
        if update_files and "extraction_formats" not in status_data:
            status_data["extraction_formats"] = extraction_formats
            save_status_json(status_path, status_data)
            print(f"  ✓ Updated status.json with extraction_formats")
            bags_updated += 1
        elif "extraction_formats" in status_data:
            print(f"  ℹ status.json already has extraction_formats field")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total rosbags with status: {total_bags}")
    print(f"  - With WEBP only: {bags_with_webp}")
    print(f"  - With ORIGINAL only: {bags_with_original}")
    print(f"  - With BOTH formats: {bags_with_both}")
    
    if update_files:
        print()
        print(f"Status files updated: {bags_updated}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check and display extraction formats from status.json files"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing status.json files to include extraction_formats field"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Path to images directory (overrides environment variable)"
    )
    
    args = parser.parse_args()
    
    # Determine images directory
    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        # Try both environment variables
        try:
            images_dir = resolve_env_path("IMAGES_PER_TOPIC_DIR", must_exist=True)
        except (EnvironmentError, FileNotFoundError):
            try:
                images_dir = resolve_env_path("ORIGINAL_IMAGES_DIR", must_exist=True)
            except (EnvironmentError, FileNotFoundError):
                print("Error: Could not find images directory.")
                print("Please set IMAGES_PER_TOPIC_DIR or ORIGINAL_IMAGES_DIR environment variable,")
                print("or use --images-dir option.")
                return
    
    check_and_display_formats(images_dir, update_files=args.update)


if __name__ == "__main__":
    main()

