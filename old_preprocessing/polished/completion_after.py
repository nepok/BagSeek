#!/usr/bin/env python3
"""
Build completion.json for all extraction types from existing files

This script scans the ROSBAGS directory (like 01_extract_media.py) and creates/updates
completion.json files for each extraction type based on what files actually exist,
without needing to rerun the extraction.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
from dotenv import load_dotenv

# Load environment variables
PARENT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Configuration
ROSBAGS = Path(os.getenv("ROSBAGS")) if os.getenv("ROSBAGS") else None
IMAGES = Path(os.getenv("IMAGES")) if os.getenv("IMAGES") else None
VIDEOS = Path(os.getenv("VIDEOS")) if os.getenv("VIDEOS") else None
POINTCLOUDS = Path(os.getenv("POINTCLOUDS")) if os.getenv("POINTCLOUDS") else None
POSITIONS = Path(os.getenv("POSITIONS")) if os.getenv("POSITIONS") else None
ODOM = Path(os.getenv("ODOM")) if os.getenv("ODOM") else None

# Extraction type configuration (matching 01_extract_media.py structure)
EXTRACTION_TYPES = ["image", "position", "video", "pointcloud", "odom"]
EXTRACTION_TYPE_DIRS = {
    "image": IMAGES,
    "position": POSITIONS,
    "video": VIDEOS,
    "pointcloud": POINTCLOUDS,
    "odom": ODOM
}



def find_rosbag_dirs(root: Path):
    """Find all rosbag directories (matching 01_extract_media.py logic)."""
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "metadata.yaml").exists():
            # Skip rosbags in EXCLUDED folders
            if "EXCLUDED" in str(p):
                continue
            yield p


def get_mcap_identifiers_from_rosbag(rosbag_path: Path) -> Set[str]:
    """Get all MCAP identifiers from actual MCAP files in the rosbag folder.
    
    Args:
        rosbag_path: Path to the rosbag directory
        
    Returns:
        Set of MCAP identifiers (e.g., {"0", "1", "100"})
    """
    def extract_mcap_identifier(mcap_path: Path) -> str:
        """Extract the MCAP identifier (number suffix) from filename."""
        name = mcap_path.stem  # Get filename without extension
        # Extract the last part after the last underscore (e.g., "100" from "rosbag2_2025_07_23-12_58_03_100")
        parts = name.split('_')
        if parts and parts[-1].isdigit():
            return parts[-1]
        return "0"  # Fallback for files without numeric suffix
    
    mcap_files = list(rosbag_path.glob("*.mcap"))
    return {extract_mcap_identifier(mcap) for mcap in mcap_files}


# Directory structure (hardcoded based on actual structure):
# - Position/Odom: extracted/type/rosbag_name/topic/mcap_id.csv
# - Images/Video/Pointcloud: extracted/type/rosbag_name/topic/mcap_id/data


def scan_single_rosbag_directory(rosbag_dir: Path, extraction_type: str) -> Optional[Dict]:
    """
    Scan a single rosbag directory and build completion data for it.
    
    Structure (hardcoded based on actual structure):
    - All types: extracted/type/rosbag_name/mcap_id/topic/data
    
    Args:
        rosbag_dir: Path to the rosbag directory in the output folder
        extraction_type: Type of extraction (image, position, video, pointcloud, odom)
    
    Returns:
        Dictionary with completion info and mcap_files, or None if no data found
    """
    if not rosbag_dir.exists():
        return None
    
    rosbag_name = rosbag_dir.name
    mcap_files = {}
    timestamp = datetime.now().isoformat()
    
    # Structure: extracted/type/rosbag_name/mcap_id/topic/data
    # First level: mcap_id directories (numeric or any subdirectory)
    for mcap_dir in rosbag_dir.iterdir():
        if not mcap_dir.is_dir():
            continue
        
        mcap_id = mcap_dir.name
        
        # Check if this mcap_id directory has topic subdirectories with data
        has_data = False
        
        if extraction_type == "position" or extraction_type == "odom":
            # For position/odom: look for CSV files in topic subdirectories
            # Structure: rosbag_name/mcap_id/topic/*.csv
            csv_files = list(mcap_dir.rglob("*.csv"))
            if csv_files:
                has_data = True
        else:
            # For images/video/pointcloud: look for files in topic subdirectories
            # Structure: rosbag_name/mcap_id/topic/data
            # Check if any topic subdirectory has files
            for topic_dir in mcap_dir.iterdir():
                if topic_dir.is_dir():
                    # Check if this topic directory has any files
                    if any(topic_dir.rglob("*")):
                        has_data = True
                        break
        
        if has_data:
            mcap_files[mcap_id] = {"completed_at": timestamp}
    
    if mcap_files:
        return {
            "completed_at": timestamp,
            "mcap_files": mcap_files
        }
    return None


def scan_extraction_type_directory_selective(extraction_type: str, rosbag_output_dirs: List[Path]) -> Dict[str, Dict]:
    """
    Scan only specific rosbag directories and build completion data.
    
    Args:
        extraction_type: Type of extraction (image, position, video, pointcloud, odom)
        rosbag_output_dirs: List of rosbag output directories to scan (e.g., output_dir/rosbag_name)
    
    Returns:
        Dictionary mapping rosbag_name to completion info with mcap_files
    """
    completed = {}
    
    # Process each rosbag directory
    for rosbag_output_dir in rosbag_output_dirs:
        rosbag_name = rosbag_output_dir.name
        result = scan_single_rosbag_directory(rosbag_output_dir, extraction_type)
        
        if result:
            completed[rosbag_name] = result
            print(f"  ✅ {rosbag_name}: Found {len(result.get('mcap_files', {}))} MCAP files")
    
    return completed


def is_completion_complete(completion_data: Dict[str, Dict], rosbag_name: str, 
                          actual_mcaps: Set[str]) -> bool:
    """Check if completion.json is complete for a rosbag.
    
    Args:
        completion_data: Loaded completion.json data
        rosbag_name: Name of the rosbag
        actual_mcaps: Set of MCAP identifiers that actually exist in the rosbag
    
    Returns:
        True if completion is complete (all actual MCAPs are recorded)
    """
    if rosbag_name not in completion_data:
        return False
    
    recorded_mcaps = set(completion_data[rosbag_name].get("mcap_files", {}).keys())
    # Filter out "additional" entries (from overlapping topics)
    recorded_mcaps = {m for m in recorded_mcaps if not m.startswith("additional")}
    
    # Check if all actual MCAPs are recorded
    return actual_mcaps.issubset(recorded_mcaps)


def load_completion(extraction_type: str) -> Dict[str, Dict]:
    """Load existing completion.json for an extraction type."""
    output_dir = EXTRACTION_TYPE_DIRS.get(extraction_type)
    if not output_dir:
        return {}
    
    completion_file = output_dir / "completion.json"
    if not completion_file.exists():
        return {}
    
    try:
        with open(completion_file, 'r') as f:
            data = json.load(f)
            completed_list = data.get('completed', [])
            result = {}
            
            for item in completed_list:
                if isinstance(item, dict):
                    rosbag_name = item.get("rosbag")
                    if rosbag_name:
                        result[rosbag_name] = {
                            "completed_at": item.get("completed_at", "unknown"),
                            "mcap_files": item.get("mcap_files", {})
                        }
            return result
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return {}


def save_completion_json(completed: Dict[str, Dict], output_path: Path):
    """Save completion data to JSON file in the expected format."""
    # Convert to the format expected by the extraction script
    data = {
        'completed': [
            {
                "rosbag": name,
                "completed_at": info.get("completed_at", "unknown"),
                "mcap_files": info.get("mcap_files", {})
            }
            for name, info in sorted(completed.items())
        ],
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  💾 Saved completion.json to: {output_path}")


def main():
    """Main function to build completion.json for all extraction types from existing files."""
    print("📋 Building completion.json for all extraction types")
    print("=" * 70)
    
    if not ROSBAGS or not ROSBAGS.exists():
        print(f"❌ ROSBAGS directory does not exist: {ROSBAGS}")
        return
    
    # Find all rosbag directories (like 01_extract_media.py does)
    print(f"\n🔍 Scanning ROSBAGS directory: {ROSBAGS}")
    rosbag_dirs = list(find_rosbag_dirs(ROSBAGS))
    print(f"📁 Found {len(rosbag_dirs)} rosbag directories")
    
    if not rosbag_dirs:
        print("❌ No rosbags found!")
        return
    
    # Process each extraction type
    for extraction_type in EXTRACTION_TYPES:
        output_dir = EXTRACTION_TYPE_DIRS.get(extraction_type)
        if not output_dir:
            print(f"\n⚠️  {extraction_type}: Output directory not configured, skipping")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"📦 Processing {extraction_type}")
        print(f"{'=' * 70}")
        
        # Load existing completion.json at the beginning
        completion_file = output_dir / "completion.json"
        existing_completion = load_completion(extraction_type)
        
        if existing_completion:
            print(f"  📂 Loaded existing completion.json with {len(existing_completion)} rosbag(s)")
        else:
            print(f"  📝 No existing completion.json found, will create new one")
        
        # Find which rosbags need to be scanned (not complete or not in completion.json)
        rosbag_dirs_to_scan = []
        skipped_complete = 0
        
        for rosbag_dir in sorted(rosbag_dirs):
            rosbag_name = rosbag_dir.name
            
            # Check if this rosbag is already complete
            if rosbag_name in existing_completion:
                # Get actual MCAPs from rosbag directory to check completeness
                rosbag_path = ROSBAGS / rosbag_name
                if rosbag_path.exists():
                    actual_mcaps = get_mcap_identifiers_from_rosbag(rosbag_path)
                    if is_completion_complete(existing_completion, rosbag_name, actual_mcaps):
                        # Already complete, skip scanning
                        skipped_complete += 1
                        continue
            
            # Need to scan this rosbag (either not in completion.json or incomplete)
            rosbag_dirs_to_scan.append(rosbag_dir)
        
        if skipped_complete > 0:
            print(f"  ⏭️  Skipping {skipped_complete} already completed rosbag(s)")
        
        if not rosbag_dirs_to_scan:
            print(f"  ✅ All rosbags are already complete, nothing to scan")
            # Still save the existing completion (in case it exists)
            if existing_completion:
                save_completion_json(existing_completion, completion_file)
            continue
        
        print(f"  🔍 Scanning {len(rosbag_dirs_to_scan)} rosbag(s) that need updating")
        
        # Scan only the rosbags that need updating
        # Map rosbag names to their output directories
        rosbag_output_dirs = []
        for rosbag_dir in rosbag_dirs_to_scan:
            rosbag_name = rosbag_dir.name
            rosbag_output_dir = output_dir / rosbag_name
            if rosbag_output_dir.exists():
                rosbag_output_dirs.append(rosbag_output_dir)
        
        scanned_completion = scan_extraction_type_directory_selective(extraction_type, rosbag_output_dirs)
        
        if not scanned_completion:
            print(f"  ⚠️  No files found for {extraction_type}")
            # Still save the existing completion (in case it exists but no new files)
            if existing_completion:
                save_completion_json(existing_completion, completion_file)
            continue
        
        # Extend existing completion incrementally
        updated_count = 0
        created_count = 0
        
        for rosbag_name, scanned_info in scanned_completion.items():
            # Get actual MCAPs from rosbag directory
            rosbag_path = ROSBAGS / rosbag_name
            if rosbag_path.exists():
                actual_mcaps = get_mcap_identifiers_from_rosbag(rosbag_path)
            else:
                actual_mcaps = set(scanned_info.get("mcap_files", {}).keys())
            
            # Check if completion is needed
            if rosbag_name not in existing_completion:
                # New rosbag, add it
                existing_completion[rosbag_name] = scanned_info
                created_count += 1
            elif not is_completion_complete(existing_completion, rosbag_name, actual_mcaps):
                # Incomplete, update it (merge mcap_files, keeping existing ones)
                existing_entry = existing_completion[rosbag_name]
                existing_mcap_files = existing_entry.get("mcap_files", {})
                scanned_mcap_files = scanned_info.get("mcap_files", {})
                
                # Merge mcap_files, with scanned taking precedence for updates
                merged_mcap_files = existing_mcap_files.copy()
                merged_mcap_files.update(scanned_mcap_files)
                
                # Update the entry
                existing_completion[rosbag_name] = {
                    "completed_at": scanned_info.get("completed_at", existing_entry.get("completed_at", datetime.now().isoformat())),
                    "mcap_files": merged_mcap_files
                }
                updated_count += 1
        
        # Save completion.json (with all updates)
        save_completion_json(existing_completion, completion_file)
        
        # Summary for this extraction type
        total_rosbags = len(existing_completion)
        total_mcaps = sum(len(info.get("mcap_files", {})) for info in existing_completion.values())
        print(f"  📊 Summary: {total_rosbags} rosbags, {total_mcaps} total MCAP files")
        if created_count > 0:
            print(f"  ✨ Created entries for {created_count} new rosbag(s)")
        if updated_count > 0:
            print(f"  🔄 Updated {updated_count} incomplete rosbag(s)")
    
    print("\n" + "=" * 70)
    print("✅ Done!")


if __name__ == "__main__":
    main()

