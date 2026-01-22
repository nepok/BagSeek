#!/usr/bin/env python3
"""
ROS2 Unbag Extraction Script

This script extracts data from ROS2 MCAP files using ros2_unbag.

Modes:
- "single": Process one specific rosbag (set SINGLE_BAG_NAME)
- "multiple": Process specific rosbags listed in MULTIPLE_BAG_NAMES array
- "all": Process all rosbags found in ROSBAGS directory

Extraction Types:
- "image": Extract images using dynamic config generation
- "position": Extract GPS data using predefined config file
- "video": Extract video data using predefined config file
- "pointcloud": Extract pointcloud data using predefined config file
- "odom": Extract odom data using predefined config file

Verbosity Modes:
- "quiet": Only show success/failure messages
- "verbose": Show full ros2 unbag progress bars and details

Topic Filtering:
- EXCLUDED_TOPICS: List of topic patterns to exclude from processing
- Case-insensitive partial matching (e.g., "odom" excludes "/novatel/oem7/odom")

Completion Tracking:
- Tracks completed rosbags per extraction type in JSON file (completion.json) in each output directory
- Each extraction type (image, position, video, pointcloud, odom) has its own completion file
- Verifies completion by checking output directories
- Stores timestamps for when each rosbag was completed
- SKIP_COMPLETED: Set to True to skip completed rosbags, False to reprocess all

Usage:
1. Edit the CONFIG section below
2. Run: python3 export_images.py
"""
import json
import re
import os
import shutil
import subprocess
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Set, Union, Optional, Dict, List, Tuple
from dotenv import load_dotenv

# =========================
# Load environment variables
# =========================

PARENT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

ROSBAGS = Path(os.getenv("ROSBAGS")) if os.getenv("ROSBAGS") else None#

BASE_STR = os.getenv("BASE")

IMAGES_STR = os.getenv("IMAGES")
ODOM_STR = os.getenv("ODOM")
POINTCLOUDS_STR = os.getenv("POINTCLOUDS")
POSITIONS_STR = os.getenv("POSITIONS")
VIDEOS_STR = os.getenv("VIDEOS")

IMAGES = Path(BASE_STR + IMAGES_STR)
ODOM = Path(BASE_STR + ODOM_STR)
POINTCLOUDS = Path(BASE_STR + POINTCLOUDS_STR)
POSITIONS = Path(BASE_STR + POSITIONS_STR)
VIDEOS = Path(BASE_STR + VIDEOS_STR)

# =========================
# CONFIG (edit these)
# =========================


MODE = "all"            # "single", "all", or "multiple"
#EXTRACTION_TYPES = ["image", "position", "pointcloud", "odom"]  # List of extraction types: "image", "position", "video", "pointcloud", "odom"
EXTRACTION_TYPES = ["image"]
SINGLE_BAG_NAME = ""   # only used when MODE = "single"

# MULTIPLE_BAG_NAMES: Dict format with rosbag name as key and list of mcap numbers as value
# If mcap numbers list is empty or None, all mcaps will be processed
"""MULTIPLE_BAG_NAMES = {
    "rosbag2_2025_08_29-11_00_49": [24],
    #"rosbag2_2025_09_25-08_10_26": [2],
    #"rosbag2_2025_09_25-08_26_44": [11],
    #"rosbag2_2025_09_09-10_21_16": [15],
    #"rosbag2_2025_09_24-14_30_42": [2],
    #"rosbag2_2025_09_04-15_01_01": [3, 6],
    #"rosbag2_2025_09_26-12_49_12": [6],
    #"rosbag2_2025_09_09-10_43_40": [3, 22],
    #"rosbag2_2025_09_10-09_15_13": [7],
    #"rosbag2_2025_08_08-13_59_06": [2, 118],
    #"rosbag2_2025_08_20-08_59_36": [0],
    #"rosbag2_2025_08_30-15_19_54": [5],
    #"rosbag2_2025_08_11-19_20_47": [78, 84],
}"""

MULTIPLE_BAG_NAMES = {
    "rosbag2_2025_07_23-12_58_03": [145, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169],
    "rosbag2_2025_07_28-07_29_07": [18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 128, 129, 206, 207],
    "rosbag2_2025_07_28-09_17_32": [22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 83],
    "rosbag2_2025_07_28-12_45_50": [28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68],
    "rosbag2_2025_07_29-08_08_48": [17, 18, 34],
    "rosbag2_2025_07_29-09_46_30": [17, 51, 181, 189, 190],
    "rosbag2_2025_08_06-10_06_45": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    "rosbag2_2025_08_19-07_20_08": [8, 9],
    "rosbag2_2025_08_22-12_27_43": [12, 13, 14, 15, 16],
    "rosbag2_2025_08_26-08_22_25": [67, 68],
    "rosbag2_2025_08_26-09_24_03": [75, 76],
    "rosbag2_2025_08_29-11_00_49": [7, 9],
    "rosbag2_2025_09_02-10_18_00": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
    "rosbag2_2025_09_02-13_04_34": [142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152],
    "rosbag2_2025_09_03-07_34_05": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    "rosbag2_2025_09_05-09_04_39": [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
    "rosbag2_2025_09_08-12_35_09": [84, 85, 106, 147, 180, 221, 222, 268, 311, 402, 455, 485, 526, 527, 564, 610, 636, 637, 700, 728, 783, 826],
    "rosbag2_2025_09_12-13_34_47": [82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93],
    "rosbag2_2025_09_15-07_34_08": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "rosbag2_2025_09_15-10_10_04": [356, 357, 358, 359, 360, 361, 362, 363, 364, 385, 386],
    "rosbag2_2025_09_17-07_18_04": [0, 17, 18],
    "rosbag2_2025_09_26-08_17_29": [614, 629],
    "rosbag2_2025_09_26-13_13_20": [228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462],
    "rosbag2_2025_09_28-14_22_56": [20, 261, 262],
    "rosbag2_2025_09_30-11_53_35": [130, 137, 138, 181, 188],
}

# Topic search terms for each extraction type (edit these lists as needed)
TOPIC_SEARCH_TERMS: Dict[str, List[str]] = {
    "image": ["image"],           # Dummy - edit as needed
    #"image": ["zed_node/left_raw"],
    "position": ["bestpos"],           # Dummy - edit as needed
    "video": ["image"],           # Dummy - edit as needed
    "pointcloud": ["zed_node/point_cloud"], # Dummy - edit as needed
    "odom": ["zed_node/odom"]              # Dummy - edit as needed
}

# Extraction type configuration
EXTRACTION_TYPE_CONFIG: Dict[str, Dict] = {
    "image": {
        "output_dir": IMAGES,
        "format": "image/png"
    },
    "position": {
        "output_dir": POSITIONS,
        "format": "table/csv@single_file"
    },
    "video": {
        "output_dir": VIDEOS,
        "format": "video/mp4"
    },
    "pointcloud": {
        "output_dir": POINTCLOUDS,
        "format": "pointcloud/pcd"
    },
    "odom": {
        "output_dir": ODOM,
        "format": "table/csv@single_file"
    }
}

# Ensure output_dir is a Path object
for ext_type, config in EXTRACTION_TYPE_CONFIG.items():
    if isinstance(config["output_dir"], str):
        config["output_dir"] = Path(config["output_dir"])
    elif not isinstance(config["output_dir"], Path):
        config["output_dir"] = Path(config["output_dir"])
    
TMP_DIR     = Path("/mnt/data/tmp/ros2unbag_configs")
CPU_PERCENT = 35.0       # __global__.cpu_percentage

# Output verbosity control
VERBOSITY = "quiet"  # "quiet" (only show success/failure messages) or "verbose" (show full ros2 unbag progress bars and details)

# Topics to exclude from processing (case-insensitive partial matching)
# Examples: ["odom", "imu", "camera_info"] - will exclude any topic containing these terms
EXCLUDED_TOPICS = []  # List of topic names/patterns to exclude

SKIP_COMPLETED = True  # Set to False to reprocess completed rosbags

# =========================
# Global process tracking for cleanup
# =========================
_active_process: Optional[subprocess.Popen] = None
_interrupted = False


def find_rosbag_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "metadata.yaml").exists():
            # Skip rosbags in EXCLUDED folders
            if "EXCLUDED" in str(p):
                continue
            yield p


def get_completion_file(extraction_type: str) -> Path:
    """Get the completion file path for a specific extraction type."""
    output_dir = EXTRACTION_TYPE_CONFIG[extraction_type]["output_dir"]
    return output_dir / "completion.json"


def load_completion(extraction_type: str) -> Dict[str, Dict]:
    """Load the dictionary of completed rosbag data from the completion file.
    
    Returns:
        Dict mapping rosbag_name to dict with keys: completed_at, errors (optional), mcap_files (optional)
        mcap_files: Dict mapping mcap_id to dict with completed_at, topics, errors
    """
    completion_file = get_completion_file(extraction_type)
    if not completion_file.exists():
        return {}
    
    try:
        with open(completion_file, 'r') as f:
            data = json.load(f)
            # Handle different formats
            completed_list = data.get('completed', [])
            result = {}
            
            for item in completed_list:
                if isinstance(item, str):
                    # Old format - just rosbag name
                    result[item] = {"completed_at": "unknown", "mcap_files": {}}
                elif isinstance(item, dict):
                    # New format - dict with rosbag name and optional fields
                    rosbag_name = item.get("rosbag")
                    if rosbag_name:
                        result[rosbag_name] = {
                            "completed_at": item.get("completed_at", "unknown"),
                            "errors": item.get("errors", []),
                            "mcap_files": item.get("mcap_files", {})
                        }
            return result
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        return {}


def save_completed_mcap(rosbag_name: str, mcap_identifier: str, extraction_type: str, 
                        topics: Optional[List[str]] = None, error: Optional[str] = None):
    """Mark a specific MCAP file as completed and save to the completion file.
    
    Args:
        rosbag_name: Name of the rosbag
        mcap_identifier: MCAP identifier (e.g., "0", "1", "100")
        extraction_type: Type of extraction (image, position, video, etc.)
        topics: Optional list of topics processed for this MCAP
        error: Optional error message if processing failed
    """
    completed = load_completion(extraction_type)
    timestamp = datetime.now().isoformat()
    
    # Initialize rosbag entry if it doesn't exist
    if rosbag_name not in completed:
        completed[rosbag_name] = {
            "completed_at": timestamp,
            "mcap_files": {}
        }
    
    # Update mcap_files
    if "mcap_files" not in completed[rosbag_name]:
        completed[rosbag_name]["mcap_files"] = {}
    
    # Update or create entry for this MCAP
    mcap_entry = {
        "completed_at": timestamp
    }
    if topics:
        mcap_entry["topics"] = topics
    if error:
        mcap_entry["error"] = error
    
    completed[rosbag_name]["mcap_files"][mcap_identifier] = mcap_entry
    
    # Update rosbag-level completed_at to latest MCAP completion
    completed[rosbag_name]["completed_at"] = timestamp
    
    # Save to file
    save_completion(extraction_type, completed)


def save_completed_rosbag(rosbag_name: str, extraction_type: str, errors: Optional[List[Dict[str, str]]] = None):
    """Mark a rosbag as completed and save to the completion file with timestamp and optional errors.
    
    Args:
        rosbag_name: Name of the rosbag
        extraction_type: Type of extraction (image, position, video, etc.)
        errors: Optional list of error dicts with keys "mcap" and "error"
    """
    completed = load_completion(extraction_type)
    timestamp = datetime.now().isoformat()
    
    # Update or create entry for this rosbag
    if rosbag_name not in completed:
        completed[rosbag_name] = {
            "completed_at": timestamp,
            "mcap_files": {}
        }
    else:
        completed[rosbag_name]["completed_at"] = timestamp
    
    if errors:
        completed[rosbag_name]["errors"] = errors
        # Also add errors to mcap_files if applicable
        if "mcap_files" not in completed[rosbag_name]:
            completed[rosbag_name]["mcap_files"] = {}
        for error_info in errors:
            mcap_name = error_info.get("mcap", "")
            # Try to extract mcap identifier from mcap name
            if mcap_name:
                # Extract identifier (last part after underscore, or use full name)
                parts = mcap_name.split('_')
                if parts and parts[-1].endswith('.mcap'):
                    mcap_id = parts[-1].replace('.mcap', '')
                else:
                    mcap_id = mcap_name
                
                if mcap_id not in completed[rosbag_name]["mcap_files"]:
                    completed[rosbag_name]["mcap_files"][mcap_id] = {}
                completed[rosbag_name]["mcap_files"][mcap_id]["error"] = error_info.get("error", "Unknown error")
    
    # Save to file
    save_completion(extraction_type, completed)


def save_completion(extraction_type: str, completed: Dict[str, Dict]):
    """Save completed rosbags dictionary to completion file.
    
    Args:
        extraction_type: Type of extraction
        completed: Dictionary of completed rosbags to save
    """
    completion_file = get_completion_file(extraction_type)
    completion_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file with proper format
    data = {
        'completed': [
            {
                "rosbag": name,
                "completed_at": info.get("completed_at", "unknown"),
                **({"errors": info["errors"]} if info.get("errors") else {}),
                **({"mcap_files": info["mcap_files"]} if info.get("mcap_files") else {})
            }
            for name, info in sorted(completed.items())
        ],
    }
    
    with open(completion_file, 'w') as f:
        json.dump(data, f, indent=2)


def is_mcap_completed(rosbag_name: str, mcap_identifier: str, extraction_type: str) -> bool:
    """Check if a specific MCAP file has been completed for a specific extraction type.
    
    Args:
        rosbag_name: Name of the rosbag
        mcap_identifier: MCAP identifier (e.g., "0", "1", "100")
        extraction_type: Type of extraction
        
    Returns:
        True if MCAP is marked as completed (even if there was an error)
    """
    if not SKIP_COMPLETED:
        return False
    
    completed = load_completion(extraction_type)
    if rosbag_name not in completed:
        return False
    
    mcap_files = completed[rosbag_name].get("mcap_files", {})
    return mcap_identifier in mcap_files


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


def get_completed_mcap_identifiers(rosbag_name: str, extraction_type: str) -> Set[str]:
    """Get all MCAP identifiers that are marked as completed in completion.json.
    
    Args:
        rosbag_name: Name of the rosbag
        extraction_type: Type of extraction
        
    Returns:
        Set of MCAP identifiers that are marked as completed
    """
    completed = load_completion(extraction_type)
    if rosbag_name not in completed:
        return set()
    
    mcap_files = completed[rosbag_name].get("mcap_files", {})
    # Return all MCAP identifiers (excluding additional config ones)
    return {mcap_id for mcap_id in mcap_files.keys() if not mcap_id.startswith("additional")}


def is_rosbag_fully_completed(rosbag_path: Path, extraction_type: str, mcap_filter: Optional[List[int]] = None) -> bool:
    """Check if all MCAPs in a rosbag have been completed for a specific extraction type.
    
    Compares the actual MCAP files in the rosbag folder (or filtered subset) with what's recorded in completion.json.
    
    Args:
        rosbag_path: Path to the rosbag directory
        extraction_type: Type of extraction
        mcap_filter: Optional list of mcap numbers to check. If None, checks all mcaps.
        
    Returns:
        True if all MCAPs (or filtered MCAPs) have been processed (even if some had errors)
    """
    if not SKIP_COMPLETED:
        return False
    
    def extract_number(mcap_path: Path) -> int:
        """Extract the number from MCAP filename for proper numerical sorting."""
        name = mcap_path.stem
        parts = name.split('_')
        if parts and parts[-1].isdigit():
            return int(parts[-1])
        return 0
    
    def extract_mcap_identifier(mcap_path: Path) -> str:
        """Extract the MCAP identifier (number suffix) from filename."""
        name = mcap_path.stem
        parts = name.split('_')
        if parts and parts[-1].isdigit():
            return parts[-1]
        return "0"
    
    # Get actual MCAP files
    all_mcap_files = list(rosbag_path.glob("*.mcap"))
    
    # Filter if mcap_filter is provided
    if mcap_filter is not None:
        mcap_files = [mcap for mcap in all_mcap_files if extract_number(mcap) in mcap_filter]
    else:
        mcap_files = all_mcap_files
    
    if not mcap_files:
        # No MCAPs found (or all filtered out), consider it "completed" (nothing to do)
        return True
    
    # Get MCAP identifiers from the files we're checking
    actual_mcaps = {extract_mcap_identifier(mcap) for mcap in mcap_files}
    
    # Get completed MCAP identifiers from completion.json
    rosbag_name = rosbag_path.name
    completed_mcaps = get_completed_mcap_identifiers(rosbag_name, extraction_type)
    
    # Check if all actual MCAPs are in the completed set
    return actual_mcaps.issubset(completed_mcaps)


def is_rosbag_completed(rosbag_name: str, extraction_type: str) -> bool:
    """Check if a rosbag has been completed for a specific extraction type.
    
    Note: This is a legacy function. Use is_rosbag_fully_completed() for proper checking.
    Returns True even if there were errors, as long as processing was attempted.
    """
    if not SKIP_COMPLETED:
        return False
    
    completed = load_completion(extraction_type)
    return rosbag_name in completed


def verify_completion(rosbag_name: str, extraction_type: str) -> bool:
    """Verify that a rosbag was actually completed by checking output directories."""
    output_dir = EXTRACTION_TYPE_CONFIG[extraction_type]["output_dir"]
    
    if extraction_type == "position" or extraction_type == "odom":
        # For position/odom: Check if rosbag directory exists and has CSV files (structure: rosbag/topic/{mcap}.csv)
        bag_output_dir = output_dir / rosbag_name
        if not bag_output_dir.exists():
            return False
        # Check if directory has any CSV files (recursively, structure is rosbag/topic/{mcap}.csv)
        return any(bag_output_dir.rglob("*.csv"))  # Check recursively for CSV files
    elif extraction_type == "image" or extraction_type == "pointcloud":
        # For images/pointcloud: Check if the main output directory exists and has files (structure: rosbag/topic/{mcap}/)
        bag_output_dir = output_dir / rosbag_name
        if not bag_output_dir.exists():
            return False
        # Check if directory has any files (recursively, structure is rosbag/topic/{mcap}/)
        return any(bag_output_dir.rglob("*"))  # Check recursively for any files
    elif extraction_type == "video":
        # For video: Check if there's at least one video file (structure: rosbag/topic/{mcap}/)
        bag_output_dir = output_dir / rosbag_name
        if not bag_output_dir.exists():
            return False
        # Check recursively for video files (mp4)
        return any(f.suffix == ".mp4" for f in bag_output_dir.rglob("*.mp4") if f.is_file())
    else:
        raise ValueError(f"Unknown extraction type: {extraction_type}")


def parse_topics_from_metadata(meta_path: Path, search_terms: List[str]) -> Set[str]:
    """
    Parse ROS2 bag metadata.yaml and return topic names that contain any of the specified search terms (case-insensitive).
    Works with common rosbag2 metadata schemas (including MCAP).
    
    Args:
        meta_path: Path to the metadata.yaml file
        search_terms: List of terms to search for in topic names (e.g., ["image"], ["novatel"], ["video", "camera"])
    
    Returns:
        Set of topic names containing any of the search terms (excluding topics in EXCLUDED_TOPICS)
    """
    text = meta_path.read_text(encoding="utf-8", errors="ignore")
    topics: Set[str] = set()

    # Try PyYAML if available (more accurate)
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)

        def walk(node: Union[dict, list]):
            if isinstance(node, dict):
                # Typical structure: topics_with_message_count → [ { topic_metadata: { name: "/foo" } } ]
                if "topic_metadata" in node and isinstance(node["topic_metadata"], dict):
                    name = node["topic_metadata"].get("name")
                    if isinstance(name, str):
                        for search_term in search_terms:
                            if search_term.lower() in name.lower():
                                topics.add(name)
                                break
                # generic fields
                for k, v in node.items():
                    if k in ("name", "topic") and isinstance(v, str) and v.startswith("/"):
                        for search_term in search_terms:
                            if search_term.lower() in v.lower():
                                topics.add(v)
                                break
                    elif isinstance(v, (dict, list)):
                        walk(v)
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, (dict, list)):
                        walk(item)

        walk(data)

    except Exception:
        # Fallback: regex scan when no YAML parser installed
        for ln in text.splitlines():
            for search_term in search_terms:
                if search_term.lower() in ln.lower():
                    for t in re.findall(r"(/[\w/]+)", ln):
                        if search_term.lower() in t.lower():
                            topics.add(t)
                    break

    # Filter out excluded topics
    filtered_topics = set()
    for topic in topics:
        should_exclude = False
        for excluded_term in EXCLUDED_TOPICS:
            if excluded_term.lower() in topic.lower():
                should_exclude = True
                break
        if not should_exclude:
            filtered_topics.add(topic)
    
    return filtered_topics

def build_topic_config(topic: str, extraction_type: str, rosbag_name: str, mcap_identifier: Optional[str] = None) -> dict:
    """Build config for a single topic based on extraction type."""
    config = EXTRACTION_TYPE_CONFIG[extraction_type]
    output_dir = config["output_dir"]
    format_str = config["format"]
    
    if extraction_type == "position" or extraction_type == "odom":
        # For position/odom: rosbag/topic/{mcap}.csv structure
        bag_out = output_dir / rosbag_name
        return {
            "format": format_str,
            "path": str(bag_out),
            "subfolder": "%name",
            "naming": f"{mcap_identifier}.csv" if mcap_identifier else f"{rosbag_name}.csv",
        }
    elif extraction_type == "image" or extraction_type == "pointcloud":
        # For images/pointcloud: rosbag/{mcap}/topic/ structure
        bag_out = output_dir / rosbag_name
        return {
            "format": format_str,
            "path": str(bag_out),
            "subfolder": f"{mcap_identifier}/%name" if mcap_identifier else "%name",
            "naming": "%timestamp",
        }
    elif extraction_type == "video":
        # For video: rosbag/{mcap}/topic/ structure (one video per mcap)
        bag_out = output_dir / rosbag_name
        return {
            "format": format_str,
            "path": str(bag_out),
            "subfolder": f"{mcap_identifier}/%name" if mcap_identifier else "%name",
            "naming": str(mcap_identifier) if mcap_identifier else str(rosbag_name),
        }
    else:
        raise ValueError(f"Unknown extraction type: {extraction_type}")


def find_topic_overlaps(all_topics_by_type: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Find topics that appear in multiple extraction types.
    
    Returns:
        Dictionary mapping topic name to list of extraction types that contain it
        Only includes topics that appear in 2+ extraction types
        Extraction types are ordered according to EXTRACTION_TYPES order
    """
    topic_to_types: Dict[str, List[str]] = {}
    
    # Build reverse mapping: topic -> list of extraction types (in EXTRACTION_TYPES order)
    for extraction_type in EXTRACTION_TYPES:
        if extraction_type not in all_topics_by_type:
            continue
        topics = all_topics_by_type[extraction_type]
        for topic in topics:
            if topic not in topic_to_types:
                topic_to_types[topic] = []
            topic_to_types[topic].append(extraction_type)
    
    # Filter to only overlaps (2+ extraction types)
    overlaps = {topic: types for topic, types in topic_to_types.items() if len(types) > 1}
    return overlaps


def build_primary_config(all_topics_by_type: Dict[str, Set[str]], overlaps: Dict[str, List[str]], 
                         rosbag_name: str, mcap_identifier: Optional[str] = None) -> Tuple[dict, List[str]]:
    """Build primary config with unique topics + overlapping topics with first format.
    
    Args:
        all_topics_by_type: Topics grouped by extraction type
        overlaps: Topics that appear in multiple extraction types
        rosbag_name: Name of the rosbag
        mcap_identifier: Optional MCAP identifier
    
    Returns:
        Tuple of (config dictionary, list of extraction types used in this config)
    """
    cfg = {}
    used_topics = set()  # Track topics already added
    extraction_types_used = set()  # Track which extraction types are in this config
    
    # Process extraction types in order (first format wins for overlaps)
    for extraction_type in EXTRACTION_TYPES:
        if extraction_type not in all_topics_by_type:
            continue
        
        topics = all_topics_by_type[extraction_type]
        for topic in sorted(topics):
            # For overlapping topics, only add if this is the first extraction type
            if topic in overlaps:
                # Check if this is the first extraction type for this topic
                first_type = overlaps[topic][0]
                if extraction_type != first_type:
                    continue  # Skip - will be handled in additional config
            
            if topic not in used_topics:
                cfg[topic] = build_topic_config(topic, extraction_type, rosbag_name, mcap_identifier)
                used_topics.add(topic)
                extraction_types_used.add(extraction_type)
    
    # Add global settings
    cfg["__global__"] = {"cpu_percentage": float(CPU_PERCENT)}
    return cfg, sorted(extraction_types_used)


def build_additional_configs(all_topics_by_type: Dict[str, Set[str]], overlaps: Dict[str, List[str]],
                            rosbag_name: str, mcap_identifier: Optional[str] = None) -> List[Tuple[dict, List[str]]]:
    """Build additional configs for overlapping topics with remaining formats.
    
    Args:
        all_topics_by_type: Topics grouped by extraction type
        overlaps: Topics that appear in multiple extraction types
        rosbag_name: Name of the rosbag
        mcap_identifier: Optional MCAP identifier
    
    Returns:
        List of tuples: (config dictionary, list of extraction types used in this config)
    """
    if not overlaps:
        return []  # No overlaps, no additional configs needed
    
    # Group overlapping topics by their remaining formats
    # Format: {format_str: {topic: extraction_type}}
    format_groups: Dict[str, Dict[str, str]] = {}
    
    for topic, extraction_types in overlaps.items():
        # Skip the first extraction type (already in primary config)
        for extraction_type in extraction_types[1:]:
            format_str = EXTRACTION_TYPE_CONFIG[extraction_type]["format"]
            if format_str not in format_groups:
                format_groups[format_str] = {}
            format_groups[format_str][topic] = extraction_type
    
    # Build one config per format group
    additional_configs = []
    for format_str, topics_dict in format_groups.items():
        cfg = {}
        extraction_types_used = set()
        for topic, extraction_type in sorted(topics_dict.items()):
            cfg[topic] = build_topic_config(topic, extraction_type, rosbag_name, mcap_identifier)
            extraction_types_used.add(extraction_type)
        cfg["__global__"] = {"cpu_percentage": float(CPU_PERCENT)}
        additional_configs.append((cfg, sorted(extraction_types_used)))
    
    return additional_configs


def write_config(cfg: dict, rosbag_name: str, mcap_identifier: Optional[str] = None) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    if mcap_identifier:
        path = TMP_DIR / f"ros2unbag_config_{rosbag_name}_{mcap_identifier}.json"
    else:
        path = TMP_DIR / f"ros2unbag_config_{rosbag_name}.json"
    path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    return path


def cleanup_process():
    """Terminate any active subprocess."""
    global _active_process
    if _active_process is not None:
        try:
            print("\n🛑 Terminating subprocess...")
            _active_process.terminate()
            try:
                _active_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("⚠️  Process didn't terminate, forcing kill...")
                _active_process.kill()
                _active_process.wait()
            _active_process = None
            print("✅ Subprocess terminated")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")


def signal_handler(signum, frame):
    """Handle interrupt signals (Ctrl+C)."""
    global _interrupted
    _interrupted = True
    print("\n\n⚠️  Interrupt signal received!")
    cleanup_process()
    sys.exit(130)  # Standard exit code for SIGINT


def run_ros2_unbag(mcap_path: Path, config_path: Path, extraction_types: Optional[List[str]] = None) -> tuple:
    """Run ros2 unbag with proper interrupt handling.
    
    Args:
        mcap_path: Path to the MCAP file
        config_path: Path to the config file
        extraction_types: Optional list of extraction types being processed (for display)
    
    Returns:
        Tuple of (return_code, error_message). error_message is None if successful.
    """
    global _active_process, _interrupted
    
    if _interrupted:
        return (1, "Processing interrupted")
    
    cmd = ["ros2", "unbag", str(mcap_path), "--config", str(config_path)]
    
    # Build display string with extraction types
    types_str = f" [{', '.join(extraction_types)}]" if extraction_types else ""
    
    stdout = None
    try:
        if VERBOSITY == "verbose":
            print(f"\n🔄 Running: {' '.join(cmd)}")
            print("=" * 80)
            # Use Popen so we can track and kill the process
            _active_process = subprocess.Popen(
                cmd, 
                text=True,
                stdout=None,  # Output goes directly to terminal
                stderr=subprocess.STDOUT
            )
            _active_process.wait()
            print("=" * 80)
        else:  # VERBOSITY == "quiet"
            # Capture output to suppress progress bars
            _active_process = subprocess.Popen(
                cmd, 
                text=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT
            )
            stdout, _ = _active_process.communicate()
        
        returncode = _active_process.returncode
        _active_process = None
        
        error_message = None
        if returncode != 0:
            print(f"❌ {mcap_path.name}, types: {types_str} - FAILED (exit code: {returncode})")
            if VERBOSITY == "quiet" and stdout:
                # Show error output in quiet mode
                print(f"   Error: {stdout}")
                # Extract error message from stdout (last few lines)
                error_lines = stdout.strip().split('\n')
                if error_lines:
                    error_message = error_lines[-1][:200]  # Last line, max 200 chars
            if not error_message:
                error_message = f"ros2 unbag failed with exit code {returncode}"
        else:
            print(f"✅ {mcap_path.name}, types: {types_str} - SUCCESS")
        
        return (returncode, error_message)
        
    except KeyboardInterrupt:
        print("\n⚠️  KeyboardInterrupt in run_ros2_unbag")
        cleanup_process()
        _interrupted = True
        raise
    except Exception as e:
        cleanup_process()
        error_msg = f"Exception: {type(e).__name__}: {str(e)}"
        print(f"❌ Error running ros2 unbag: {error_msg}")
        return (1, error_msg)


def process_bag(bag_dir: Path, mcap_filter: Optional[List[int]] = None):
    """Process a single rosbag for all configured extraction types.
    
    Args:
        bag_dir: Path to the rosbag directory
        mcap_filter: Optional list of mcap numbers to process. If None, process all mcaps.
    """
    global _interrupted
    
    if _interrupted:
        return
    
    rosbag_name = bag_dir.name
    meta = bag_dir / "metadata.yaml"
    
    # Sort MCAP files numerically instead of lexicographically
    def extract_number(mcap_path) -> int:
        """Extract the number from MCAP filename for proper numerical sorting."""
        name = mcap_path.stem  # Get filename without extension
        # Extract the last part after the last underscore (e.g., "100" from "rosbag2_2025_07_23-12_58_03_100")
        parts = name.split('_')
        if parts and parts[-1].isdigit():
            return int(parts[-1])
        return 0  # Fallback for files without numeric suffix
    
    def extract_mcap_identifier(mcap_path: Path) -> str:
        """Extract the MCAP identifier (number suffix) from filename."""
        name = mcap_path.stem  # Get filename without extension
        # Extract the last part after the last underscore (e.g., "100" from "rosbag2_2025_07_23-12_58_03_100")
        parts = name.split('_')
        if parts and parts[-1].isdigit():
            return parts[-1]
        return "0"  # Fallback for files without numeric suffix
    
    all_mcaps = sorted(bag_dir.glob("*.mcap"), key=extract_number)
    
    # Filter mcaps if filter is provided
    if mcap_filter is not None:
        mcaps = [mcap for mcap in all_mcaps if extract_number(mcap) in mcap_filter]
        if mcaps:
            print(f"\n📦 {rosbag_name}: {len(mcaps)} mcap file(s) (filtered from {len(all_mcaps)} total)")
        else:
            print(f"\n📦 {rosbag_name}: No matching mcap files (filter: {mcap_filter}, total: {len(all_mcaps)})")
    else:
        mcaps = all_mcaps

    if not mcaps:
        print(f"\n📦 {rosbag_name}: (skip: no .mcap files)")
        return

    print(f"\n📦 {rosbag_name}: {len(mcaps)} mcap file(s)")

    # Check completion status for all extraction types by comparing actual MCAPs with completion.json
    if SKIP_COMPLETED:
        all_completed = True
        for extraction_type in EXTRACTION_TYPES:
            if not is_rosbag_fully_completed(bag_dir, extraction_type, mcap_filter):
                all_completed = False
                break
        
        if all_completed:
            print(f"  ⏭️  All extraction types already completed for all MCAPs (skipping)")
            return

    # Parse topics for all extraction types
    all_topics_by_type: Dict[str, Set[str]] = {}
    for extraction_type in EXTRACTION_TYPES:
        search_terms = TOPIC_SEARCH_TERMS.get(extraction_type, [])
        if not search_terms:
            print(f"  ⚠️  No search terms configured for {extraction_type} (skipping)")
            continue
        
        topics = parse_topics_from_metadata(meta, search_terms)
        if topics:
            all_topics_by_type[extraction_type] = topics
            print(f"  📝 {extraction_type}: Found {len(topics)} topic(s)")
        else:
            print(f"  ⚠️  {extraction_type}: No topics found matching {search_terms}")
    
    if not all_topics_by_type:
        print(f"  ❌ No topics found for any extraction type (skipping)")
        return

    # Create output directories for all extraction types
    for extraction_type, topics in all_topics_by_type.items():
        output_dir = EXTRACTION_TYPE_CONFIG[extraction_type]["output_dir"]
        if extraction_type == "position" or extraction_type == "odom":
            (output_dir / rosbag_name).mkdir(parents=True, exist_ok=True)
        elif extraction_type == "image" or extraction_type == "video" or extraction_type == "pointcloud":
            (output_dir / rosbag_name).mkdir(parents=True, exist_ok=True)

    # Identify topic overlaps
    overlaps = find_topic_overlaps(all_topics_by_type)
    if overlaps:
        print(f"  🔄 Found {len(overlaps)} overlapping topic(s) - will create additional configs")
    
    # Process all MCAP files
    failures = 0
    errors = []  # Track errors per mcap file
    
    for mcap in mcaps:
        if _interrupted:
            print(f"  ⚠️  Processing interrupted, stopping {rosbag_name}")
            errors.append({
                "mcap": mcap.name,
                "error": "Processing interrupted by user"
            })
            break
        
        # Extract MCAP identifier for unique naming
        mcap_identifier = extract_mcap_identifier(mcap)
        
        # Check which extraction types are already completed for this MCAP
        completed_extraction_types = []
        remaining_extraction_types = {}
        
        if SKIP_COMPLETED:
            for extraction_type, topics in all_topics_by_type.items():
                if is_mcap_completed(rosbag_name, mcap_identifier, extraction_type):
                    completed_extraction_types.append(extraction_type)
                else:
                    remaining_extraction_types[extraction_type] = topics
            
            # Print skip messages only if verbose
            if completed_extraction_types and VERBOSITY == "verbose":
                print(f"  ⏭️  MCAP {mcap_identifier}: Already completed for {', '.join(completed_extraction_types)} (skipping those)")
        
            # If all extraction types are completed, skip this MCAP entirely
            if not remaining_extraction_types:
                if VERBOSITY == "verbose":
                    print(f"  ⏭️  MCAP {mcap_identifier} already completed for all extraction types (skipping)")
                continue
        else:
            remaining_extraction_types = all_topics_by_type.copy()
        
        # If no remaining extraction types, skip this MCAP
        if not remaining_extraction_types:
            continue
        
        # Recalculate overlaps for remaining extraction types
        remaining_overlaps = find_topic_overlaps(remaining_extraction_types)
        
        # Build primary config (unique topics + first format for overlaps)
        primary_cfg, primary_types = build_primary_config(remaining_extraction_types, remaining_overlaps, rosbag_name, mcap_identifier)
        
        # Skip if no primary config (all topics filtered out)
        if not primary_cfg or len(primary_cfg) <= 1:  # Only __global__ left
            if VERBOSITY == "verbose":
                print(f"  ⏭️  MCAP {mcap_identifier}: No remaining extraction types to process (skipping)")
            continue
        
        primary_cfg_path = write_config(primary_cfg, rosbag_name, mcap_identifier)
        
        # Build additional configs for overlapping topics with remaining formats
        additional_configs = build_additional_configs(remaining_extraction_types, remaining_overlaps, rosbag_name, mcap_identifier)
        
        # Run ros2 unbag with primary config
        types_str = ", ".join(primary_types)
        error_message = None
        success = False
        processed_topics = list(primary_cfg.keys())
        processed_topics.remove("__global__")  # Remove global settings
        
        try:
            rc, error_msg = run_ros2_unbag(mcap, primary_cfg_path, primary_types)
            if rc == 0:
                success = True
                # Log successful completion for each extraction type
                for ext_type in primary_types:
                    save_completed_mcap(rosbag_name, mcap_identifier, ext_type, topics=processed_topics)
            else:
                failures += 1
                error_message = error_msg or f"ros2 unbag failed with exit code {rc}"
                print(f"    ❌ Failed for {types_str}")
                # Log error for each extraction type
                for ext_type in primary_types:
                    save_completed_mcap(rosbag_name, mcap_identifier, ext_type, topics=processed_topics, error=error_message)

        except Exception as e:
            failures += 1
            error_message = f"Exception during processing: {type(e).__name__}: {str(e)}"
            print(f"    ❌ Failed for {types_str}: {error_message}")
            # Log error for each extraction type
            for ext_type in primary_types:
                save_completed_mcap(rosbag_name, mcap_identifier, ext_type, topics=processed_topics, error=error_message)
        finally:
            # Clean up primary config file
            try:
                if primary_cfg_path.exists():
                    primary_cfg_path.unlink()
            except Exception as e:
                print(f"  ⚠️  Warning: Could not delete temporary config file {primary_cfg_path.name}: {e}")
        
        # Record error if one occurred (only if not already logged above)
        if error_message:
            errors.append({
                "mcap": mcap.name,
                "error": f"[{types_str}] {error_message}"
            })
        
        # Run ros2 unbag with additional configs (only if overlaps exist)
        for idx, (additional_cfg, additional_types) in enumerate(additional_configs):
            # Check if this additional config MCAP is already completed
            if SKIP_COMPLETED:
                all_additional_completed = True
                for ext_type in additional_types:
                    if not is_mcap_completed(rosbag_name, f"{mcap_identifier}_additional_{idx}", ext_type):
                        all_additional_completed = False
                        break
                if all_additional_completed:
                    continue
            if _interrupted:
                errors.append({
                    "mcap": mcap.name,
                    "error": "Processing interrupted by user (during additional config)"
                })
                break
            
            additional_cfg_path = write_config(additional_cfg, rosbag_name, f"{mcap_identifier}_additional_{idx}")
            types_str = ", ".join(additional_types)
            #print(f"    🔄 Running ros2 unbag for: {types_str} (additional)")
            
            error_message = None
            additional_topics = list(additional_cfg.keys())
            additional_topics.remove("__global__")  # Remove global settings
            
            try:
                rc, error_msg = run_ros2_unbag(mcap, additional_cfg_path, additional_types)
                additional_mcap_id = f"{mcap_identifier}_additional_{idx}"
                if rc == 0:
                    # Log successful completion for each extraction type
                    for ext_type in additional_types:
                        save_completed_mcap(rosbag_name, additional_mcap_id, ext_type, topics=additional_topics)
                else:
                    failures += 1
                    error_message = error_msg or f"ros2 unbag failed with exit code {rc}"
                    print(f"    ❌ Failed for {types_str} (additional)")
                    # Log error for each extraction type
                    for ext_type in additional_types:
                        save_completed_mcap(rosbag_name, additional_mcap_id, ext_type, topics=additional_topics, error=error_message)

            except Exception as e:
                failures += 1
                error_message = f"Exception during processing: {type(e).__name__}: {str(e)}"
                print(f"    ❌ Failed for {types_str} (additional): {error_message}")
                # Log error for each extraction type
                additional_mcap_id = f"{mcap_identifier}_additional_{idx}"
                for ext_type in additional_types:
                    save_completed_mcap(rosbag_name, additional_mcap_id, ext_type, topics=additional_topics, error=error_message)
            finally:
                # Clean up additional config file
                try:
                    if additional_cfg_path.exists():
                        additional_cfg_path.unlink()
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not delete temporary config file {additional_cfg_path.name}: {e}")
            
            # Record error if one occurred
            if error_message:
                errors.append({
                    "mcap": mcap.name,
                    "error": f"[{types_str}] {error_message}"
                })

    # Save completion status for all extraction types (with errors if any occurred)
    if failures:
        print(f"  ❌ Done with {failures} failure(s)")
    else:
        print(f"  ✅ Completed all extraction types")
    
    for extraction_type in all_topics_by_type.keys():
        save_completed_rosbag(rosbag_name, extraction_type, errors if errors else None)


def main():
    global _interrupted
    
    # Register signal handlers for proper cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if shutil.which("ros2") is None:
            raise SystemExit("ros2 not found in PATH")
        
        if ROSBAGS is None:
            raise SystemExit("ROSBAGS environment variable not set")

        if MODE not in ("single", "all", "multiple"):
            raise SystemExit("MODE must be 'single', 'all', or 'multiple'")
        
        # Validate extraction types
        valid_types = {"image", "position", "video", "pointcloud", "odom"}
        if not EXTRACTION_TYPES:
            raise SystemExit("EXTRACTION_TYPES cannot be empty")
        for ext_type in EXTRACTION_TYPES:
            if ext_type not in valid_types:
                raise SystemExit(f"Invalid extraction type: {ext_type}. Must be one of: {valid_types}")
            if ext_type not in EXTRACTION_TYPE_CONFIG:
                raise SystemExit(f"Missing configuration for extraction type: {ext_type}")
            if ext_type not in TOPIC_SEARCH_TERMS:
                raise SystemExit(f"Missing topic search terms for extraction type: {ext_type}")
        
        if VERBOSITY not in ("quiet", "verbose"):
            raise SystemExit("VERBOSITY must be 'quiet' or 'verbose'")
        
        print(f"🔧 EXTRACTION_TYPES: {EXTRACTION_TYPES}")
        print(f"🔇 VERBOSITY: {VERBOSITY}")
        if EXCLUDED_TOPICS:
            print(f"🚫 EXCLUDED_TOPICS: {EXCLUDED_TOPICS}")

        # Show completion status per extraction type
        if SKIP_COMPLETED:
            print(f"⏭️  SKIP_COMPLETED: {SKIP_COMPLETED} (will skip completed rosbags)")
            for ext_type in EXTRACTION_TYPES:
                completed = load_completion(ext_type)
                if completed:
                    print(f"   📋 {ext_type}: {len(completed)} previously completed rosbag(s)")
                else:
                    print(f"   📋 {ext_type}: No previously completed rosbags")
        else:
            print(f"🔄 SKIP_COMPLETED: {SKIP_COMPLETED} (will reprocess all rosbags)")

        if MODE == "single":
            bag_dir = ROSBAGS / SINGLE_BAG_NAME
            if not (bag_dir / "metadata.yaml").exists():
                raise SystemExit(f"metadata.yaml not found in {bag_dir}")
            # Skip rosbags in EXCLUDED folders
            if "EXCLUDED" in str(bag_dir):
                print(f"⏭️  Skipping {SINGLE_BAG_NAME}: rosbag is in EXCLUDED folder")
                return
            print(f"▶️ MODE = single → Processing only: {SINGLE_BAG_NAME}")
            process_bag(bag_dir)

        elif MODE == "multiple":
            # Handle both dict and list formats for MULTIPLE_BAG_NAMES
            if isinstance(MULTIPLE_BAG_NAMES, dict):
                # Dict format: {"rosbag_name": [mcap_numbers]}
                bag_items = list(MULTIPLE_BAG_NAMES.items())
                print(f"▶️ MODE = multiple → Processing {len(bag_items)} specific bags (with mcap filtering):")
                for bag_name, mcap_numbers in bag_items:
                    if mcap_numbers:
                        print(f"   - {bag_name}: mcaps {mcap_numbers}")
                    else:
                        print(f"   - {bag_name}: all mcaps")
            else:
                # List format: ["rosbag_name1", "rosbag_name2", ...]
                bag_items = [(bag_name, None) for bag_name in MULTIPLE_BAG_NAMES]
                print(f"▶️ MODE = multiple → Processing {len(bag_items)} specific bags:")
                for bag_name, _ in bag_items:
                    print(f"   - {bag_name}")
            
            processed_count = 0
            skipped_count = 0
            
            for bag_name, mcap_filter in bag_items:
                if _interrupted:
                    break
                bag_dir = ROSBAGS / bag_name
                if not (bag_dir / "metadata.yaml").exists():
                    print(f"⚠️  Skipping {bag_name}: metadata.yaml not found in {bag_dir}")
                    skipped_count += 1
                    continue
                
                # Skip rosbags in EXCLUDED folders
                if "EXCLUDED" in str(bag_dir):
                    print(f"⏭️  Skipping {bag_name}: rosbag is in EXCLUDED folder")
                    skipped_count += 1
                    continue
                
                process_bag(bag_dir, mcap_filter=mcap_filter)
                processed_count += 1
            
            print(f"\n📊 Summary: {processed_count} processed, {skipped_count} skipped")

        else:  # MODE == "all"
            print("▶️ MODE = all → Processing ALL bags in ROSBAGS")
            for bag_dir in find_rosbag_dirs(ROSBAGS):
                if _interrupted:
                    break
                process_bag(bag_dir)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  KeyboardInterrupt received in main()")
        cleanup_process()
        _interrupted = True
        print("🛑 Script terminated by user")
        sys.exit(130)
    except Exception as e:
        cleanup_process()
        print(f"\n❌ Fatal error: {e}")
        raise
    finally:
        # Final cleanup
        cleanup_process()


if __name__ == "__main__":
    main()