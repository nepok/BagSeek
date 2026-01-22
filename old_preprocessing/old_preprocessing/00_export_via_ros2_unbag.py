#!/usr/bin/env python3
"""
ROS2 Unbag Extraction Script

This script extracts data from ROS2 MCAP files using ros2_unbag.

Modes:
- "single": Process one specific rosbag (set SINGLE_BAG_NAME)
- "all": Process all rosbags found in BAGS_ROOT directory
- "multiple": Process specific rosbags listed in MULTIPLE_BAG_NAMES array

Extraction Types:
- "image": Extract images using dynamic config generation
- "gps": Extract GPS data using predefined config file

Verbosity Modes:
- "quiet": Only show success/failure messages
- "verbose": Show full ros2 unbag progress bars and details

Topic Filtering:
- EXCLUDED_TOPICS: List of topic patterns to exclude from processing
- Case-insensitive partial matching (e.g., "odom" excludes "/novatel/oem7/odom")

Completion Tracking:
- Tracks completed rosbags in JSON file to avoid reprocessing
- Verifies completion by checking output directories
- SKIP_COMPLETED: Set to True to skip completed rosbags, False to reprocess all

Usage:
1. Edit the CONFIG section below
2. Run: python3 export_images.py
"""
import json
import re
import shutil
import subprocess
import signal
import sys
from pathlib import Path
from typing import Iterable, Set, Union, Optional

# =========================
# CONFIG (edit these)
# =========================

# "single"   → only process ONE rosbag
# "all"      → process ALL rosbags in BAGS_ROOT
# "multiple" → process SPECIFIC rosbags from MULTIPLE_BAG_NAMES list
MODE = "all"            # "single", "all", or "multiple"
# "image" → extract images using dynamic config generation
# "gps"   → -xtract GPS data using predefined config file
EXTRACTION_TYPE = "image"  # "image" or "gps"


SINGLE_BAG_NAME = "rosbag2_2025_07_24-16_01_22"   # only used when MODE="single"

# Only used when MODE="multiple" - list of rosbag basenames to process
"""MULTIPLE_BAG_NAMES = [
    "rosbag2_2025_07_24-16_01_22",
    "rosbag2_2025_07_29-08_08_48",
    "rosbag2_2025_07_30-10_33_53",
    "rosbag2_2025_08_04-15_00_07",
    "rosbag2_2025_08_05-11_01_40",
    "rosbag2_2025_08_19-07_54_26",
    "rosbag2_2025_08_22-11_42_48",
    "rosbag2_2025_08_22-12_27_43",
    "rosbag2_2025_08_26-07_27_19",
    "rosbag2_2025_09_11-12_59_00",
    "rosbag2_2025_09_25-08_10_26"
    # Add more rosbag basenames here as needed
]"""
MULTIPLE_BAG_NAMES = [
    "rosbag2_2025_08_08-13_59_06",
    "rosbag2_2025_08_11-19_20_47",
    "rosbag2_2025_08_20-08_59_36",
    "rosbag2_2025_08_30-15_19_54",
    "rosbag2_2025_09_09-10_43_40",
    "rosbag2_2025_09_10-09_15_13",
    "rosbag2_2025_09_24-12_52_43",
    "rosbag2_2025_09_24-13_23_22",
    "rosbag2_2025_08_06_06_40_54",
    "rosbag2_2025_08_26-07_27_19",
    "rosbag2_2025_08_29-11_00_49",
    "rosbag2_2025_09_04-15_01_01",
    "rosbag2_2025_09_04-15_13_38",
    "rosbag2_2025_09_09-10_21_16",
    "rosbag2_2025_09_10-15_03_01",
    "rosbag2_2025_09_15-07_34_08",
    "rosbag2_2025_09_24-14_30_42",
    "rosbag2_2025_09_25-08_10_26",
    "rosbag2_2025_09_25-08_26_44",
    "rosbag2_2025_09_26-12_49_12"
]

BAGS_ROOT   = Path("/home/nepomuk/sflnas/DataReadOnly334/tractor_data/autorecord")

if EXTRACTION_TYPE == "image":
    OUTPUT_DIR = Path("/home/nepomuk/sflnas/DataReadWrite334/0_shared/Feldschwarm/bagseek/src/extracted_images_per_topic")
elif EXTRACTION_TYPE == "gps":
    OUTPUT_DIR = Path("/home/nepomuk/sflnas/DataReadWrite334/0_shared/Feldschwarm/bagseek/src/positional_data")
else:
    raise SystemExit("EXTRACTION_TYPE must be 'image' or 'gps'")
    
TMP_DIR     = Path("/mnt/data/tmp/ros2unbag_configs")
CPU_PERCENT = 25.0       # __global__.cpu_percentage

# Output verbosity control
# "quiet"   → only show success/failure messages
# "verbose" → show full ros2 unbag progress bars and details
VERBOSITY = "quiet"  # "quiet" or "verbose"

# Topics to exclude from processing (case-insensitive partial matching)
# Examples: ["odom", "imu", "camera_info"] - will exclude any topic containing these terms
EXCLUDED_TOPICS = ["odom"]  # List of topic names/patterns to exclude

# Completion tracking
if EXTRACTION_TYPE == "gps":
    COMPLETION_FILE = Path("/home/nepomuk/sflnas/DataReadWrite334/0_shared/Feldschwarm/bagseek/src/positional_data/completion.json")
else:
    COMPLETION_FILE = Path("/home/nepomuk/sflnas/DataReadWrite334/0_shared/Feldschwarm/bagseek/src/extracted_images_per_topic_header_timestamp/completion.json")
SKIP_COMPLETED = True  # Set to False to reprocess completed rosbags

# =========================
# Global process tracking for cleanup
# =========================
_active_process: Optional[subprocess.Popen] = None
_interrupted = False


def find_rosbag_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "metadata.yaml").exists():
            yield p


def load_completed_rosbags() -> Set[str]:
    """Load the set of completed rosbag names from the completion file."""
    if not COMPLETION_FILE.exists():
        return set()
    
    try:
        with open(COMPLETION_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('completed', []))
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return set()


def save_completed_rosbag(rosbag_name: str):
    """Mark a rosbag as completed and save to the completion file."""
    completed = load_completed_rosbags()
    completed.add(rosbag_name)
    
    # Ensure directory exists
    COMPLETION_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    data = {
        'completed': sorted(list(completed)),
    }
    
    with open(COMPLETION_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def is_rosbag_completed(rosbag_name: str) -> bool:
    """Check if a rosbag has been completed."""
    if not SKIP_COMPLETED:
        return False
    
    completed = load_completed_rosbags()
    return rosbag_name in completed


def verify_completion(rosbag_name: str) -> bool:
    """Verify that a rosbag was actually completed by checking output directories."""
    if EXTRACTION_TYPE == "gps":
        # Check if both bestpos and fix directories exist and have files
        bestpos_dir = OUTPUT_DIR / "bestpos" / rosbag_name
        fix_dir = OUTPUT_DIR / "fix" / rosbag_name
        
        bestpos_has_files = bestpos_dir.exists() and any(bestpos_dir.iterdir())
        fix_has_files = fix_dir.exists() and any(fix_dir.iterdir())
        
        return bestpos_has_files and fix_has_files
    else:
        # For images, check if the main output directory exists and has files
        output_dir = OUTPUT_DIR / rosbag_name
        return output_dir.exists() and any(output_dir.iterdir())


def parse_topics_from_metadata(meta_path: Path, search_term: str) -> Set[str]:
    """
    Parse ROS2 bag metadata.yaml and return topic names that contain the specified search term (case-insensitive).
    Works with common rosbag2 metadata schemas (including MCAP).
    
    Args:
        meta_path: Path to the metadata.yaml file
        search_term: Term to search for in topic names (e.g., "image", "novatel", "gps")
    
    Returns:
        Set of topic names containing the search term (excluding topics in EXCLUDED_TOPICS)
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
                    if isinstance(name, str) and search_term in name.lower():
                        topics.add(name)
                # generic fields
                for k, v in node.items():
                    if k in ("name", "topic") and isinstance(v, str) and v.startswith("/"):
                        if search_term in v.lower():
                            topics.add(v)
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
            if search_term in ln.lower():
                for t in re.findall(r"(/[\w/]+)", ln):
                    if search_term in t.lower():
                        topics.add(t)

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

def build_config(topics: Set[str], rosbag_name: str, mcap_identifier: Optional[str] = None) -> dict:

    if EXTRACTION_TYPE == "gps":
        cfg = {}
        for topic in sorted(topics):
            # Extract the last part of the topic path (e.g., "bestpos" from "/novatel/oem7/bestpos")
            topic_parts = topic.strip('/').split('/')
            topic_name = topic_parts[-1] if topic_parts else topic.strip('/')
            
            # Create path with topic subdirectory
            topic_out = OUTPUT_DIR / topic_name
            
            cfg[topic] = {
                "format": "table/csv@single_file",
                "path": str(topic_out),
                "subfolder": str(rosbag_name),
                "naming": str(rosbag_name + "_" + mcap_identifier),
            }
    else:
        # For images, use the original logic
        bag_out = OUTPUT_DIR / rosbag_name / mcap_identifier
        cfg = {
            t: {
                "format": "image/png",
                "path": str(bag_out),
                "subfolder": "%name",
                "naming": "%timestamp",
            }
            for t in sorted(topics)
        }
    cfg["__global__"] = {"cpu_percentage": float(CPU_PERCENT)}
    return cfg


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


def run_ros2_unbag(mcap_path: Path, config_path: Path) -> int:
    """Run ros2 unbag with proper interrupt handling."""
    global _active_process, _interrupted
    
    if _interrupted:
        return 1
    
    cmd = ["ros2", "unbag", str(mcap_path), "--config", str(config_path)]
    
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
        
        if returncode != 0:
            print(f"❌ {mcap_path.name} - FAILED (exit code: {returncode})")
            if VERBOSITY == "quiet" and stdout:
                # Show error output in quiet mode
                print(f"   Error: {stdout}")
        else:
            print(f"✅ {mcap_path.name} - SUCCESS")
        
        return returncode
        
    except KeyboardInterrupt:
        print("\n⚠️  KeyboardInterrupt in run_ros2_unbag")
        cleanup_process()
        _interrupted = True
        raise
    except Exception as e:
        cleanup_process()
        print(f"❌ Error running ros2 unbag: {e}")
        return 1


def process_bag(bag_dir: Path):
    global _interrupted
    
    if _interrupted:
        return
    
    rosbag_name = bag_dir.name
    meta = bag_dir / "metadata.yaml"
    
    # Check if already completed
    if is_rosbag_completed(rosbag_name):
        if verify_completion(rosbag_name):
            print(f"\n⏭️  {rosbag_name}: Already completed (skipping)")
            return
        else:
            print(f"\n⚠️  {rosbag_name}: Marked as completed but verification failed (reprocessing)")
    
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
    
    mcaps = sorted(bag_dir.glob("*.mcap"), key=extract_number)

    print(f"\n📦 {rosbag_name}: {len(mcaps)} mcap file(s)")

    if not mcaps:
        print("   (skip: no .mcap files)")
        return

    if EXTRACTION_TYPE == "gps":
        # Parse GPS topics and build dynamic config
        search_term = "novatel"
        topics = parse_topics_from_metadata(meta, search_term)
        if not topics:
            print(f"   (skip: no '{search_term}' topics found)")
            return

        print(f"   📝 topics: {len(topics)}")
        
    else:  # EXTRACTION_TYPE == "image"
        # Parse image topics and build dynamic config
        search_term = "image"
        topics = parse_topics_from_metadata(meta, search_term)
        if not topics:
            print(f"   (skip: no '{search_term}' topics found)")
            return

        print(f"   📝 topics: {len(topics)}")

    # Create output directories
    if EXTRACTION_TYPE == "gps":
        # Create topic subdirectories for GPS data
        for topic in topics:
            topic_parts = topic.strip('/').split('/')
            topic_name = topic_parts[-1] if topic_parts else topic.strip('/')
            topic_dir = OUTPUT_DIR / topic_name / rosbag_name
            topic_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Create single output directory for images
        (OUTPUT_DIR / rosbag_name).mkdir(parents=True, exist_ok=True)

    failures = 0
    for mcap in mcaps:
        if _interrupted:
            print(f"\n⚠️  Processing interrupted, stopping {rosbag_name}")
            break
        
        # Extract MCAP identifier for unique naming
        mcap_identifier = extract_mcap_identifier(mcap)
        
        # Generate unique config for each MCAP file
        cfg = build_config(topics, rosbag_name, mcap_identifier)
        cfg_path = write_config(cfg, rosbag_name, mcap_identifier)
        
        #if EXTRACTION_TYPE == "gps":
        #    print(f"   🔄 Processing {mcap.name} → config: {cfg_path.name}")
        
        try:
            rc = run_ros2_unbag(mcap, cfg_path)
            failures += (rc != 0)
        finally:
            # Clean up temporary config file
            try:
                if cfg_path.exists():
                    cfg_path.unlink()
            except Exception as e:
                print(f"⚠️  Warning: Could not delete temporary config file {cfg_path.name}: {e}")

    if failures:
        print(f"   ❌ Done with {failures} failure(s) in {rosbag_name}")
    else:
        print(f"   ✅ Completed {rosbag_name}")
        # Mark as completed
        save_completed_rosbag(rosbag_name)


def main():
    global _interrupted
    
    # Register signal handlers for proper cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if shutil.which("ros2") is None:
            raise SystemExit("ros2 not found in PATH")

        if MODE not in ("single", "all", "multiple"):
            raise SystemExit("MODE must be 'single', 'all', or 'multiple'")
        
        if EXTRACTION_TYPE not in ("image", "gps"):
            raise SystemExit("EXTRACTION_TYPE must be 'image' or 'gps'")
        
        if VERBOSITY not in ("quiet", "verbose"):
            raise SystemExit("VERBOSITY must be 'quiet' or 'verbose'")
        
        print(f"🔧 EXTRACTION_TYPE: {EXTRACTION_TYPE}")
        print(f"🔇 VERBOSITY: {VERBOSITY}")
        if EXCLUDED_TOPICS:
            print(f"🚫 EXCLUDED_TOPICS: {EXCLUDED_TOPICS}")

        # Show completion status
        completed = load_completed_rosbags()
        if completed:
            print(f"📋 Previously completed: {len(completed)} rosbag(s)")
            if SKIP_COMPLETED:
                print(f"⏭️  SKIP_COMPLETED: {SKIP_COMPLETED} (will skip completed rosbags)")
            else:
                print(f"🔄 SKIP_COMPLETED: {SKIP_COMPLETED} (will reprocess all rosbags)")
        else:
            print("📋 No previously completed rosbags found")

        if MODE == "single":
            bag_dir = BAGS_ROOT / SINGLE_BAG_NAME
            if not (bag_dir / "metadata.yaml").exists():
                raise SystemExit(f"metadata.yaml not found in {bag_dir}")
            print(f"▶️ MODE=single → Processing only: {SINGLE_BAG_NAME}")
            process_bag(bag_dir)

        elif MODE == "multiple":
            print(f"▶️ MODE=multiple → Processing {len(MULTIPLE_BAG_NAMES)} specific bags:")
            for bag_name in MULTIPLE_BAG_NAMES:
                print(f"   - {bag_name}")
            
            processed_count = 0
            skipped_count = 0
            
            for bag_name in MULTIPLE_BAG_NAMES:
                if _interrupted:
                    break
                bag_dir = BAGS_ROOT / bag_name
                if not (bag_dir / "metadata.yaml").exists():
                    print(f"⚠️  Skipping {bag_name}: metadata.yaml not found in {bag_dir}")
                    skipped_count += 1
                    continue
                
                process_bag(bag_dir)
                processed_count += 1
            
            print(f"\n📊 Summary: {processed_count} processed, {skipped_count} skipped")

        else:  # MODE == "all"
            print("▶️ MODE=all → Processing ALL bags in BAGS_ROOT")
            for bag_dir in find_rosbag_dirs(BAGS_ROOT):
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