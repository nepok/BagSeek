#!/usr/bin/env python3
"""
ROS2 Alignment and Metadata Generation Script

This script generates alignment CSV files and topic metadata JSON files from ROS2 MCAP rosbags.
It extracts timestamps from message headers, determines a reference topic (highest frequency),
creates a refined timeline, and aligns all topic timestamps to this reference.

The output CSV files contain aligned timestamps for all topics, allowing synchronization
across topics with different frequencies. The JSON files contain topic names and types.

Usage:
1. Set environment variables: ROSBAGS, LOOKUP_TABLES, TOPICS
2. Run: python3 02_generate_alignment_and_metadata.py
"""

import csv
import json
import os
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from multiprocessing import Manager, RLock
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import yaml
from dotenv import load_dotenv
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# =========================
# Load environment variables
# =========================

PARENT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

ROSBAGS = Path(os.getenv("ROSBAGS"))

BASE_STR = os.getenv("BASE")
LOOKUP_TABLES_STR = os.getenv("LOOKUP_TABLES")
TOPICS_STR = os.getenv("TOPICS")    

LOOKUP_TABLES = Path(BASE_STR + LOOKUP_TABLES_STR)
TOPICS = Path(BASE_STR + TOPICS_STR)

# =========================
# Configuration constants
# =========================

SKIP: List[str] = []
MAX_WORKERS: int = 8  # Workers for rosbag-level parallelism
MCAP_MAX_WORKERS: int = 8  # Workers for MCAP-level parallelism within a rosbag
WARNED_TOPICS: Set[str] = set()

# =========================
# Rosbag selection mode
# =========================

MODE = "all"  # "single", "all", or "multiple"
SINGLE_BAG_NAME = "rosbag2_2025_07_28-12_45_50"  # Only used when MODE = "single"
MULTIPLE_BAG_NAMES = []  # List of rosbag names, only used when MODE = "multiple"
# Example: MULTIPLE_BAG_NAMES = ["rosbag2_2025_07_23-12_58_03", "rosbag2_2025_07_28-07_29_07"]

# =========================
# Global process tracking for cleanup
# =========================

executor: Optional[ProcessPoolExecutor] = None
manager: Optional[Manager] = None
stop_event: Optional[object] = None


# =========================
# Signal handling
# =========================

def signal_handler(signum, frame) -> None:
    """Handle keyboard interrupt (Ctrl+C) and termination signals gracefully.
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
    
    # Signal all workers to stop
    if stop_event:
        stop_event.set()
    
    if executor:
        print("📋 Terminating all worker processes...")
        try:
            # Get all processes - they're stored in _processes dict
            processes = list(executor._processes.values())
            
            # Terminate all worker processes
            for process in processes:
                try:
                    if process.is_alive():
                        process.terminate()
                except Exception:
                    pass
            
            # Wait a bit for graceful termination
            time.sleep(2)
            
            # Force kill any remaining processes
            for process in processes:
                try:
                    if process.is_alive():
                        process.kill()
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️  Error terminating processes: {e}")
        
        print("📋 Shutting down executor...")
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass
    
    print("✅ Shutdown complete.")
    sys.exit(130)  # Standard exit code for SIGINT


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =========================
# Helper functions
# =========================

def extract_mcap_number(mcap_path: Path) -> int:
    """Extract the number from MCAP filename for proper numerical sorting.
    
    Args:
        mcap_path: Path to the MCAP file
        
    Returns:
        Numeric identifier from filename, or 0 if not found
    """
    name = mcap_path.stem  # Get filename without extension
    # Extract the last part after the last underscore (e.g., "100" from "rosbag2_2025_07_23-12_58_03_100")
    parts = name.split('_')
    if parts and parts[-1].isdigit():
        return int(parts[-1])
    return 0  # Fallback for files without numeric suffix


def find_mcap_files(rosbag_path: Path) -> List[Path]:
    """Find all MCAP files in a rosbag directory, sorted numerically.
    
    Args:
        rosbag_path: Path to the rosbag directory
        
    Returns:
        List of MCAP file paths, sorted by numeric identifier
    """
    mcap_files = sorted(rosbag_path.glob("*.mcap"), key=extract_mcap_number)
    return mcap_files


def get_total_message_count(rosbag_path: Path) -> Optional[int]:
    """Extract total message count from metadata.yaml for accurate progress tracking.
    
    Args:
        rosbag_path: Path to the rosbag directory
        
    Returns:
        Total number of messages across all topics, or None if metadata cannot be read
    """
    metadata_path = rosbag_path / "metadata.yaml"
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        # Sum all message counts across all topics
        bag_info = metadata.get("rosbag2_bagfile_information", {})
        topics = bag_info.get("topics_with_message_count", [])
        
        total_messages = sum(
            int(topic.get("message_count", 0)) 
            for topic in topics
        )
        return total_messages
    except Exception:
        return None


def get_all_topics_from_metadata(rosbag_path: Path) -> Optional[List[str]]:
    """Extract all topic names from metadata.yaml file.
    
    Args:
        rosbag_path: Path to the rosbag directory containing metadata.yaml
        
    Returns:
        Sorted list of topic names, or None if metadata.yaml cannot be read
    """
    metadata_path = rosbag_path / "metadata.yaml"
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        # Navigate to topics_with_message_count
        bag_info = metadata.get("rosbag2_bagfile_information", {})
        topics_with_counts = bag_info.get("topics_with_message_count", [])
        
        # Extract topic names from topic_metadata.name
        topic_names = []
        for topic_entry in topics_with_counts:
            topic_metadata = topic_entry.get("topic_metadata", {})
            topic_name = topic_metadata.get("name")
            if topic_name:
                topic_names.append(topic_name)
        
        # Return sorted list of unique topic names
        return sorted(list(set(topic_names))) if topic_names else None
    except Exception:
        return None


def get_all_topics_from_mcap(rosbag_path: Path) -> Optional[List[str]]:
    """Extract all topic names from first MCAP file's channels as fallback.
    
    Used when metadata.yaml doesn't exist.
    
    Args:
        rosbag_path: Path to the rosbag directory
        
    Returns:
        Sorted list of topic names, or None if MCAP files cannot be read
    """
    mcap_files = find_mcap_files(rosbag_path)
    if not mcap_files:
        return None
    
    try:
        with open(mcap_files[0], "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            summary = reader.get_summary()
            channels = summary.channels if summary else {}
            
            topic_names = [channel.topic for channel in channels.values()]
            return sorted(list(set(topic_names))) if topic_names else None
    except Exception:
        return None


def determine_reference_topic(topic_timestamps: Dict[str, List[int]]) -> str:
    """Determine the topic with the most messages to use as the reference timeline.
    
    Args:
        topic_timestamps: Dictionary mapping topic names to lists of timestamps
        
    Returns:
        Name of the topic with the highest message count
    """
    return max(topic_timestamps.items(), key=lambda x: len(x[1]))[0]


def create_reference_timestamps(timestamps: List[int], factor: int = 2) -> np.ndarray:
    """Create a refined reference timeline with smaller intervals to improve alignment accuracy.
    
    The factor reduces the mean interval to create a denser timeline for better matching.
    
    Args:
        timestamps: List of timestamps from the reference topic
        factor: Factor to reduce the mean interval (default: 2)
        
    Returns:
        NumPy array of refined reference timestamps
    """
    timestamps = sorted(set(timestamps))
    if len(timestamps) < 2:
        return np.array(timestamps, dtype=np.int64)
    
    diffs = np.diff(timestamps)
    mean_interval = np.mean(diffs)
    refined_interval = mean_interval / factor
    ref_start = timestamps[0]
    ref_end = timestamps[-1]
    return np.arange(ref_start, ref_end, refined_interval).astype(np.int64)


def align_topic_to_reference(
    topic_ts: List[int], 
    ref_ts: np.ndarray, 
    max_diff: int = int(1e8)
) -> List[Optional[int]]:
    """Align timestamps of a topic to the closest reference timestamps within max_diff.
    
    For each reference timestamp, find the closest topic timestamp to align messages across topics.
    
    Args:
        topic_ts: List of timestamps from the topic to align
        ref_ts: Array of reference timestamps
        max_diff: Maximum allowed difference in nanoseconds (default: 100ms)
        
    Returns:
        List of aligned timestamps (None if no match within max_diff)
    """
    aligned = []
    topic_idx = 0
    for rt in ref_ts:
        while (topic_idx + 1 < len(topic_ts) and 
               abs(topic_ts[topic_idx + 1] - rt) < abs(topic_ts[topic_idx] - rt)):
            topic_idx += 1
        closest = topic_ts[topic_idx]
        if abs(closest - rt) <= max_diff:
            aligned.append(closest)
        else:
            aligned.append(None)
    return aligned


def extract_timestamp(
    topic: str, 
    ros2_msg: object, 
    log_time: int,
    topic_type_map: Dict[str, str], 
    process_id: int
) -> Tuple[int, str]:
    """Extract timestamp from message log_time.
    
    Always uses log_time from MCAP, which is the time when the message was recorded.
    
    Args:
        topic: Topic name
        ros2_msg: Decoded ROS2 message object (unused, kept for compatibility)
        log_time: Log time from MCAP in nanoseconds
        topic_type_map: Dictionary mapping topic names to message types
        process_id: Process ID for logging (unused, kept for compatibility)
        
    Returns:
        Tuple of (timestamp in nanoseconds, message type)
    """
    topic_type = topic_type_map.get(topic)
    return log_time, topic_type


# =========================
# Main processing functions
# =========================

def process_single_mcap(mcap_path: Path) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    """Process a single MCAP file and extract timestamps and topic types.
    
    Args:
        mcap_path: Path to the MCAP file
        
    Returns:
        Tuple of (topic_data: Dict[str, List[int]], topic_types: Dict[str, str])
        topic_data maps topic names to lists of timestamps
        Returns empty dicts if processing fails or is interrupted
    """
    topic_data: Dict[str, List[int]] = defaultdict(list)
    topic_types: Dict[str, str] = {}
    
    process_id = os.getpid()
    mcap_name = mcap_path.name
    
    try:
        # Read all messages from this MCAP file using make_reader
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            
            # Get topic types from MCAP summary
            topic_type_map: Dict[str, str] = {}
            try:
                summary = reader.get_summary()
                channels = summary.channels if summary else {}
                schemas = summary.schemas if summary else {}
                
                # Build topic_type_map: channel.topic -> schema.name
                for channel_id, channel in channels.items():
                    schema_id = channel.schema_id
                    if schema_id in schemas:
                        schema = schemas[schema_id]
                        topic_type_map[channel.topic] = schema.name
            except Exception as exc:
                tqdm.write(f"CPU-{process_id}: Warning: Could not get topic types from {mcap_name}: {exc}")
            
            for schema, channel, message, ros2_msg in reader.iter_decoded_messages(
                log_time_order=True,
                reverse=False
            ):
                try:
                    topic = channel.topic
                    log_time = message.log_time
                    
                    timestamp, resolved_type = extract_timestamp(topic, ros2_msg, log_time, topic_type_map, process_id)
                    topic_data[topic].append(timestamp)
                    topic_types[topic] = resolved_type
                except (ValueError, AttributeError, Exception) as e:
                    # Skip messages without headers or with errors
                    continue
    except KeyboardInterrupt:
        raise
    except Exception as e:
        tqdm.write(f"CPU-{process_id}: Error reading MCAP {mcap_name}: {e}")
        # Return empty dicts on error - other MCAPs can still be processed
        return {}, {}
    
    return topic_data, topic_types


def create_alignment_csv(
    topic_data: Dict[str, List[int]],
    topic_types: Dict[str, str],
    csv_path: Path,
    process_id: int,
    all_topics: Optional[List[str]] = None,
    status_bar: Optional[tqdm] = None
) -> None:
    """Create alignment CSV from topic data.
    
    Args:
        topic_data: Dictionary mapping topic names to lists of timestamps
        topic_types: Dictionary mapping topic names to their types
        csv_path: Path where the alignment CSV will be written
        process_id: Process ID for logging
        all_topics: Optional complete list of all topics (for consistent CSV columns)
        status_bar: Optional tqdm progress bar for status updates
    """
    if not topic_data and not all_topics:
        return
    
    # Sort timestamps for each topic
    if status_bar:
        status_bar.set_description_str(f"CPU-{process_id}: Sorting timestamps...")
    for topic in topic_data:
        topic_data[topic] = sorted(topic_data[topic])

    # Determine reference topic by message count
    if status_bar:
        status_bar.set_description_str(f"CPU-{process_id}: Aligning topics...")
    
    if topic_data:
        ref_topic = determine_reference_topic(topic_data)
        ref_ts = create_reference_timestamps(topic_data[ref_topic])
    elif all_topics:
        # No topics in this MCAP, but we still need to create CSV with all topics as None
        # Use a dummy reference timeline (empty, will create empty CSV with just header)
        ref_ts = np.array([], dtype=np.int64)
    else:
        # No topics and no all_topics list, nothing to do
        return

    # Align each topic to the reference timeline to synchronize timestamps across topics
    aligned_data: Dict[str, List[Optional[int]]] = {}
    for topic, timestamps in topic_data.items():
        # Align timestamps
        aligned_data[topic] = align_topic_to_reference(timestamps, ref_ts)
    
    # If all_topics is provided, ensure all topics are in aligned_data (with None values if missing)
    if all_topics:
        for topic in all_topics:
            if topic not in aligned_data:
                # Create empty aligned data (all None) for missing topics
                aligned_data[topic] = [None] * len(ref_ts)

    # Prepare CSV data
    if status_bar:
        status_bar.set_description_str(f"CPU-{process_id}: Writing CSV...")
    # The CSV rows contain the reference timestamp, aligned timestamps for each topic,
    # and the max timestamp distance across topics
    csv_data: List[List[Union[int, str]]] = []
    # Use all_topics if provided, otherwise use aligned_data.keys()
    if all_topics:
        topics = all_topics
    else:
        topics = list(aligned_data.keys())
    # Header: Reference Timestamp, then for each topic, then Max Distance
    header = ['Reference Timestamp'] + topics + ['Max Distance']
    
    for i, ref_time in enumerate(ref_ts):
        row: List[Union[int, str]] = [int(ref_time)]
        row_values: List[int] = []

        # Add timestamps for each topic
        # Note: topics list contains original topic names (with /), but header uses underscores
        for topic in topics:
            if topic in aligned_data:
                ts = aligned_data[topic][i]
                if ts is not None:
                    row.append(f"{ts}")
                    row_values.append(ts)
                else:
                    row.append("None")
            else:
                # Topic not in this MCAP, add None
                row.append("None")

        max_distance = max(row_values) - min(row_values) if len(row_values) > 1 else None
        row.append(f"{max_distance}" if max_distance is not None else "None")
        csv_data.append(row)

    # Write to CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        if csv_data:
            writer.writerows(csv_data)

    if status_bar:
        status_bar.set_description_str(f"CPU-{process_id}: CSV generated")


def process_single_mcap_with_alignment(
    mcap_path: Path,
    csv_path: Path,
    all_topics: Optional[List[str]] = None,
    bar_position: int = 0
) -> Tuple[Dict[str, str], bool]:
    """Process one MCAP file and generate its alignment CSV immediately.
    
    Args:
        mcap_path: Path to the MCAP file
        csv_path: Path where the alignment CSV will be written
        all_topics: Optional complete list of all topics (for consistent CSV columns)
        bar_position: Position for progress bar (unused, kept for compatibility)
        
    Returns:
        Tuple of (topic_types: Dict[str, str], success: bool)
    """
    process_id = os.getpid()
    mcap_name = mcap_path.name
    
    # Extract timestamps from MCAP
    topic_data: Dict[str, List[int]] = defaultdict(list)
    topic_types: Dict[str, str] = {}
    
    try:
        # Read all messages from this MCAP file using make_reader
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            
            # Get topic types from MCAP summary
            topic_type_map: Dict[str, str] = {}
            try:
                summary = reader.get_summary()
                channels = summary.channels if summary else {}
                schemas = summary.schemas if summary else {}
                
                # Build topic_type_map: channel.topic -> schema.name
                for channel_id, channel in channels.items():
                    schema_id = channel.schema_id
                    if schema_id in schemas:
                        schema = schemas[schema_id]
                        topic_type_map[channel.topic] = schema.name
            except Exception as exc:
                tqdm.write(f"CPU-{process_id}: Warning: Could not get topic types from {mcap_name}: {exc}")
            
            for schema, channel, message, ros2_msg in reader.iter_decoded_messages(
                log_time_order=True,
                reverse=False
            ):
                try:
                    topic = channel.topic
                    log_time = message.log_time
                    
                    timestamp, resolved_type = extract_timestamp(topic, ros2_msg, log_time, topic_type_map, process_id)
                    topic_data[topic].append(timestamp)
                    topic_types[topic] = resolved_type
                except (ValueError, AttributeError, Exception) as e:
                    # Skip messages without headers or with errors
                    continue
    except KeyboardInterrupt:
        raise
    except Exception as e:
        tqdm.write(f"CPU-{process_id}: Error reading MCAP {mcap_name}: {e}")
        return {}, False
    
    if not topic_data:
        return {}, False
    
    # Create alignment CSV
    try:
        create_alignment_csv(topic_data, topic_types, csv_path, process_id, all_topics=all_topics)
        return topic_types, True
    except Exception as e:
        tqdm.write(f"CPU-{process_id}: Error creating alignment CSV for {mcap_name}: {e}")
        return topic_types, False


def _process_mcap_with_alignment_wrapper(args: Tuple[Path, Path, Optional[List[str]]]) -> Tuple[Dict[str, str], bool]:
    """Wrapper function for process_map to process MCAP with alignment.
    
    Args:
        args: Tuple of (mcap_path, csv_path, all_topics)
        
    Returns:
        Tuple of (topic_types, success)
    """
    mcap_path, csv_path, all_topics = args
    return process_single_mcap_with_alignment(mcap_path, csv_path, all_topics)


def process_rosbag(
    rosbag_path: Path,
    csv_path: Path,
    topics_json_path: Optional[Path] = None,
    bar_position: int = 0,
    status_bar: Optional[tqdm] = None,
    shared_stop_event: Optional[object] = None
) -> None:
    """Process a single rosbag to generate alignment CSVs per MCAP and topics JSON.
    
    Processes all MCAP files in the rosbag directory in parallel, creating one
    alignment CSV per MCAP file and one topics JSON per rosbag.
    
    Args:
        rosbag_path: Path to the rosbag directory
        csv_path: Base path for CSV files (will create subdirectory with MCAP numbers)
        topics_json_path: Optional path where topics JSON will be written
        bar_position: Position for progress bar
        status_bar: Optional tqdm progress bar for status updates
        shared_stop_event: Optional event to signal workers to stop
    """
    # Check if we should stop
    if shared_stop_event and shared_stop_event.is_set():
        return
    
    rosbag_name = rosbag_path.name
    parent_name = rosbag_path.parent.name
    # Build display name: for multipart rosbags show parent/Part_N
    if parent_name.endswith("_multi_parts"):
        display_name = f"{parent_name}/{rosbag_name}"
    else:
        display_name = rosbag_name
    process_id = os.getpid()
    
    # Find all MCAP files in the rosbag directory
    mcap_files = find_mcap_files(rosbag_path)
    
    if not mcap_files:
        if status_bar:
            status_bar.set_description_str(f"CPU-{process_id}: No MCAP files found in {display_name}")
        return
    
    # Get all topics from metadata.yaml, with fallback to MCAP channels
    all_topics = get_all_topics_from_metadata(rosbag_path)
    if all_topics is None:
        # Fallback: read topics from first MCAP file's channels
        all_topics = get_all_topics_from_mcap(rosbag_path)
        if all_topics is None:
            # Final fallback: will use topics from processed MCAPs
            all_topics = None
            if status_bar:
                status_bar.set_description_str(f"CPU-{process_id}: Warning: Could not read topics from metadata.yaml or MCAP for {display_name}")
    
    # Generate CSV paths for each MCAP
    # CSV path structure: {csv_path.parent}/{rosbag_name}/{mcap_number}.csv
    # For multipart: {csv_path.parent}/{parent_name}/{part_name}/{mcap_number}.csv
    csv_base_dir = csv_path.parent
    if parent_name.endswith("_multi_parts"):
        # For multipart rosbags, use parent_name/part_name structure
        csv_mcap_dir = csv_base_dir / parent_name / rosbag_name
    else:
        # For regular rosbags, use rosbag_name
        csv_mcap_dir = csv_base_dir / rosbag_name
    
    csv_mcap_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare list of (mcap_path, csv_path, all_topics) tuples for processing
    mcap_csv_pairs: List[Tuple[Path, Path, Optional[List[str]]]] = []
    for mcap_file in mcap_files:
        mcap_number = extract_mcap_number(mcap_file)
        mcap_csv_path = csv_mcap_dir / f"{mcap_number}.csv"
        
        # Skip if CSV already exists
        if mcap_csv_path.exists():
            continue
        
        mcap_csv_pairs.append((mcap_file, mcap_csv_path, all_topics))
    
    if not mcap_csv_pairs:
        if status_bar:
            status_bar.set_description_str(f"CPU-{process_id}: All MCAP CSVs already exist for {display_name}")
        # Initialize topic_types for topics JSON collection
        topic_types: Dict[str, str] = {}
    else:
        if status_bar:
            status_bar.set_description_str(f"CPU-{process_id}: Processing {len(mcap_csv_pairs)} MCAP file(s) in parallel...")
        
        # Check if we should stop before starting MCAP processing
        if shared_stop_event and shared_stop_event.is_set():
            return
        
        # Limit workers to avoid overwhelming the system with nested executors
        # (main executor already uses MAX_WORKERS for rosbag-level parallelism)
        mcap_workers = min(MCAP_MAX_WORKERS, len(mcap_csv_pairs))
        
        # Collect topic types from all MCAPs for topics JSON
        topic_types: Dict[str, str] = {}
        processed_results = None
        
        try:
            # Use process_map to automatically combine progress from all MCAP workers
            processed_results = process_map(
                _process_mcap_with_alignment_wrapper,
                mcap_csv_pairs,
                max_workers=mcap_workers,
                desc=f"CPU-{process_id}: Processing MCAPs for {display_name}",
                unit="MCAP",
                chunksize=1
            )
            
            # Collect topics from MCAPs that were just processed
            for mcap_topic_types, success in processed_results:
                if success:
                    for topic, topic_type in mcap_topic_types.items():
                        if topic not in topic_types:
                            topic_types[topic] = topic_type
        except KeyboardInterrupt:
            if status_bar:
                status_bar.set_description_str(f"CPU-{process_id}: Interrupted - stopping {display_name}")
            raise
        except Exception as e:
            tqdm.write(f"CPU-{process_id}: Error processing rosbag {display_name}: {e}")
            return
    
    # Also collect topics from MCAPs that already had CSVs (if any were skipped)
    if len(mcap_csv_pairs) < len(mcap_files):
        if status_bar:
            status_bar.set_description_str(f"CPU-{process_id}: Collecting topics from existing MCAPs...")
        
        # Find MCAPs that weren't processed (already had CSVs)
        processed_mcap_numbers = {extract_mcap_number(mcap) for mcap, _ in mcap_csv_pairs}
        unprocessed_mcaps = [mcap for mcap in mcap_files if extract_mcap_number(mcap) not in processed_mcap_numbers]
        
        if unprocessed_mcaps:
            mcap_workers = min(MCAP_MAX_WORKERS, len(unprocessed_mcaps))
            try:
                topic_data_results = process_map(
                    process_single_mcap,
                    unprocessed_mcaps,
                    max_workers=mcap_workers,
                    desc=f"CPU-{process_id}: Collecting topics for {display_name}",
                    unit="MCAP",
                    chunksize=1
                )
                
                # Merge topic types from unprocessed MCAPs
                for mcap_topic_data, mcap_topic_types in topic_data_results:
                    for topic, topic_type in mcap_topic_types.items():
                        if topic not in topic_types:
                            topic_types[topic] = topic_type
            except Exception as e:
                tqdm.write(f"CPU-{process_id}: Warning: Error collecting topics from {display_name}: {e}")
    elif not mcap_csv_pairs:
        # All MCAPs already had CSVs, need to collect topics from all of them
        if status_bar:
            status_bar.set_description_str(f"CPU-{process_id}: Collecting topics from all MCAPs...")
        
        mcap_workers = min(MCAP_MAX_WORKERS, len(mcap_files))
        try:
            topic_data_results = process_map(
                process_single_mcap,
                mcap_files,
                max_workers=mcap_workers,
                desc=f"CPU-{process_id}: Collecting topics for {display_name}",
                unit="MCAP",
                chunksize=1
            )
            
            # Merge topic types from all MCAPs
            for mcap_topic_data, mcap_topic_types in topic_data_results:
                for topic, topic_type in mcap_topic_types.items():
                    if topic not in topic_types:
                        topic_types[topic] = topic_type
        except Exception as e:
            tqdm.write(f"CPU-{process_id}: Warning: Error collecting topics from {display_name}: {e}")
    
    # Save topics JSON only if an explicit output path is provided
    if topics_json_path and topic_types:
        if status_bar:
            status_bar.set_description_str(f"CPU-{process_id}: Generating topics JSON...")
        topics = sorted(topic_types.keys())
        topics_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(topics_json_path, 'w') as jsonfile:
            json.dump({"topics": topics, "types": topic_types}, jsonfile, indent=2)
        if status_bar:
            status_bar.set_description_str(f"CPU-{process_id}: Topics JSON generated")

def process_single_rosbag(rosbag_info: Tuple[Path, Path, Optional[Path], int, object]) -> Tuple[bool, str]:
    """Process a single rosbag: generate CSV and topics JSON if needed.
    
    Args:
        rosbag_info: Tuple containing (rosbag_path, csv_path, topics_json_path, bar_position, shared_stop_event)
        
    Returns:
        Tuple of (success: bool, rosbag_name: str)
    """
    rosbag_path, csv_path, topics_json_path, bar_position, shared_stop_event = rosbag_info
    
    # Check if we should stop before starting
    if shared_stop_event and shared_stop_event.is_set():
        return False, rosbag_path.name
    
    rosbag_name = rosbag_path.name
    parent_name = rosbag_path.parent.name
    is_multipart = parent_name.endswith("_multi_parts")
    process_id = os.getpid()
    
    # Get timing info for better progress tracking
    total_messages = get_total_message_count(rosbag_path)
    message_info = f" ({total_messages:,} msgs)" if total_messages else ""
    
    # Create status line for this worker (with more spacing)
    status_position = int(bar_position)

    with tqdm(total=0, desc=f"CPU-{process_id}: Starting {rosbag_name}{message_info}", 
              position=status_position, leave=False, bar_format='{desc}') as status_bar:
        start_time = time.time()
    
        try:
            # Check if all MCAP CSVs exist
            mcap_files = find_mcap_files(rosbag_path)
            
            # Determine CSV directory structure
            csv_base_dir = csv_path.parent
            if is_multipart:
                csv_mcap_dir = csv_base_dir / parent_name / rosbag_name
            else:
                csv_mcap_dir = csv_base_dir / rosbag_name
            
            # Check if all MCAP CSVs exist
            all_csvs_exist = True
            if mcap_files:
                for mcap_file in mcap_files:
                    mcap_number = extract_mcap_number(mcap_file)
                    mcap_csv_path = csv_mcap_dir / f"{mcap_number}.csv"
                    if not mcap_csv_path.exists():
                        all_csvs_exist = False
                        break
            
            # Process rosbag if any MCAP CSVs are missing
            csv_was_created = False
            if not all_csvs_exist:
                if is_multipart:
                    status_bar.set_description_str(f"CPU-{process_id}: Processing MCAP CSVs: {parent_name}/{rosbag_name}")
                else:
                    status_bar.set_description_str(f"CPU-{process_id}: Processing MCAP CSVs: {rosbag_name}")
                
                # Check again before processing
                if shared_stop_event and shared_stop_event.is_set():
                    status_bar.set_description_str(f"CPU-{process_id}: Interrupted - stopping {rosbag_name}")
                    return False, rosbag_name
                
                # For multipart rosbags, write topics JSON in subdirectory, one file per part
                if is_multipart:
                    topics_folder = TOPICS / parent_name
                    topics_folder.mkdir(parents=True, exist_ok=True)
                    topics_json_out = topics_folder / f"{rosbag_name}.json"
                else:
                    topics_json_out = TOPICS / f"{rosbag_name}.json"

                process_rosbag(rosbag_path, csv_path, topics_json_out, bar_position=bar_position, 
                              status_bar=status_bar, shared_stop_event=shared_stop_event)
                # Check if at least some CSVs were created
                csv_was_created = any(
                    (csv_mcap_dir / f"{extract_mcap_number(mcap_file)}.csv").exists()
                    for mcap_file in mcap_files
                ) if mcap_files else False
            else:
                status_bar.set_description_str(f"CPU-{process_id}: Skipping - all MCAP CSVs already exist: {rosbag_name}")
                csv_was_created = True
                
                # Still need to generate topics JSON if it doesn't exist
                if topics_json_path and not topics_json_path.exists():
                    if is_multipart:
                        topics_folder = TOPICS / parent_name
                        topics_folder.mkdir(parents=True, exist_ok=True)
                        topics_json_out = topics_folder / f"{rosbag_name}.json"
                    else:
                        topics_json_out = TOPICS / f"{rosbag_name}.json"
                    
                    # Process just to generate topics JSON (all CSVs exist, so it will skip CSV generation)
                    process_rosbag(rosbag_path, csv_path, topics_json_out, bar_position=bar_position, 
                                  status_bar=status_bar, shared_stop_event=shared_stop_event)
            
            # Topics JSON is generated within process_rosbag at the correct target path.
            # We skip separate generation here to avoid duplicates and wrong locations.
            
            # Check if we were interrupted during processing
            if shared_stop_event and shared_stop_event.is_set():
                status_bar.set_description_str(f"CPU-{process_id}: Interrupted - stopping {rosbag_name}")
                return False, rosbag_name
            
            elapsed = time.time() - start_time
            status_bar.set_description_str(f"CPU-{process_id}: Completed {rosbag_name} in {elapsed:.1f}s")
            return True, rosbag_name
        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            status_bar.set_description_str(f"CPU-{process_id}: Interrupted {rosbag_name} after {elapsed:.1f}s")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            status_bar.set_description_str(f"CPU-{process_id}: Failed {rosbag_name} after {elapsed:.1f}s: {e}")
            return False, rosbag_name


def collect_rosbags(rosbag_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """Collect all rosbag paths that need processing.
    
    Args:
        rosbag_dir: Root directory to search for rosbags
        
    Returns:
        List of tuples containing (rosbag_path, csv_path, topics_json_path)
    """
    rosbag_list: List[Tuple[Path, Path, Path]] = []
    
    # Find all directories containing metadata.yaml
    all_rosbag_paths = []
    for metadata_path in rosbag_dir.rglob("metadata.yaml"):
        rosbag_path = metadata_path.parent
        
        # Skip excluded directories
        if "EXCLUDED" in str(rosbag_path):
            continue
        
        all_rosbag_paths.append(rosbag_path)
    
    # Filter based on MODE
    if MODE == "single":
        if not SINGLE_BAG_NAME:
            raise SystemExit("MODE is 'single' but SINGLE_BAG_NAME is not set")
        target_path = rosbag_dir / SINGLE_BAG_NAME
        if target_path not in all_rosbag_paths:
            raise SystemExit(f"Rosbag '{SINGLE_BAG_NAME}' not found in {rosbag_dir}")
        all_rosbag_paths = [target_path]
    elif MODE == "multiple":
        if not MULTIPLE_BAG_NAMES:
            raise SystemExit("MODE is 'multiple' but MULTIPLE_BAG_NAMES is empty")
        # Filter to only include rosbags in MULTIPLE_BAG_NAMES
        filtered_paths = []
        for rosbag_path in all_rosbag_paths:
            if rosbag_path.name in MULTIPLE_BAG_NAMES:
                filtered_paths.append(rosbag_path)
        all_rosbag_paths = filtered_paths
        
        # Warn about missing rosbags
        found_names = {p.name for p in filtered_paths}
        missing = set(MULTIPLE_BAG_NAMES) - found_names
        if missing:
            print(f"⚠️  Warning: {len(missing)} rosbag(s) not found: {sorted(missing)}")
    # else MODE == "all": use all_rosbag_paths as-is
    
    # Build rosbag_list from filtered paths
    for rosbag_path in all_rosbag_paths:
        rosbag_name = rosbag_path.name
        parent_name = rosbag_path.parent.name

        if parent_name in SKIP:
            print(f"⏭️  Skipping {rosbag_path}: rosbag is in SKIP list")
            continue
        
        # Check if this is a part of a multipart rosbag (parent folder ends with _multi_parts)
        if parent_name.endswith("_multi_parts"):
            # This is a Part_N inside a multipart rosbag folder
            # Extract part number from the Part_N folder name
            part_name = rosbag_name  # e.g., "Part_1"
            csv_folder = LOOKUP_TABLES / parent_name
            csv_folder.mkdir(parents=True, exist_ok=True)
            csv_path = csv_folder / f"{part_name}.csv"
            
            # Topics JSON in subdirectory, one file per part
            topics_folder = TOPICS / parent_name
            topics_folder.mkdir(parents=True, exist_ok=True)
            topics_json_path = topics_folder / f"{part_name}.json"
        else:
            # Regular rosbag
            csv_path = LOOKUP_TABLES / f"{rosbag_name}.csv"
            topics_json_path = TOPICS / f"{rosbag_name}.json"
        
        rosbag_list.append((rosbag_path, csv_path, topics_json_path))
    
    return rosbag_list


# =========================
# Main execution
# =========================

def main() -> None:
    """Main function to process all rosbags in parallel."""
    start_time = time.time()
    print(f"Starting processing at {datetime.now().strftime('%H:%M:%S')}")
    
    # Validate environment variables
    if ROSBAGS is None:
        raise SystemExit("ROSBAGS environment variable not set")
    if LOOKUP_TABLES is None:
        raise SystemExit("LOOKUP_TABLES environment variable not set")
    if TOPICS is None:
        raise SystemExit("TOPICS environment variable not set")
    
    if not ROSBAGS.exists():
        raise SystemExit(f"ROSBAGS directory does not exist: {ROSBAGS}")
    
    # Validate MODE
    if MODE not in ("single", "all", "multiple"):
        raise SystemExit("MODE must be 'single', 'all', or 'multiple'")
    
    if MODE == "single" and not SINGLE_BAG_NAME:
        raise SystemExit("MODE is 'single' but SINGLE_BAG_NAME is not set")
    
    if MODE == "multiple" and not MULTIPLE_BAG_NAMES:
        raise SystemExit("MODE is 'multiple' but MULTIPLE_BAG_NAMES is empty")
    
    # Display mode
    if MODE == "single":
        print(f"▶️ MODE = single → Processing only: {SINGLE_BAG_NAME}")
    elif MODE == "multiple":
        print(f"▶️ MODE = multiple → Processing {len(MULTIPLE_BAG_NAMES)} specific rosbag(s):")
        for bag_name in MULTIPLE_BAG_NAMES:
            print(f"   - {bag_name}")
    else:  # MODE == "all"
        print("▶️ MODE = all → Processing ALL rosbags in ROSBAGS")
    
    # Set up tqdm lock for multiprocessing
    tqdm.set_lock(RLock())
    
    # Collect all rosbags
    print(f"Scanning directory: {ROSBAGS}")
    all_rosbags = collect_rosbags(ROSBAGS)
    print(f"Found {len(all_rosbags)} rosbags")
    
    if not all_rosbags:
        print("No rosbags found to process")
        return
    
    print(f"\nTotal rosbags to check: {len(all_rosbags)}")
    print(f"Using {MAX_WORKERS} CPU cores for parallel processing\n")
    
    # Process rosbags in parallel using limited workers to avoid I/O bottleneck
    global executor, stop_event, manager
    manager = Manager()
    stop_event = manager.Event()
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    
    futures: Dict = {}
    try:
        with executor:
            # Submit all rosbags for processing
            for idx, rosbag_info in enumerate(all_rosbags):
                if stop_event.is_set():
                    break
                bar_position = idx + 1  # unique position per rosbag (start at 1)
                rosbag_info_with_pos = (*rosbag_info, bar_position, stop_event)
                future = executor.submit(process_single_rosbag, rosbag_info_with_pos)
                futures[future] = rosbag_info_with_pos
            
            # Track results
            succeeded = 0
            failed = 0
            
            # Process completed futures as they finish
            for future in as_completed(futures):
                if stop_event.is_set():
                    # Cancel remaining futures
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
                
                rosbag_info = futures[future]
                rosbag_name = rosbag_info[0].name
                
                try:
                    success, name = future.result(timeout=1)
                    if success:
                        succeeded += 1
                    else:
                        failed += 1
                except TimeoutError:
                    # Future is still running, skip it
                    failed += 1
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"Exception processing {rosbag_name}: {e}")
                    failed += 1
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received. Shutting down...")
        stop_event.set()
        # Cancel all pending futures
        if futures:
            for future in list(futures.keys()):
                if not future.done():
                    future.cancel()
        
        # Terminate all worker processes
        if executor:
            try:
                processes = list(executor._processes.values())
                for process in processes:
                    try:
                        if process.is_alive():
                            process.terminate()
                    except Exception:
                        pass
                
                # Wait briefly
                time.sleep(1)
                
                # Force kill any remaining processes
                for process in processes:
                    try:
                        if process.is_alive():
                            process.kill()
                    except Exception:
                        pass
            except Exception:
                pass
        
        if executor:
            executor.shutdown(wait=False)
        sys.exit(130)
    
    # Print final timing and summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Average per rosbag: {total_time/len(all_rosbags):.1f}s")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Total: {len(all_rosbags)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
