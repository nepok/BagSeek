import os
import csv
from pathlib import Path
from flask import config
import numpy as np
import json
#from rosbags.rosbag2 import Reader  # type: ignore
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures
import time
import yaml
from datetime import datetime
from multiprocessing import RLock, Manager
import signal
import sys

PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Define constants for paths
ROSBAGS_DIR_MNT = os.getenv("ROSBAGS_DIR_MNT")
ROSBAGS_DIR_NAS = os.getenv("ROSBAGS_DIR_NAS")
LOOKUP_TABLES_DIR = os.getenv("LOOKUP_TABLES_DIR")
TOPICS_DIR = os.getenv("TOPICS_DIR")

SKIP = ["rosbag2_2025_08_19-09_33_16_multi_parts"]

# Timestamp source configuration
# Options: "log_time" (MCAP recording timestamp) or "header_stamp" (message internal timestamp)
TIMESTAMP_SOURCE = 'header_stamp'  # or "log_time"
TFMESSAGE_TYPE = "tf2_msgs/msg/TFMessage"
WARNED_TOPICS = set()

max_workers = 2  # Reduced to avoid terminal output contention

# Global executor for clean shutdown
executor = None
# Global manager for shared state
manager = None
# Global event to signal workers to stop
stop_event = None

def signal_handler(signum, frame):
    """Handle keyboard interrupt (Ctrl+C) gracefully."""
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

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_worker_position():
    """Assign a unique position to each worker process based on process ID."""
    process_id = os.getpid()
    # Use process ID to get a consistent position for each worker
    # Multiply by 3 to give more space between progress bars
    return process_id % max_workers

def get_total_message_count(rosbag_path):
    """Extract total message count from metadata.yaml for accurate progress tracking."""
    metadata_path = os.path.join(rosbag_path, "metadata.yaml")
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
    except:
        return None


# Determine the topic with the most messages to use as the reference timeline
def determine_reference_topic(topic_timestamps):
    return max(topic_timestamps.items(), key=lambda x: len(x[1]))[0]

# Create a refined reference timeline with smaller intervals to improve alignment accuracy
# The factor reduces the mean interval to create a denser timeline for better matching
def create_reference_timestamps(timestamps, factor=2):
    timestamps = sorted(set(timestamps))
    diffs = np.diff(timestamps)
    mean_interval = np.mean(diffs)
    refined_interval = mean_interval / factor
    ref_start = timestamps[0]
    ref_end = timestamps[-1]
    return np.arange(ref_start, ref_end, refined_interval).astype(np.int64)

# Align timestamps of a topic to the closest reference timestamps within max_diff
# For each reference timestamp, find the closest topic timestamp to align messages across topics
def align_topic_to_reference(topic_ts, ref_ts, max_diff=int(1e8)):
    aligned = []
    topic_idx = 0
    for rt in ref_ts:
        while topic_idx + 1 < len(topic_ts) and abs(topic_ts[topic_idx + 1] - rt) < abs(topic_ts[topic_idx] - rt):
            topic_idx += 1
        closest = topic_ts[topic_idx]
        if abs(closest - rt) <= max_diff:
            aligned.append(closest)
        else:
            aligned.append(None)
    return aligned


def extract_timestamp(topic, data, log_time, topic_type_map, process_id):
    topic_type = topic_type_map.get(topic)

    if TIMESTAMP_SOURCE == "header_stamp":
        if topic_type == TFMESSAGE_TYPE:
            return log_time, topic_type

        if topic_type:
            try:
                msg_type = get_message(topic_type)
                msg = deserialize_message(data, msg_type)
                header = getattr(msg, "header", None)
                if header is None:
                    raise AttributeError("message has no header")
                timestamp = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
                return timestamp, topic_type
            except Exception as exc:
                if topic not in WARNED_TOPICS:
                    tqdm.write(f"CPU-{process_id}: Warning: Could not extract header timestamp from {topic}: {exc}")
                    WARNED_TOPICS.add(topic)
                return log_time, topic_type

        return log_time, "unknown"

    return log_time, topic_type or "unknown"

def process_rosbag(rosbag_path, csv_path, topics_json_path=None, bar_position=0, status_bar=None, shared_stop_event=None):
    topic_data = defaultdict(list)
    topic_types = {}
    
    # Check if we should stop
    if shared_stop_event and shared_stop_event.is_set():
        return
    
    # Get total message count for accurate progress tracking
    total_messages = get_total_message_count(rosbag_path)
    rosbag_name = os.path.basename(rosbag_path)
    parent_name = os.path.basename(os.path.dirname(rosbag_path))
    # Build display name: for multipart rosbags show parent/Part_N
    if parent_name.endswith("_multi_parts"):
        display_name = f"{parent_name}/{rosbag_name}"
    else:
        display_name = rosbag_name
    process_id = os.getpid()
    
    reader = None
    try:
        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=rosbag_path, storage_id="mcap"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
        )
        
        # Get topic types for header_stamp extraction
        topic_type_map = {}
        if TIMESTAMP_SOURCE == "header_stamp":
            try:
                all_topics = reader.get_all_topics_and_types()
                topic_type_map = {topic.name: topic.type for topic in all_topics}
            except Exception as exc:
                tqdm.write(f"CPU-{process_id}: Warning: Could not get topic types: {exc}")
        
        if status_bar:
            status_bar.set_description_str(f"CPU-{process_id}: Reading messages (timestamp: {TIMESTAMP_SOURCE})...")
        
        if total_messages:
            # Use known total for accurate percentage
            with tqdm(total=total_messages, desc=f"CPU-{process_id}: {display_name}", 
                      position=bar_position, leave=False, unit="msg", unit_scale=True, dynamic_ncols=True, miniters=1000, maxinterval=1.0) as pbar:
                while reader.has_next():
                    # Check if we should stop
                    if shared_stop_event and shared_stop_event.is_set():
                        tqdm.write(f"CPU-{process_id}: Interrupted - stopping {display_name}")
                        break
                    
                    try:
                        topic, data, log_time = reader.read_next()
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        # If reader was closed/interrupted, break
                        if shared_stop_event and shared_stop_event.is_set():
                            break
                        raise
                    
                    timestamp, resolved_type = extract_timestamp(topic, data, log_time, topic_type_map, process_id)
                    topic_data[topic].append(timestamp)
                    topic_types[topic] = resolved_type
                    pbar.update(1)
        else:
            # Fallback: count as we go (no percentage)
            with tqdm(desc=f"CPU-{process_id}: {display_name}", 
                      position=bar_position, leave=False, unit="msg", unit_scale=True, dynamic_ncols=True) as pbar:
                while reader.has_next():
                    # Check if we should stop
                    if shared_stop_event and shared_stop_event.is_set():
                        tqdm.write(f"CPU-{process_id}: Interrupted - stopping {display_name}")
                        break
                    
                    try:
                        topic, data, log_time = reader.read_next()
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        # If reader was closed/interrupted, break
                        if shared_stop_event and shared_stop_event.is_set():
                            break
                        raise
                    
                    timestamp, resolved_type = extract_timestamp(topic, data, log_time, topic_type_map, process_id)
                    topic_data[topic].append(timestamp)
                    topic_types[topic] = resolved_type
                    pbar.update(1)
        
        # Close the reader explicitly
        if reader:
            try:
                del reader
            except:
                pass
    except KeyboardInterrupt:
        # Clean up reader on interrupt
        if reader:
            try:
                del reader
            except:
                pass
        if status_bar:
            status_bar.set_description_str(f"CPU-{process_id}: Interrupted - stopping {display_name}")
        raise
    except Exception as e:
        # Clean up reader on error
        if reader:
            try:
                del reader
            except:
                pass
        # Print error after progress bar is done
        tqdm.write(f"CPU-{process_id}: Error reading bag {display_name}: {e}")
        # Error logging code here...#
        # ==================== TEMPORARY ERROR LOGGING - REMOVE AFTER DEBUGGING ====================
        error_log_path = "/mnt/data/rosbag_errors.csv"
        try:
            # Check if file exists to write header
            file_exists = os.path.exists(error_log_path)
            with open(error_log_path, 'a', newline='', encoding='utf-8') as error_file:
                writer = csv.writer(error_file)
                if not file_exists:
                    writer.writerow(['rosbag_path', 'error_type', 'error_message'])
                writer.writerow([rosbag_path, type(e).__name__, str(e)])
        except:
            pass  # Don't fail the main process if error logging fails
        # ========================================================================================
        return
    
    if not topic_data:
        # print(f"CPU-{process_id}: No topics found in {rosbag_path}")
        return
    
    # Save topics JSON only if an explicit output path is provided
    if topics_json_path:
        if status_bar:
            status_bar.set_description_str(f"CPU-{process_id}: Generating topics JSON...")
        topics = sorted(topic_data.keys())
        os.makedirs(TOPICS_DIR, exist_ok=True)
        with open(topics_json_path, 'w') as jsonfile:
            json.dump({"topics": topics, "types": topic_types}, jsonfile, indent=2)
        if status_bar:
            status_bar.set_description_str(f"CPU-{process_id}: Topics JSON generated")

    # Determine reference topic by message count
    if status_bar:
        status_bar.set_description_str(f"CPU-{process_id}: Aligning topics...")
    ref_topic = determine_reference_topic(topic_data)
    ref_ts = create_reference_timestamps(topic_data[ref_topic])

    # Align each topic to the reference timeline to synchronize timestamps across topics
    aligned_data = {}
    for topic, timestamps in topic_data.items():
        aligned_data[topic] = align_topic_to_reference(timestamps, ref_ts)

    # Prepare CSV data
    if status_bar:
        status_bar.set_description_str(f"CPU-{process_id}: Writing CSV...")
    # The CSV rows contain the reference timestamp, aligned timestamps for each topic, and the max timestamp distance across topics
    csv_data = []
    header = ['Reference Timestamp'] + list(aligned_data.keys()) + ['Max Distance']
    for i, ref_time in enumerate(ref_ts):
        row = [int(ref_time)]
        row_values = []

        for topic in aligned_data.keys():
            ts = aligned_data[topic][i]
            row.append(f"{ts}" if ts is not None else "None")
            if ts is not None:
                row_values.append(ts)

        max_distance = max(row_values) - min(row_values) if len(row_values) > 1 else None
        row.append(f"{max_distance}" if max_distance is not None else "None")
        csv_data.append(row)

    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(csv_data)

    if status_bar:
        status_bar.set_description_str(f"CPU-{process_id}: CSV generated")

def create_topics_json_from_csv(csv_path):
    try:
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            # Skip the first column (Reference Timestamp) and the last column (Max Distance)
            topics = header[1:-1]
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    if not topics:
        print(f"No topics found in {csv_path}")
        return

    topics = sorted(topics)
    os.makedirs(TOPICS_DIR, exist_ok=True)
    rosbag_name = os.path.splitext(os.path.basename(csv_path))[0]
    topics_json_path = os.path.join(TOPICS_DIR, f"{rosbag_name}.json")
    if not os.path.exists(topics_json_path):
        with open(topics_json_path, 'w') as jsonfile:
            json.dump({"topics": topics}, jsonfile, indent=2)
        print(f"Topics JSON file generated: {topics_json_path}")
    else:
        print(f"Skipping already existing topics JSON: {topics_json_path}")

def process_single_rosbag(rosbag_info):
    """Process a single rosbag: generate CSV and topics JSON if needed."""
    rosbag_path, csv_path, topics_json_path, bar_position, shared_stop_event = rosbag_info
    
    # Check if we should stop before starting
    if shared_stop_event and shared_stop_event.is_set():
        return False, os.path.basename(rosbag_path)
    
    rosbag_name = os.path.basename(rosbag_path)
    parent_name = os.path.basename(os.path.dirname(rosbag_path))
    is_multipart = parent_name.endswith("_multi_parts")
    process_id = os.getpid()
    
    # Get timing info for better progress tracking
    total_messages = get_total_message_count(rosbag_path)
    message_info = f" ({total_messages:,} msgs)" if total_messages else ""
    
    # Create status line for this worker (with more spacing)
    status_position = int(bar_position)

    with tqdm(total=0, desc=f"CPU-{process_id}: Starting {rosbag_name}{message_info}", 
              position=status_position, leave=False, bar_format='{desc}') as status_bar:
        # ... rest of the function stays the same
        
        start_time = time.time()
    
        try:
            # Process rosbag to CSV if needed
            csv_was_created = False
            if not os.path.exists(csv_path):
                if parent_name.endswith("_multi_parts"):
                    status_bar.set_description_str(f"CPU-{process_id}: Processing CSV: {parent_name}/{rosbag_name}")
                else:
                    status_bar.set_description_str(f"CPU-{process_id}: Processing CSV: {rosbag_name}")
                
                # Check again before processing
                if shared_stop_event and shared_stop_event.is_set():
                    status_bar.set_description_str(f"CPU-{process_id}: Interrupted - stopping {rosbag_name}")
                    return False, rosbag_name
                
                # For multipart rosbags, write topics JSON only once from Part_1 to parent-named JSON
                if is_multipart:
                    if rosbag_name == "Part_1" or rosbag_name.startswith("Part_1"):
                        topics_json_out = os.path.join(TOPICS_DIR, f"{parent_name}.json")
                    else:
                        topics_json_out = None
                else:
                    topics_json_out = os.path.join(TOPICS_DIR, f"{rosbag_name}.json")

                process_rosbag(rosbag_path, csv_path, topics_json_out, bar_position=bar_position, status_bar=status_bar, shared_stop_event=shared_stop_event)
                # Check if CSV was actually created (process_rosbag might fail with corrupted bags)
                csv_was_created = os.path.exists(csv_path)
            else:
                status_bar.set_description_str(f"CPU-{process_id}: Skipping already processed CSV: {rosbag_name}")
                csv_was_created = True
            
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


def collect_rosbags(rosbag_dir):
    
    """Collect all rosbag paths that need processing."""
    rosbag_list = []
    multipart_counters = {}  # Track part numbers for multipart rosbags
    
    for root, dirs, files in os.walk(rosbag_dir):
        if "metadata.yaml" in files:
            print(f"Found rosbag: {root}")
            rosbag_path = root
            
            if "EXCLUDED" in rosbag_path:
                continue
            
            rosbag_name = os.path.basename(rosbag_path)
            parent_name = os.path.basename(os.path.dirname(rosbag_path))

            if parent_name in SKIP:
                continue
            
            # Check if this is a part of a multipart rosbag (parent folder ends with _multi_parts)
            if parent_name.endswith("_multi_parts"):
                # This is a Part_N inside a multipart rosbag folder
                # Extract part number from the Part_N folder name
                part_name = rosbag_name  # e.g., "Part_1"
                csv_folder = os.path.join(LOOKUP_TABLES_DIR, parent_name)
                os.makedirs(csv_folder, exist_ok=True)
                csv_path = os.path.join(csv_folder, f"{part_name}.csv")
                
                # Topics JSON should be named after the parent multipart rosbag
                topics_json_path = os.path.join(TOPICS_DIR, f"{parent_name}.json")
            else:
                # Regular rosbag
                csv_path = os.path.join(LOOKUP_TABLES_DIR, f"{rosbag_name}.csv")
                topics_json_path = os.path.join(TOPICS_DIR, f"{rosbag_name}.json")
            
            rosbag_list.append((rosbag_path, csv_path, topics_json_path))
    
    return rosbag_list


# Main function walks through rosbags directory and processes each rosbag in parallel
# This avoids redundant computation by checking for existing CSV and JSON files before processing
def main():
    start_time = time.time()
    print(f"Starting processing at {datetime.now().strftime('%H:%M:%S')}")
    
    # Set up tqdm lock for multiprocessing
    tqdm.set_lock(RLock())
    
    # Collect all rosbags from both directories
    all_rosbags = []
    for dir in [ROSBAGS_DIR_NAS]: #ROSBAGS_DIR_MNT, 
        if dir:  # Only process if directory is defined
            print(f"Scanning directory: {dir}")
            rosbags = collect_rosbags(dir)
            all_rosbags.extend(rosbags)
            print(f"Found {len(rosbags)} rosbags in {dir}")
    
    if not all_rosbags:
        print("No rosbags found to process")
        return
    
    print(f"\nTotal rosbags to check: {len(all_rosbags)}")
    # Use fewer workers to reduce I/O contention
    print(f"Using {max_workers} CPU cores for parallel processing\n")
    
    # Process rosbags in parallel using limited workers to avoid I/O bottleneck
    global executor, stop_event, manager
    manager = Manager()
    stop_event = manager.Event()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    
    futures = {}
    try:
        with executor:
            # Submit all rosbags for processing
            # Reserve e.g. line 0 for overall status, then assign one line per worker starting at 1
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
            for future in concurrent.futures.as_completed(futures):
                if stop_event.is_set():
                    # Cancel remaining futures
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
                
                rosbag_info = futures[future]
                rosbag_name = os.path.basename(rosbag_info[0])
                
                try:
                    success, name = future.result(timeout=1)
                    if success:
                        succeeded += 1
                    else:
                        failed += 1
                except concurrent.futures.TimeoutError:
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