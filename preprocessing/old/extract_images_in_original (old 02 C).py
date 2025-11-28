import concurrent.futures
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from rclpy.serialization import deserialize_message  # type: ignore
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions  # type: ignore
from rosidl_runtime_py.utilities import get_message  # type: ignore
from rosbags.typesys import Stores, get_typestore
from sensor_msgs.msg import CompressedImage, Image  # type: ignore
from tqdm import tqdm

PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

MISSING_IMAGE_ABS_TOLERANCE = int(os.getenv("MISSING_IMAGE_TOLERANCE", "3"))
MISSING_IMAGE_RATIO_TOLERANCE = float(os.getenv("MISSING_IMAGE_RATIO_TOLERANCE", "0.0"))

# Configuration: Set to False to reprocess all rosbags (even if previously attempted)
SKIP_PROCESSED_ROSBAGS = True

# Configuration: Specify which rosbags to process (empty list = process all)
# Example: TARGET_ROSBAGS = ["rosbag2_2025_07_29-16_32_17", "rosbag2_2025_08_04-15_00_07"]
TARGET_ROSBAGS = [
    "rosbag2_2025_07_24-16_01_22" ,
    "rosbag2_2025_07_29-08_08_48" ,
    "rosbag2_2025_07_30-10_33_53" ,
    "rosbag2_2025_08_04-15_00_07" ,
    "rosbag2_2025_08_05-11_01_40" ,
    "rosbag2_2025_08_19-07_54_26" ,
    "rosbag2_2025_08_22-11_42_48" ,
    "rosbag2_2025_08_22-12_27_43" ,
    "rosbag2_2025_08_26-07_27_19" ,
    "rosbag2_2025_09_11-12_59_00" ,
    "rosbag2_2025_09_25-08_10_26" ,
]
COMPLETED_STATUS_STATES = {"complete", "complete_with_tolerance"}

max_workers = 4

def resolve_env_path(var_name: str, must_exist: bool) -> Path:
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(f"Environment variable '{var_name}' is not set")

    path = Path(value).expanduser()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path specified by '{var_name}' does not exist: {path}")
    return path


def compute_metadata_fingerprint(metadata_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(metadata_path, "rb") as metadata_file:
        for chunk in iter(lambda: metadata_file.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def topic_to_directory_name(topic: str) -> str:
    return topic.replace("/", "__")


def load_bag_status(status_path: Path) -> dict[str, Any] | None:
    if not status_path.exists():
        return None

    try:
        with open(status_path, "r", encoding="utf-8") as status_file:
            return json.load(status_file)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to load status file {status_path}: {exc}")
        return None


def save_bag_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as status_file:
        json.dump(payload, status_file, indent=2, sort_keys=True)


def collect_failed_extractions(bag_output_dir: Path, topic_dir_map: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    failure_root = bag_output_dir / "_failed_extractions"
    if not failure_root.exists():
        return {}

    failed: dict[str, dict[str, list[str]]] = {}
    for topic, dir_name in topic_dir_map.items():
        topic_failure_dir = failure_root / dir_name
        if not topic_failure_dir.is_dir():
            continue
        topic_failed: dict[str, list[str]] = {}
        for log_path in sorted(topic_failure_dir.glob("*.log")):
            try:
                raw = log_path.read_text(encoding="utf-8")
            except Exception:
                raw = ""
            details = [line for line in raw.splitlines() if line.strip()]
            topic_failed[log_path.stem] = details or ["No details recorded"]
        if topic_failed:
            failed[topic] = topic_failed
    return failed


def record_failed_extraction(topic_dir: Path, topic: str, timestamp: str, reason: str) -> None:
    failure_root = topic_dir.parent / "_failed_extractions"
    failure_dir = failure_root / topic_to_directory_name(topic)
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure_log = failure_dir / f"{timestamp}.log"
    with open(failure_log, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{datetime.utcnow().isoformat()}Z] {reason}\n")


def write_missing_summary(output_dir: Path, bag_name: str, issues: list[dict[str, Any]]) -> None:
    summary_path = output_dir / "missing_summary.json"
    payload = {
        "bag_name": bag_name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "issues": issues,
    }
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(payload, summary_file, indent=2, sort_keys=True)


def finalize_bag_status(
    rosbag_name: str,
    output_dir: Path,
    metadata_fingerprint: str,
    metadata_counts: dict[str, int],
    image_topics: list[str],
    existing_files_map: dict[str, set[str]],
    remaining_missing: dict[str, set[str]],
    diagnostics: dict[str, dict[str, int | None]],
    tolerated_topics: set[str] | None = None,
) -> dict[str, Any]:
    tolerated_topics = tolerated_topics or set()
    existing_counts: dict[str, int] = {
        topic: len(existing_files_map.get(topic, set())) for topic in image_topics
    }
    sorted_missing: dict[str, list[str]] = {
        topic: sorted(timestamps) for topic, timestamps in remaining_missing.items() if timestamps
    }

    missing_counts: dict[str, int | None] = {}
    issues: list[dict[str, Any]] = []
    for topic in image_topics:
        expected = metadata_counts.get(topic)
        missing_count: int | None = None
        if expected is not None:
            missing_count = max(expected - existing_counts.get(topic, 0), 0)
        missing_counts[topic] = missing_count

        pending_timestamps = sorted_missing.get(topic, [])
        if (missing_count and missing_count > 0) or pending_timestamps:
            issues.append(
                {
                    "topic": topic,
                    "expected_from_metadata": expected,
                    "extracted_images": existing_counts.get(topic, 0),
                    "missing_vs_metadata": missing_count,
                    "pending_timestamps": pending_timestamps,
                    "within_tolerance": topic in tolerated_topics,
                }
            )

    topic_dir_map = {topic: topic_to_directory_name(topic) for topic in image_topics}
    failed_extractions = collect_failed_extractions(output_dir, topic_dir_map)

    unresolved_issues = [issue for issue in issues if not issue.get("within_tolerance")]
    if not issues:
        state = "complete"
    elif not unresolved_issues:
        state = "complete_with_tolerance"
    else:
        state = "incomplete"

    status_payload: dict[str, Any] = {
        "bag_name": rosbag_name,
        "checked_at": datetime.utcnow().isoformat() + "Z",
        "state": state,
        "processed": True,
        "metadata_fingerprint": metadata_fingerprint,
        "metadata_counts": metadata_counts,
        "existing_counts": existing_counts,
        "missing_counts": missing_counts,
        "diagnostics": diagnostics,
        "remaining_missing": sorted_missing,
        "failed_extractions": failed_extractions,
        "tolerated_topics": sorted(tolerated_topics),
    }

    status_path = output_dir / "status.json"
    save_bag_status(status_path, status_payload)
    write_missing_summary(output_dir, rosbag_name, issues)

    return status_payload


# Define validated paths
ROSBAGS_DIR_NAS = resolve_env_path("ROSBAGS_DIR_NAS", must_exist=True)
LOOKUP_TABLES_DIR = resolve_env_path("LOOKUP_TABLES_DIR", must_exist=False)
ORIGINAL_IMAGES = resolve_env_path("ORIGINAL_IMAGES_DIR", must_exist=False)
TOPICS_DIR = resolve_env_path("TOPICS_DIR", must_exist=True)

# Create a typestore and get the Image message class
typestore = get_typestore(Stores.LATEST)

# Loads message counts for image topics from metadata.yaml
def load_image_topic_counts_from_metadata(bag_path: Path) -> dict[str, int]:
    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Warning: failed to parse {metadata_path}: {exc}")
        return {}

    info = metadata.get("rosbag2_bagfile_information", {})
    topics = info.get("topics_with_message_count", []) or []

    counts: dict[str, int] = {}
    for entry in topics:
        meta = entry.get("topic_metadata", {}) or {}
        type_name = meta.get("type")
        if not type_name or "Image" not in type_name:
            continue

        topic_name = meta.get("name")
        if not topic_name:
            continue

        counts[topic_name] = int(entry.get("message_count", 0))

    return counts

# Detects whether a bag file is in .mcap format (db3 support removed)
def detect_bag_format(bag_path: Path):
    metadata_file = bag_path / "metadata.yaml"

    if not metadata_file.is_file():
        return None

    if metadata_file.stat().st_size == 0:
        print(f"Skipping {bag_path.name}: metadata.yaml is empty")
        return None
    
    # Only check for mcap format since all rosbags are mcap
    for entry in bag_path.iterdir():
        if entry.suffix == ".mcap":
            return "mcap"

    return None

# Load the lookup table from a CSV file into a dictionary
def load_lookup_table(csv_path: Path):
    lookup = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ref_timestamp = row['Reference Timestamp']
            topic_timestamps = {topic: row[topic] for topic in reader.fieldnames[1:] if row[topic] != 'None'}
            lookup[ref_timestamp] = topic_timestamps
    return lookup


def build_topic_timestamp_map(aligned_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Create fast lookup maps: topic -> {message timestamp -> reference timestamp}."""
    topic_timestamp_map: dict[str, dict[str, str]] = {}

    if "Reference Timestamp" not in aligned_df.columns:
        return topic_timestamp_map

    reference_values = aligned_df["Reference Timestamp"].tolist()

    for column in aligned_df.columns:
        if column == "Reference Timestamp":
            continue

        topic_values = aligned_df[column].tolist()
        mapping: dict[str, str] = {}

        for ref_ts, topic_ts in zip(reference_values, topic_values):
            if not topic_ts or topic_ts in {"None", "nan", "NaN"}:
                continue
            mapping[str(topic_ts)] = str(ref_ts)

        if mapping:
            topic_timestamp_map[column] = mapping

    return topic_timestamp_map


def analyze_missing_messages(
    rosbag_path: Path,
    image_topics: list[str],
    output_dir: Path,
    topic_timestamp_map: dict[str, dict[str, str]],
    metadata_counts: dict[str, int] | None = None,
) -> tuple[
    dict[str, int],
    dict[str, set[str]],
    set[str],
    dict[str, dict[str, int | None]],
    dict[str, set[str]],
]:
    """Inspect metadata.yaml and existing outputs to determine what still needs extraction.

    Returns:
        metadata_counts: topic -> total messages reported by metadata.yaml.
        missing_timestamp_map: topic -> set of timestamps that still need to be extracted (based on lookup tables).
        full_scan_topics: topics that require scanning all messages (metadata indicates missing messages but lookup data is incomplete).
        diagnostics: topic -> summary counts for logging purposes.
        existing_files_map: topic -> set of already extracted image stems.
    """
    if metadata_counts is None:
        metadata_counts = load_image_topic_counts_from_metadata(rosbag_path)
    missing_timestamp_map: dict[str, set[str]] = {}
    full_scan_topics: set[str] = set()
    diagnostics: dict[str, dict[str, int | None]] = {}
    existing_files_map: dict[str, set[str]] = {}

    for topic in image_topics:
        topic_dir = output_dir / topic_to_directory_name(topic)
        existing_files = set()
        if topic_dir.exists():
            # Look for any image files in the topic directory and all mcap subdirectories
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
                # Check direct topic directory
                for p in topic_dir.glob(f"*{ext}"):
                    if p.is_file():
                        existing_files.add(p.stem)
                # Check mcap subdirectories
                for mcap_subdir in topic_dir.iterdir():
                    if mcap_subdir.is_dir():
                        for p in mcap_subdir.glob(f"*{ext}"):
                            if p.is_file():
                                existing_files.add(p.stem)
        existing_files_map[topic] = existing_files

        expected_count = metadata_counts.get(topic)
        existing_count = len(existing_files)
        missing_vs_metadata: int | None = None
        if expected_count is not None:
            missing_vs_metadata = max(expected_count - existing_count, 0)

        mapped_timestamps = set(topic_timestamp_map.get(topic, {}).keys())
        missing_from_lookup = {ts for ts in mapped_timestamps if ts not in existing_files}

        unmatched_missing = 0
        if missing_vs_metadata is not None:
            unmatched_missing = max(missing_vs_metadata - len(missing_from_lookup), 0)

        diagnostics[topic] = {
            "metadata_count": expected_count if expected_count is not None else None,
            "existing_images": existing_count,
            "missing_vs_metadata": missing_vs_metadata,
            "missing_identified": len(missing_from_lookup),
            "missing_unmapped": unmatched_missing if missing_vs_metadata is not None else None,
        }

        if missing_vs_metadata is None:
            if missing_from_lookup:
                missing_timestamp_map[topic] = missing_from_lookup
            continue

        if missing_vs_metadata == 0:
            continue

        if missing_from_lookup:
            missing_timestamp_map[topic] = missing_from_lookup

        if unmatched_missing > 0 or not missing_from_lookup:
            full_scan_topics.add(topic)

    # Topics marked for full scan should not keep targeted-only sets; we'll re-scan every message.
    for topic in full_scan_topics:
        missing_timestamp_map.pop(topic, None)

    return metadata_counts, missing_timestamp_map, full_scan_topics, diagnostics, existing_files_map


def compute_topic_missing_totals(
    topic: str,
    diagnostics: dict[str, dict[str, int | None]],
) -> tuple[int, int | None]:
    diag = diagnostics.get(topic, {}) or {}
    missing_identified = int(diag.get("missing_identified") or 0)
    missing_vs_metadata_raw = diag.get("missing_vs_metadata")
    missing_vs_metadata = int(missing_vs_metadata_raw) if missing_vs_metadata_raw is not None else None

    if missing_vs_metadata is None:
        total_missing = missing_identified
    else:
        total_missing = max(missing_vs_metadata, missing_identified)

    expected_count_raw = diag.get("metadata_count")
    expected_count = int(expected_count_raw) if expected_count_raw is not None else None
    return total_missing, expected_count

# Reads messages from an .mcap rosbag with file tracking using rosbags Reader
def read_mcap_messages(bag_path: Path, allowed_topics: set[str]):
    from rosbags.rosbag2 import Reader as RosbagReader
    
    with RosbagReader(str(bag_path)) as reader:
        connections = [x for x in reader.connections if x.topic in allowed_topics]
        
        # Get the list of MCAP files from the rosbag
        mcap_files = list(bag_path.glob("*.mcap"))
        mcap_files.sort()  # Ensure consistent ordering
        
        # Track which file we're currently reading from
        current_file_index = 0
        messages_processed = 0
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # Simple heuristic to determine which MCAP file we're reading from
            # This is not perfect but gives us a reasonable approximation
            if len(mcap_files) > 1:
                # Estimate which file based on message count and file sizes
                if current_file_index < len(mcap_files) - 1:
                    # Check if we should move to next file (heuristic)
                    # This is a rough approximation
                    pass
            
            current_mcap_file = mcap_files[current_file_index].name if current_file_index < len(mcap_files) else mcap_files[0].name
            
            try:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                yield connection.topic, msg, timestamp, current_mcap_file
                messages_processed += 1
            except Exception as exc:
                print(f"Error deserializing message from {connection.topic} at {timestamp}: {exc}")
                continue

# Saves the image to a file in its original format
def save_image_original(img, img_filepath: Path, original_format: str) -> None:
    """Save image in its original format without conversion."""
    if original_format.lower() in ['jpg', 'jpeg']:
        success = cv2.imwrite(str(img_filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif original_format.lower() == 'png':
        success = cv2.imwrite(str(img_filepath), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        # Default to JPEG for other formats
        success = cv2.imwrite(str(img_filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    if not success:
        raise IOError(f"cv2.imwrite returned False for {img_filepath}")

# Extracts and saves an image from a single message in original format
def extract_image_from_message(msg, topic, timestamp, output_dir: Path, mcap_file: str = None):
    # Create folder structure: rosbag -> topics -> mcaps -> images
    if mcap_file:
        # Remove .mcap extension for folder name
        mcap_folder = mcap_file.replace('.mcap', '')
        topic_dir = output_dir / topic_to_directory_name(topic) / mcap_folder
    else:
        topic_dir = output_dir / topic_to_directory_name(topic)
    
    topic_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine original format and save accordingly
    if isinstance(msg, CompressedImage) or msg.__class__.__name__ == "sensor_msgs__msg__CompressedImage":
        # For compressed images, try to determine format from data
        data = msg.data
        # Convert numpy array to bytes if needed
        if hasattr(data, 'tobytes'):
            data_bytes = data.tobytes()
        else:
            data_bytes = bytes(data)
        
        if data_bytes.startswith(b'\xff\xd8\xff'):
            # JPEG format
            img_filename = f"{timestamp}.jpg"
            original_format = "jpg"
        elif data_bytes.startswith(b'\x89PNG'):
            # PNG format
            img_filename = f"{timestamp}.png"
            original_format = "png"
        else:
            # Default to JPEG
            img_filename = f"{timestamp}.jpg"
            original_format = "jpg"
    else:
        # For raw images, default to PNG to preserve quality
        img_filename = f"{timestamp}.png"
        original_format = "png"
    
    img_filepath = topic_dir / img_filename
    
    if img_filepath.exists():
        print(f"Skipping already extracted image: {img_filepath}")
        return

    try:
        if isinstance(msg, CompressedImage) or msg.__class__.__name__ == "sensor_msgs__msg__CompressedImage":
            # For compressed image, save the raw data directly
            with open(img_filepath, 'wb') as f:
                # Convert numpy array to bytes if needed
                if hasattr(msg.data, 'tobytes'):
                    f.write(msg.data.tobytes())
                else:
                    f.write(bytes(msg.data))
        elif isinstance(msg, Image) or msg.__class__.__name__ == "sensor_msgs__msg__Image":
            # For raw image in ROS 2
            channels = {
                "mono8": 1,
                "rgb8": 3,
                "bgr8": 3,
                "rgba8": 4,
                "bgra8": 4
            }.get(msg.encoding)

            if channels is None:
                raise NotImplementedError(f"Unsupported encoding: {msg.encoding}")

            img_data = np.frombuffer(msg.data, dtype=np.uint8)
            img_data = img_data.reshape((msg.height, msg.width, channels))

            if msg.encoding == "rgb8":
                img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "rgba8":
                img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else:
                img = img_data

            save_image_original(img, img_filepath, original_format)
        else:
            raise TypeError(f"Unsupported message type: {type(msg)}")

    except Exception as e:  # noqa: BLE001
        reason = f"Failed to extract image from {topic} at {timestamp}: {e}"
        record_failed_extraction(output_dir, topic, str(timestamp), reason)
        print(reason)

# Extract images from a single rosbag and save them to the output directory
def extract_images_from_rosbag(rosbag_path: Path, output_dir: Path, csv_path: Path, format: str):
    # CHECK STATUS FIRST - before any expensive operations!
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    existing_status = load_bag_status(status_path)
    
    # Migrate old status.json files to include "processed" field
    if existing_status and "processed" not in existing_status:
        existing_status["processed"] = True
        save_bag_status(status_path, existing_status)
        print(f"{rosbag_path.name}: migrated old status.json to include 'processed' field")
    
    # Early exit if already processed (saves time on CSV/JSON loading)
    if SKIP_PROCESSED_ROSBAGS and existing_status and existing_status.get("processed"):
        # Still need to verify metadata hasn't changed
        metadata_path = rosbag_path / "metadata.yaml"
        if metadata_path.exists():
            current_fingerprint = compute_metadata_fingerprint(metadata_path)
            if existing_status.get("metadata_fingerprint") == current_fingerprint:
                state = existing_status.get("state", "unknown")
                print(f"{rosbag_path.name}: already processed (state={state}) — skipping")
                return
        # If metadata changed or doesn't exist, continue processing
    
    # Now do the expensive stuff only if needed
    aligned_data = pd.read_csv(csv_path, dtype=str)
    topic_timestamp_map = build_topic_timestamp_map(aligned_data)
    del aligned_data

    if not topic_timestamp_map:
        print(f"No timestamp alignment data available for {rosbag_path.name}, skipping")
        return

    topic_json_path = TOPICS_DIR / f"{rosbag_path.name}.json"
    if not topic_json_path.exists():
        raise FileNotFoundError(f"Missing topic metadata JSON: {topic_json_path}")

    with open(topic_json_path, "r", encoding="utf-8") as f:
        topic_metadata = json.load(f)

    image_topics = [
        topic
        for topic, msg_type in topic_metadata.get("types", {}).items()
        if msg_type in ("sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image")
    ]

    if not image_topics:
        print(f"No image topics found for {rosbag_path.name}, skipping")
        return

    topic_dir_map = {topic: topic_to_directory_name(topic) for topic in image_topics}
    image_topic_set = set(image_topics)

    metadata_counts = load_image_topic_counts_from_metadata(rosbag_path)
    metadata_path = rosbag_path / "metadata.yaml"
    metadata_fingerprint = compute_metadata_fingerprint(metadata_path) if metadata_path.exists() else ""

    metadata_counts, missing_timestamp_map, full_scan_topics, diagnostics, existing_files_map = analyze_missing_messages(
        rosbag_path,
        image_topics,
        output_dir,
        topic_timestamp_map,
        metadata_counts=metadata_counts,
    )

    existing_counts = {topic: len(existing_files_map.get(topic, set())) for topic in image_topics}

    tolerance_notes: dict[str, str] = {}
    tolerated_topics: set[str] = set()
    for topic in image_topics:
        total_missing, expected_count = compute_topic_missing_totals(topic, diagnostics)
        if total_missing <= 0:
            continue

        within_abs = MISSING_IMAGE_ABS_TOLERANCE >= 0 and total_missing <= MISSING_IMAGE_ABS_TOLERANCE
        within_ratio = (
            expected_count is not None
            and expected_count > 0
            and MISSING_IMAGE_RATIO_TOLERANCE > 0.0
            and (total_missing / expected_count) <= MISSING_IMAGE_RATIO_TOLERANCE
        )

        if within_abs or within_ratio:
            tolerated_topics.add(topic)
            reasons: list[str] = []
            if within_abs:
                reasons.append(f"{total_missing} missing <= abs tolerance {MISSING_IMAGE_ABS_TOLERANCE}")
            if within_ratio and expected_count:
                ratio_value = total_missing / expected_count
                reasons.append(f"{ratio_value:.6f} ratio <= {MISSING_IMAGE_RATIO_TOLERANCE:.6f}")
            tolerance_notes[topic] = "; ".join(reasons)

    active_full_scan_topics = {topic for topic in full_scan_topics if topic not in tolerated_topics}
    topics_to_process = (set(missing_timestamp_map.keys()) | active_full_scan_topics) - tolerated_topics

    # Check for existing completion status (processed flag was already checked at start of function)
    if (
        existing_status
        and existing_status.get("state") in COMPLETED_STATUS_STATES
        and existing_status.get("metadata_fingerprint") == metadata_fingerprint
    ):
        matches_expected = True
        for topic in image_topics:
            expected = metadata_counts.get(topic)
            existing_count = existing_counts.get(topic, 0)
            if expected is not None and existing_count < expected and topic not in tolerated_topics:
                matches_expected = False
                break
        if matches_expected and not topics_to_process:
            finalize_bag_status(
                rosbag_name=rosbag_path.name,
                output_dir=output_dir,
                metadata_fingerprint=metadata_fingerprint,
                metadata_counts=metadata_counts,
                image_topics=image_topics,
                existing_files_map=existing_files_map,
                remaining_missing=missing_timestamp_map,
                diagnostics=diagnostics,
                tolerated_topics=tolerated_topics,
            )
            print(f"{rosbag_path.name}: status.json indicates extraction complete — skipping")
            return

    if not topics_to_process:
        finalize_bag_status(
            rosbag_name=rosbag_path.name,
            output_dir=output_dir,
            metadata_fingerprint=metadata_fingerprint,
            metadata_counts=metadata_counts,
            image_topics=image_topics,
            existing_files_map=existing_files_map,
            remaining_missing=missing_timestamp_map,
            diagnostics=diagnostics,
            tolerated_topics=tolerated_topics,
        )
        if tolerated_topics:
            print(f"{rosbag_path.name}: missing counts within tolerance — skipping re-extraction")
        else:
            print(f"{rosbag_path.name}: metadata indicates no missing image messages — skipping")
        return

    print(f"{rosbag_path.name}: metadata summary for image topics with missing data:")
    for topic in image_topics:
        diag = diagnostics.get(topic, {})
        missing_meta = diag.get("missing_vs_metadata")
        identified = diag.get("missing_identified")
        if missing_meta or identified:
            print(
                f"  {topic}: metadata={diag.get('metadata_count')}, "
                f"existing={diag.get('existing_images')}, "
                f"missing_vs_metadata={missing_meta}, "
                f"identified_missing={identified}, "
                f"unmapped_missing={diag.get('missing_unmapped')}"
            )
            if topic in tolerated_topics:
                note = tolerance_notes.get(topic)
                if note:
                    print(f"    -> within tolerance: {note}")
                else:
                    print("    -> within tolerance: skipping re-extraction.")
            elif topic in active_full_scan_topics:
                print("    -> scheduling full scan for this topic (missing data not mapped).")
            else:
                planned = len(missing_timestamp_map.get(topic, set()))
                if planned:
                    print(f"    -> re-extracting {planned} mapped timestamp(s).")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures: list[concurrent.futures.Future] = []
        future_to_key: dict[concurrent.futures.Future, tuple[str, str | None]] = {}
        scheduled_full_scan: dict[str, set[str]] = {topic: set() for topic in active_full_scan_topics}

        # Only process mcap format (db3 removed since all rosbags are mcap)
        if format == 'mcap':
            message_reader = read_mcap_messages(rosbag_path, image_topic_set)
            for topic, msg, timestamp, mcap_file in tqdm(message_reader, desc=f"Extracting {rosbag_path.name}"):
                if topic not in topics_to_process:
                    continue

                timestamp_str = str(timestamp)
                missing_set = missing_timestamp_map.get(topic)
                existing_files = existing_files_map.setdefault(topic, set())

                if missing_set is not None:
                    if timestamp_str not in missing_set:
                        continue
                    missing_set.discard(timestamp_str)
                else:
                    if timestamp_str in existing_files:
                        continue
                    if timestamp_str in scheduled_full_scan.get(topic, set()):
                        continue
                    scheduled_full_scan.setdefault(topic, set()).add(timestamp_str)

                # Create the proper folder structure: rosbag -> topics -> mcaps -> images
                if mcap_file:
                    mcap_folder = mcap_file.replace('.mcap', '')
                    topic_dir = output_dir / topic_dir_map[topic] / mcap_folder
                else:
                    topic_dir = output_dir / topic_dir_map[topic]
                topic_dir.mkdir(parents=True, exist_ok=True)

                # Check for any existing image file (not just .webp) - fast check
                has_existing = False
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
                    if (topic_dir / f"{timestamp_str}{ext}").exists():
                        has_existing = True
                        break
                if has_existing:
                    existing_files.add(timestamp_str)
                    continue

                future = executor.submit(
                    extract_image_from_message,
                    msg,
                    topic,
                    timestamp,
                    output_dir,  # Pass the base output directory
                    mcap_file,   # Pass the MCAP file name
                )
                futures.append(future)
                future_to_key[future] = (topic, timestamp_str)

        for future in futures:
            topic, timestamp_str = future_to_key.get(future, (None, None))
            try:
                future.result()
            except Exception as exc:
                print(f"Error extracting {topic} at {timestamp_str}: {exc}")
                if topic and timestamp_str:
                    if topic in missing_timestamp_map:
                        missing_timestamp_map[topic].add(timestamp_str)
                    else:
                        scheduled_full_scan.setdefault(topic, set()).discard(timestamp_str)
                        existing_files_map.setdefault(topic, set()).discard(timestamp_str)
                continue

            if topic and timestamp_str:
                topic_dir = output_dir / topic_dir_map[topic]
                # Check for any image file with this timestamp - fast check (including mcap subdirs)
                has_existing = False
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
                    # Check direct topic directory
                    if (topic_dir / f"{timestamp_str}{ext}").exists():
                        has_existing = True
                        break
                    # Check mcap subdirectories
                    if not has_existing:
                        for mcap_subdir in topic_dir.iterdir():
                            if mcap_subdir.is_dir() and (mcap_subdir / f"{timestamp_str}{ext}").exists():
                                has_existing = True
                                break
                        if has_existing:
                            break
                if not has_existing:
                    print(f"Warning: expected image {timestamp_str} missing after extraction")
                    if topic in missing_timestamp_map:
                        missing_timestamp_map[topic].add(timestamp_str)
                    else:
                        scheduled_full_scan.setdefault(topic, set()).discard(timestamp_str)
                else:
                    existing_files_map.setdefault(topic, set()).add(timestamp_str)

    remaining_missing = {topic: ts for topic, ts in missing_timestamp_map.items() if ts}
    if remaining_missing:
        total_left = sum(len(ts) for ts in remaining_missing.values())
        print(f"Warning: {rosbag_path.name} still missing {total_left} image(s) with known timestamps")
        for topic, timestamps in remaining_missing.items():
            print(f"  {topic}: {len(timestamps)} timestamp(s) still pending")

    for topic in active_full_scan_topics:
        expected = diagnostics.get(topic, {}).get("metadata_count")
        existing = len(existing_files_map.get(topic, set()))
        missing_meta = diagnostics.get(topic, {}).get("missing_vs_metadata")
        if expected is not None and missing_meta:
            actual_missing = max(expected - existing, 0)
            if actual_missing > 0:
                print(
                    f"Warning: {rosbag_path.name} topic {topic} still missing "
                    f"{actual_missing} image(s) compared to metadata after processing"
                )

    finalize_bag_status(
        rosbag_name=rosbag_path.name,
        output_dir=output_dir,
        metadata_fingerprint=metadata_fingerprint,
        metadata_counts=metadata_counts,
        image_topics=image_topics,
        existing_files_map=existing_files_map,
        remaining_missing=missing_timestamp_map,
        diagnostics=diagnostics,
        tolerated_topics=tolerated_topics,
    )

def iter_rosbag_dirs(base_dir: Path):
    for dirpath, _dirnames, filenames in os.walk(base_dir):
        if "metadata.yaml" not in filenames:
            continue

        bag_dir = Path(dirpath)
        if "EXCLUDED" in str(bag_dir):
            continue

        # If TARGET_ROSBAGS is specified and not empty, only process those rosbags
        if TARGET_ROSBAGS:
            rosbag_name = bag_dir.name
            if rosbag_name not in TARGET_ROSBAGS:
                continue

        yield bag_dir


# Main function to iterate over all rosbags and extract images
def main():
    rosbag_root = ROSBAGS_DIR_NAS
    if not rosbag_root.exists():
        raise FileNotFoundError(f"ROSBAGS_DIR_NAS path does not exist: {rosbag_root}")

    lookup_root = LOOKUP_TABLES_DIR
    images_root = ORIGINAL_IMAGES
    lookup_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    # Log target rosbags configuration
    if TARGET_ROSBAGS:
        print(f"Target rosbags specified: {TARGET_ROSBAGS}")
        print(f"Will only process {len(TARGET_ROSBAGS)} specific rosbag(s)")
    else:
        print("No target rosbags specified - will process all rosbags")

    for bag_dir in iter_rosbag_dirs(rosbag_root):
        rosbag_name = bag_dir.name
        output_dir = images_root / rosbag_name
        csv_path = lookup_root / f"{rosbag_name}.csv"

        bag_format = detect_bag_format(bag_dir)
        if not bag_format:
            print(f"Skipping unknown format: {rosbag_name}")
            continue

        if not csv_path.exists():
            print(f"Skipping {rosbag_name}: missing lookup table CSV at {csv_path}")
            continue

        print(f"Processing rosbag: {rosbag_name}")
        extract_images_from_rosbag(bag_dir, output_dir, csv_path, bag_format)

if __name__ == "__main__":
    main()
