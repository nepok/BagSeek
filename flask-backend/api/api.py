from csv import reader
import json
import os
from pathlib import Path
from rosbags.rosbag2 import Reader, Writer  # type: ignore
from rosbags.typesys import Stores, get_typestore, get_types_from_msg # type: ignore
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file, abort # type: ignore
from flask_cors import CORS  # type: ignore
import logging
import pandas as pd
import struct
import torch
import faiss
from typing import TYPE_CHECKING, cast
from rosbags.interfaces import ConnectionExtRosbag2  # type: ignore
import open_clip
import math
from threading import Thread, Lock
from datetime import datetime
import gc
from collections import defaultdict
from dotenv import load_dotenv
from time import time

# Load environment variables from .env file
PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Flask API for BagSeek: A tool for exploring Rosbag data and using semantic search via CLIP and FAISS to locate safety critical and relevant scenes.
# This API provides endpoints for loading data, searching via CLIP embeddings, exporting segments, and more.

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend (e.g., React)

#
# --- Global Variables and Constants ---
#
# ROSBAGS_DIR: Directory where all Rosbag files are stored
# BASE_DIR: Base directory for backend resources (topics, images, embeddings, etc.)
# TOPICS_DIR: Directory for storing available topics per Rosbag
# IMAGES_DIR: Directory for extracted image frames
# LOOKUP_TABLES_DIR: Directory for lookup tables mapping reference timestamps to topic timestamps
# EMBEDDINGS_DIR: Directory for precomputed CLIP embeddings
# CANVASES_FILE: JSON file for UI canvas state persistence
# INDICES_DIR: Directory for FAISS indices for semantic search
# EXPORT_DIR: Directory where exported Rosbags are saved
# SELECTED_ROSBAG: Currently selected Rosbag file path

ROSBAGS_DIR_MNT = os.getenv("ROSBAGS_DIR_MNT")
ROSBAGS_DIR_NAS = os.getenv("ROSBAGS_DIR_NAS")
BASE_DIR = os.getenv("BASE_DIR")
TOPICS_DIR = os.getenv("TOPICS_DIR")
IMAGES_DIR = os.getenv("IMAGES_DIR")
IMAGES_PER_TOPIC_DIR = os.getenv("IMAGES_PER_TOPIC_DIR")
LOOKUP_TABLES_DIR = os.getenv("LOOKUP_TABLES_DIR")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR")
EMBEDDINGS_PER_TOPIC_DIR = os.getenv("EMBEDDINGS_PER_TOPIC_DIR")
CANVASES_FILE = os.getenv("CANVASES_FILE")
INDICES_DIR = os.getenv("INDICES_DIR")
EXPORT_DIR = os.getenv("EXPORT_DIR")
ADJACENT_SIMILARITIES_DIR = os.getenv("ADJACENT_SIMILARITIES_DIR")
REPRESENTATIVE_IMAGES_DIR = os.getenv("REPRESENTATIVE_IMAGES_DIR")

SELECTED_ROSBAG = 'mnt/data/rosbags/output_bag'

# ALIGNED_DATA: DataFrame mapping reference timestamps to per-topic timestamps for alignment
ALIGNED_DATA = pd.read_csv(os.path.join(LOOKUP_TABLES_DIR, os.path.basename(SELECTED_ROSBAG) + '.csv'), dtype=str)

# EXPORT_PROGRESS: Dictionary to track progress and status of export jobs
EXPORT_PROGRESS = {"status": "idle", "progress": -1}
SEARCH_PROGRESS = {"status": "idle", "progress": -1, "message": "idle"}

# SELECTED_MODEL: Default CLIP model for semantic search (format: <model_name>__<pretrained_name>)
SELECTED_MODEL = 'ViT-B-16-quickgelu__openai'
SEARCHED_ROSBAGS = []

# MAX_K: Number of top results to return for semantic search
MAX_K = 100

# Cache setup for expensive rosbag discovery
FILE_PATH_CACHE_TTL_SECONDS = 60
_file_path_cache = {"paths": [], "timestamp": 0.0}
_file_path_cache_lock = Lock()

# Initialize the type system for ROS2 deserialization, including custom Novatel messages
# This is required to correctly deserialize messages from Rosbags, especially for custom message types
typestore = get_typestore(Stores.ROS2_HUMBLE)
novatel_msg_folder = Path('/opt/ros/humble/share/novatel_oem7_msgs/msg')
custom_types = {}
for msg_file in novatel_msg_folder.glob('*.msg'):
    try:
        text = msg_file.read_text()
        typename = f"novatel_oem7_msgs/msg/{msg_file.stem}"
        result = get_types_from_msg(text, typename)
        custom_types.update(result)
    except Exception as e:
        logging.warning(f"Failed to parse {msg_file.name}: {e}")
typestore.register(custom_types)

# Used to track the currently selected reference timestamp and its aligned mappings
# current_reference_timestamp: The reference timestamp selected by the user
# mapped_timestamps: Dictionary mapping topic names to their corresponding timestamps for the selected reference timestamp
current_reference_timestamp = None
mapped_timestamps = {}

# Endpoint to set the current reference timestamp and retrieve its aligned mappings
@app.route('/api/set-reference-timestamp', methods=['POST'])
def set_reference_timestamp():
    global current_reference_timestamp, mapped_timestamps
    try:
        data = request.get_json()
        referenceTimestamp = data.get('referenceTimestamp')
        
        if not referenceTimestamp:
            return jsonify({"error": "Missing referenceTimestamp"}), 400

        row = ALIGNED_DATA[ALIGNED_DATA["Reference Timestamp"] == str(referenceTimestamp)]
        if row.empty:
            return jsonify({"error": "Reference timestamp not found in CSV"}), 404

        current_reference_timestamp = referenceTimestamp
        mapped_timestamps = row.iloc[0].to_dict()
        # Convert NaNs to None for safe JSON serialization
        cleaned_mapped_timestamps = {
            k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in mapped_timestamps.items()
        }
        return jsonify({"mappedTimestamps": cleaned_mapped_timestamps, "message": "Reference timestamp updated"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Compute normalized CLIP embedding vector for a text query
def get_text_embedding(text, model, tokenizer, device):
    with torch.no_grad():
        tokens = tokenizer([text])
        tokens = tokens.to(device)
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

# Canvas config utility functions (UI state persistence)
def load_canvases():
    if os.path.exists(CANVASES_FILE):
        with open(CANVASES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_canvases(data):
    with open(CANVASES_FILE, "w") as f:
        json.dump(data, f, indent=4)


def _collect_rosbag_paths(force_refresh: bool = False):
    now = time()
    with _file_path_cache_lock:
        cached_paths = list(_file_path_cache["paths"])
        cached_timestamp = _file_path_cache["timestamp"]

    if (
        cached_paths
        and not force_refresh
        and (now - cached_timestamp) < FILE_PATH_CACHE_TTL_SECONDS
    ):
        return cached_paths

    ros_files = []
    if not ROSBAGS_DIR_NAS:
        return ros_files

    for root, dirs, files in os.walk(ROSBAGS_DIR_NAS, topdown=True):
        if 'metadata.yaml' in files:
            if "EXCLUDED" in root:
                dirs[:] = []
                continue
            ros_files.append(root)
            dirs[:] = []

    ros_files.sort()

    with _file_path_cache_lock:
        _file_path_cache["paths"] = ros_files
        _file_path_cache["timestamp"] = time()

    return ros_files

# Copy messages from one Rosbag to another within a timestamp range and selected topics
def export_rosbag_with_topics(src: Path, dst: Path, includedTopics, start_timestamp, end_timestamp) -> None:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(src) as reader, Writer(dst, version=9) as writer:
        conn_map = {}
        all_msgs = list(reader.messages())
        total_msgs = len(all_msgs)
        msg_counter = 0

        EXPORT_PROGRESS["status"] = "starting"
        EXPORT_PROGRESS["progress"] = -1

        for conn in reader.connections:
            ext = cast(ConnectionExtRosbag2, conn.ext)
            if conn.topic in includedTopics:
                conn_map[conn.id] = writer.add_connection(
                    conn.topic,
                    conn.msgtype,
                    typestore=typestore,
                    serialization_format=ext.serialization_format,
                    offered_qos_profiles=ext.offered_qos_profiles,
                )

        EXPORT_PROGRESS["status"] = "running"
        EXPORT_PROGRESS["progress"] = 0.00

        for conn, timestamp, data in reader.messages():
            msg_counter += 1
            current_progress = (msg_counter / total_msgs)
            EXPORT_PROGRESS["progress"] = round(current_progress, 2)
            if conn.id not in conn_map:
                continue

            if not (timestamp >= start_timestamp and timestamp <= end_timestamp):
                continue
            
            msg = typestore.deserialize_cdr(data, conn.msgtype)
            if head := getattr(msg, 'header', None):
                headstamp = head.stamp.sec * 10**9 + head.stamp.nanosec
                head.stamp.sec = headstamp // 10**9
                head.stamp.nanosec = headstamp % 10**9
                outdata = typestore.serialize_cdr(msg, conn.msgtype)
            else:
                outdata = data

            writer.write(conn_map[conn.id], timestamp, outdata)

        EXPORT_PROGRESS["status"] = "done"
        EXPORT_PROGRESS["progress"] = 1.0

# Return the current export progress and status
@app.route('/api/export-status', methods=['GET'])
def get_export_status():
    return jsonify(EXPORT_PROGRESS)

# Set the model for performing semantic search
@app.route('/api/set-model', methods=['POST'])
def post_model():
    """
    Set the current CLIP model for semantic search.
    """
    try:
        data = request.get_json()  # Get the JSON payload
        model_value = data.get('model')  # The path value from the JSON

        global SELECTED_MODEL
        SELECTED_MODEL = model_value

        # Model reload is handled on demand in the search endpoint
        return jsonify({"message": f"Model {model_value} successfully posted."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Get all available models for semantisch searching    
@app.route('/api/get-models', methods=['GET'])
def get_models():
    """
    List available CLIP models (based on embeddings directory).
    """
    try:
        models = []
        for model in os.listdir(EMBEDDINGS_PER_TOPIC_DIR):
            if not model in ['.DS_Store', 'README.md']:
                models.append(model)
        return jsonify({"models": models}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500

# Set the correct file paths for the selected rosbag and the lookup table after selecting a rosbag
@app.route('/api/set-file-paths', methods=['POST'])
def post_file_paths():
    """
    Set the current Rosbag file and update alignment data accordingly.
    """
    try:
        data = request.get_json()  # Get the JSON payload
        path_value = data.get('path')  # The path value from the JSON

        global SELECTED_ROSBAG
        SELECTED_ROSBAG = path_value

        global ALIGNED_DATA
        csv_path = LOOKUP_TABLES_DIR + "/" + os.path.basename(SELECTED_ROSBAG) + ".csv"
        ALIGNED_DATA = pd.read_csv(csv_path, dtype=str)

        global SEARCHED_ROSBAGS
        SEARCHED_ROSBAGS = [path_value]  # Reset searched rosbags to the selected one
        logging.warning(SEARCHED_ROSBAGS)
        return jsonify({"message": f"File path updated successfully to {path_value}."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Get all available rosbags
@app.route('/api/get-file-paths', methods=['GET'])
def get_file_paths():
    """
    Return all available Rosbag file paths (excluding those in EXCLUDED).
    """
    try:
        ros_files = _collect_rosbag_paths()
        return jsonify({"paths": ros_files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/refresh-file-paths', methods=['POST'])
def refresh_file_paths():
    """Force refresh of the cached Rosbag file paths."""
    try:
        ros_files = _collect_rosbag_paths(force_refresh=True)
        return jsonify({"paths": ros_files, "message": "Rosbag file paths refreshed."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Get the currently selected rosbag
@app.route('/api/get-selected-rosbag', methods=['GET'])
def get_selected_rosbag():
    """
    Return the currently selected Rosbag file name.
    """
    try:
        selectedRosbag = os.path.basename(SELECTED_ROSBAG)
        return jsonify({"selectedRosbag": selectedRosbag}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500

@app.route('/api/set-searched-rosbags', methods=['POST'])
def set_searched_rosbags():
    """
    Set the currently searched Rosbag file name.
    This is used to filter available topics and images based on the selected Rosbag.
    """
    try:
        data = request.get_json()  # Get the JSON payload
        searchedRosbags = data.get('searchedRosbags')  # The list of searched Rosbags

        if not isinstance(searchedRosbags, list):
            return jsonify({"error": "searchedRosbags must be a list"}), 400

        global SEARCHED_ROSBAGS
        SEARCHED_ROSBAGS = searchedRosbags

        return jsonify({"message": "Searched Rosbags updated successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500 
    
# Get available topics from pre-generated JSON file for the current Rosbag
@app.route('/api/get-available-topics', methods=['GET'])
def get_available_rosbag_topics():
    try:
        rosbag_name = os.path.basename(SELECTED_ROSBAG)
        topics_json_path = os.path.join(TOPICS_DIR, f"{rosbag_name}.json")

        if not os.path.exists(topics_json_path):
            return jsonify({'availableTopics': []}), 200

        with open(topics_json_path, 'r') as f:
            topics_data = json.load(f)

        topics = topics_data.get("topics", [])
        return jsonify({'availableTopics': topics}), 200

    except Exception as e:
        logging.error(f"Error reading topics JSON: {e}")
        return jsonify({'availableTopics': []}), 200
    

@app.route('/api/get-available-image-topics', methods=['GET'])
def get_available_image_topics():
    try:
        model_params = request.args.getlist("models")
        rosbag_params = request.args.getlist("rosbags")

        if not model_params or not rosbag_params:
            return jsonify({'availableTopics': {}}), 200

        results = {}

        for model_param in model_params:
            model_path = os.path.join(ADJACENT_SIMILARITIES_DIR, model_param)
            if not os.path.isdir(model_path):
                continue

            model_entry = {}
            for rosbag_param in rosbag_params:
                rosbag_name = os.path.basename(rosbag_param)
                rosbag_path = os.path.join(model_path, rosbag_name)
                if not os.path.isdir(rosbag_path):
                    continue

                topics = []
                for topic in os.listdir(rosbag_path):
                    topics.append(topic.replace("__", "/"))

                model_entry[rosbag_name] = sorted(topics)

            if model_entry:
                results[model_param] = model_entry

        return jsonify({'availableTopics': results}), 200

    except Exception as e:
        logging.error(f"Error scanning adjacent similarities: {e}")
        return jsonify({'availableTopics': {}}), 200

# Read list of topic types from generated JSON for current Rosbag
@app.route('/api/get-available-topic-types', methods=['GET'])
def get_available_rosbag_topic_types():
    try:
        rosbag_name = os.path.basename(SELECTED_ROSBAG)
        topics_json_path = os.path.join(TOPICS_DIR, f"{rosbag_name}.json")

        if not os.path.exists(topics_json_path):
            return jsonify({'availableTopicTypes': []}), 200

        with open(topics_json_path, 'r') as f:
            topics_data = json.load(f)

        availableTopicTypes = topics_data.get("types", [])
        return jsonify({'availableTopicTypes': availableTopicTypes}), 200

    except Exception as e:
        logging.error(f"Error reading topics JSON: {e}")
        return jsonify({'availableTopicTypes': []}), 200
    
# Returns list of all reference timestamps used for data alignment
@app.route('/api/get-available-timestamps', methods=['GET'])
def get_available_timestamps():
    availableTimestamps = ALIGNED_DATA['Reference Timestamp'].astype(str).tolist()
    return jsonify({'availableTimestamps': availableTimestamps}), 200

@app.route('/api/get-timestamp-lengths', methods=['GET'])
def get_timestamp_lengths():
    rosbags = request.args.getlist("rosbags")
    timestampLengths = {}

    for rosbag in rosbags:
        csv_path = os.path.join(LOOKUP_TABLES_DIR, os.path.basename(rosbag) + '.csv')
        try:
            df = pd.read_csv(csv_path, dtype=str)
            count = df['Reference Timestamp'].notnull().sum()
            timestampLengths[rosbag] = int(count)
        except Exception as e:
            timestampLengths[rosbag] = f"Error: {str(e)}"

    return jsonify({'timestampLengths': timestampLengths})

# Returns density of valid data fields per timestamp (used for heatmap intensity visualization)
@app.route('/api/get-timestamp-density', methods=['GET'])
def get_timestamp_density():
    density_array = ALIGNED_DATA.drop(columns=["Reference Timestamp"]).notnull().sum(axis=1).tolist()
    return jsonify({'timestampDensity': density_array})
    
@app.route('/images/<path:image_path>')
def serve_image(image_path):
    """
    Serve an extracted image file from the backend image directory.
    """
    response = send_from_directory(IMAGES_PER_TOPIC_DIR, image_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

@app.route('/adjacency-image/<path:adjacency_image_path>')
def serve_adjacency_image(adjacency_image_path):
    """
    Serve an extracted image file from the backend image directory.
    """
    response = send_from_directory(ADJACENT_SIMILARITIES_DIR, adjacency_image_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

@app.route('/representative-image/<path:representative_image_path>')
def serve_representative_image(representative_image_path):
    """
    Serve an extracted image file from the backend image directory.
    """
    response = send_from_directory(REPRESENTATIVE_IMAGES_DIR, representative_image_path)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

# Return preview image URL for a topic at a given reference index or position within a rosbag
@app.route('/api/get-topic-image-preview', methods=['GET'])
def get_topic_image_preview():
    try:
        rosbag_name = request.args.get('rosbag', type=str)
        topic = request.args.get('topic', type=str)
        index = request.args.get('index', type=int)
        pos = request.args.get('pos', type=float)

        if not rosbag_name or not topic:
            return jsonify({'error': 'Missing rosbag or topic'}), 400

        csv_path = os.path.join(LOOKUP_TABLES_DIR, f"{rosbag_name}.csv")
        if not os.path.exists(csv_path):
            return jsonify({'error': 'Lookup table not found'}), 404

        df = pd.read_csv(csv_path, dtype=str)
        total = len(df)
        if total == 0:
            return jsonify({'error': 'No timestamps available'}), 404

        # Determine reference index from pos or index
        if index is None:
            if pos is None:
                index = 0
            else:
                index = int(max(0, min(total - 1, round(pos * (total - 1)))))
        else:
            index = int(max(0, min(total - 1, index)))

        # Find the per-topic timestamp at or near the requested reference index
        def non_nan(v):
            if v is None:
                return False
            if isinstance(v, float):
                return not math.isnan(v)
            s = str(v)
            return s.lower() != 'nan' and s != '' and s != 'None'

        chosen_idx = index
        topic_ts = None
        if topic in df.columns:
            val = df.iloc[index][topic]
            if non_nan(val):
                topic_ts = str(val)
            else:
                # search nearest non-empty within a window
                radius = min(500, total - 1)
                for off in range(1, radius + 1):
                    left = index - off
                    right = index + off
                    if left >= 0:
                        v = df.iloc[left][topic]
                        if non_nan(v):
                            chosen_idx = left
                            topic_ts = str(v)
                            break
                    if right < total:
                        v = df.iloc[right][topic]
                        if non_nan(v):
                            chosen_idx = right
                            topic_ts = str(v)
                            break
        else:
            return jsonify({'error': 'Topic column not found'}), 404

        if not topic_ts:
            return jsonify({'error': 'No valid topic timestamp found nearby'}), 404

        topic_safe = topic.replace('/', '__')
        filename = f"{topic_safe}-{topic_ts}.webp"
        file_path = os.path.join(IMAGES_DIR, rosbag_name, filename)

        # If file missing, search outward for a nearby timestamp with existing image
        if not os.path.exists(file_path):
            # try nearby indices on topic column
            radius = min(500, total - 1)
            best_path = None
            best_idx = chosen_idx
            for off in range(1, radius + 1):
                for candidate in (chosen_idx - off, chosen_idx + off):
                    if 0 <= candidate < total:
                        v = df.iloc[candidate][topic]
                        if non_nan(v):
                            fp = os.path.join(IMAGES_DIR, rosbag_name, f"{topic_safe}-{str(v)}.webp")
                            if os.path.exists(fp):
                                best_path = fp
                                best_idx = candidate
                                topic_ts = str(v)
                                break
                if best_path:
                    break
            if best_path:
                file_path = best_path
                chosen_idx = best_idx
            else:
                return jsonify({'error': 'Image not found for nearby timestamps'}), 404

        rel_url = f"/images/{rosbag_name}/{os.path.basename(file_path)}"
        return jsonify({'imageUrl': rel_url, 'timestamp': topic_ts, 'index': chosen_idx, 'total': total}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New endpoint to serve image reference map JSON file for a given rosbag
@app.route('/image-reference-map/<rosbag_name>.json', methods=['GET'])
def get_image_reference_map(rosbag_name):
    map_path = os.path.join(BASE_DIR, "image_reference_maps", f"{rosbag_name}.json")
    if not os.path.exists(map_path):
        return jsonify({"error": f"Image reference map for {rosbag_name} not found."}), 404
    return send_file(map_path, mimetype='application/json')

# Return deserialized message content (image, TF, IMU, etc.) for currently selected topic and reference timestamp
@app.route('/api/content', methods=['GET'])
def get_ros():
    """
    Return deserialized message content (image, TF, IMU, etc.) for the currently selected topic and reference timestamp.
    The logic:
      - Uses the mapped_timestamps for the current reference timestamp to get the aligned topic timestamp.
      - Opens the Rosbag and iterates messages for the topic, looking for the exact timestamp.
      - Deserializes the message and returns a JSON structure based on the message type (point cloud, position, tf, imu, etc).
    """
    global mapped_timestamps
    topic = request.args.get('topic', default=None, type=str)
    timestamp = mapped_timestamps.get(topic) if topic else None

    if not timestamp:
        return jsonify({'error': 'No mapped timestamp found for the provided topic'})

    # Open the rosbag to find the message at the requested timestamp and topic
    with Reader(SELECTED_ROSBAG) as reader:
        connections = [x for x in reader.connections if x.topic == topic]

        for connection, msg_timestamp, rawdata in reader.messages(connections=connections):
            if str(msg_timestamp) == timestamp:
                # Deserialize the message based on the connection's message type
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                match connection.msgtype:
                    case 'sensor_msgs/msg/PointCloud2':
                        # Extract point cloud data, filtering out NaNs, Infs, and zeros
                        pointCloud = []
                        point_step = msg.point_step
                        for i in range(0, len(msg.data), point_step):
                            x, y, z = struct.unpack_from('fff', msg.data, i)
                            if all(np.isfinite([x, y, z])) and not (x == 0 and y == 0 and z == 0):
                                pointCloud.extend([x, y, z])
                        return jsonify({'type': 'pointCloud', 'pointCloud': pointCloud, 'timestamp': timestamp})
                    case 'sensor_msgs/msg/NavSatFix':
                        return jsonify({'type': 'position', 'position': {'latitude': msg.latitude, 'longitude': msg.longitude, 'altitude': msg.altitude}, 'timestamp': timestamp})
                    case 'novatel_oem7_msgs/msg/BESTPOS':
                        return jsonify({'type': 'position', 'position': {'latitude': msg.lat, 'longitude': msg.lon, 'altitude': msg.hgt}, 'timestamp': timestamp})
                    case 'tf2_msgs/msg/TFMessage':
                        # Assume single transform for simplicity
                        if len(msg.transforms) > 0:
                            transform = msg.transforms[0]
                            translation = transform.transform.translation
                            rotation = transform.transform.rotation
                            tf_data = {
                                'translation': {
                                    'x': translation.x,
                                    'y': translation.y,
                                    'z': translation.z
                                },
                                'rotation': {
                                    'x': rotation.x,
                                    'y': rotation.y,
                                    'z': rotation.z,
                                    'w': rotation.w
                                }
                            }
                            return jsonify({'type': 'tf', 'tf': tf_data, 'timestamp': timestamp})
                    case 'sensor_msgs/msg/Imu':
                        imu_data = {
                            "orientation": {
                                "x": msg.orientation.x,
                                "y": msg.orientation.y,
                                "z": msg.orientation.z,
                                "w": msg.orientation.w
                            },
                            "angular_velocity": {
                                "x": msg.angular_velocity.x,
                                "y": msg.angular_velocity.y,
                                "z": msg.angular_velocity.z
                            },
                            "linear_acceleration": {
                                "x": msg.linear_acceleration.x,
                                "y": msg.linear_acceleration.y,
                                "z": msg.linear_acceleration.z
                            }
                        }
                        return jsonify({'type': 'imu', 'imu': imu_data, 'timestamp': timestamp})
                    case _:
                        # Fallback for unsupported or unknown message types: return string representation
                        return jsonify({'type': 'text', 'text': str(msg), 'timestamp': timestamp})

    return jsonify({'error': 'No message found for the provided timestamp and topic'})

@app.route('/api/search-status', methods=['GET'])
def get_search_status():
    return jsonify(SEARCH_PROGRESS)

@app.route('/api/search-new', methods=['GET'])
def search_new():
    SEARCH_PROGRESS["status"] = "running"
    SEARCH_PROGRESS["progress"] = 0.00

    query_text = request.args.get('query', default=None, type=str)
    models = request.args.get('models', default=None, type=str)
    rosbags = request.args.get('rosbags', default=None, type=str)
    timeRange = request.args.get('timeRange', default=None, type=str)
    accuracy = request.args.get('accuracy', default=None, type=int)

    SEARCH_PROGRESS["message"] = f"Starting search...\n\n(sampling every {accuracy}th embedding)"

    if query_text is None:
        return jsonify({'error': 'No query text provided'}), 400
    if models is None:
        return jsonify({'error': 'No models provided'}), 400
    if rosbags is None:
        return jsonify({'error': 'No rosbags provided'}), 400
    if timeRange is None:
        return jsonify({'error': 'No time range provided'}), 400
    if accuracy is None:
        return jsonify({'error': 'No accuracy provided'}), 400

    models = models.split(",")
    rosbags = rosbags.split(",")
    rosbags = [os.path.basename(r) for r in rosbags]
    time_start, time_end = map(int, timeRange.split(","))

    # marks is now a dict: {(model, rosbag, topic): set(indices)}
    marks = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = []

    try:

        total_steps = 0
        for model in models:
            model_path = os.path.join(EMBEDDINGS_PER_TOPIC_DIR, model)
            if not os.path.isdir(model_path):
                continue
            for rosbag in rosbags:
                rosbag_path = os.path.join(model_path, rosbag)
                if not os.path.isdir(rosbag_path):
                    continue
                for topic in os.listdir(rosbag_path):
                    topic_path = os.path.join(rosbag_path, topic)
                    if os.path.isdir(topic_path):
                        total_steps += 1

        #total_steps = len(models) * len(rosbags)
        #logging.warning(f"Total steps to process: {total_steps}")
        step_count = 0

        for model_idx, model_name in enumerate(models):
            name, pretrained = model_name.split('__')
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained, device=device, cache_dir="/mnt/data/openclip_cache")
            tokenizer = open_clip.get_tokenizer(name)
            query_embedding = get_text_embedding(query_text, model, tokenizer, device)
            #logging.warning(np.linalg.norm(query_embedding))

            model_dir = os.path.join(EMBEDDINGS_PER_TOPIC_DIR, model_name)
            if not os.path.isdir(model_dir):
                del model
                torch.cuda.empty_cache()
                continue

            # For each rosbag, iterate over topic folders
            for rosbag_name in rosbags:
                rosbag_dir = os.path.join(model_dir, rosbag_name)
                if not os.path.isdir(rosbag_dir):
                    step_count += 1
                    continue

                # --- Load aligned CSV for this rosbag_name ---
                aligned_path = os.path.join(LOOKUP_TABLES_DIR, rosbag_name + ".csv")
                aligned_data = pd.read_csv(aligned_path, dtype=str)

                # List all topic folders in this rosbag_dir
                topic_folders = [t for t in os.listdir(rosbag_dir) if os.path.isdir(os.path.join(rosbag_dir, t))]
                for topic_folder in topic_folders:
                    topic_name = topic_folder.replace("__", "/")
                    topic_dir = os.path.join(rosbag_dir, topic_folder)
                    SEARCH_PROGRESS["status"] = "running"
                    SEARCH_PROGRESS["progress"] = round((step_count / total_steps) * 0.95, 2)
                    SEARCH_PROGRESS["message"] = (
                        f"Loading embeddings...\n\n"
                        f"Model: {model_name}\n"
                        f"Rosbag: {rosbag_name}\n"
                        f"Topic: {topic_name}\n\n"
                        f"(Searching every {accuracy}th embedding)"
                    )
                    step_count += 1
                    embedding_files = []
                    for file in os.listdir(topic_dir):
                        if file.endswith("_embedding.pt"):
                            try:
                                ts_part = file.rsplit("-", 1)[-1].replace("_embedding.pt", "")
                                timestamp = int(ts_part)
                                dt = datetime.utcfromtimestamp(timestamp / 1e9)
                                minute_of_day = dt.hour * 60 + dt.minute
                                if time_start <= minute_of_day <= time_end:
                                    emb_path = os.path.join(topic_dir, file)
                                    embedding_files.append(emb_path)
                            except Exception as e:
                                continue
                    selected_paths = embedding_files[::accuracy]
                    model_embeddings = []
                    model_paths = []
                    for emb_path in selected_paths:
                        try:
                            embedding = torch.load(emb_path, map_location='cpu')
                            if isinstance(embedding, torch.Tensor):
                                embedding = embedding.cpu().numpy()
                            model_embeddings.append(embedding)
                            model_paths.append(emb_path)
                        except Exception as e:
                            continue
                    if not model_embeddings:
                        continue
                    embeddings = np.vstack(model_embeddings).astype('float32')
                    n, d = embeddings.shape
                    index = faiss.IndexFlatL2(d)
                    SEARCH_PROGRESS["status"] = "running"
                    SEARCH_PROGRESS["progress"] = round((step_count / total_steps) * 0.95, 2)
                    SEARCH_PROGRESS["message"] = (
                        f"Creating FAISS index using {index.__class__.__name__}) "
                        f"from {len(model_embeddings)} sampled embeddings (every {accuracy}th)."
                    )
                    index.add(embeddings)
                    embedding_paths = np.array(model_paths)
                    SEARCH_PROGRESS["status"] = "running"
                    SEARCH_PROGRESS["progress"] = round((step_count / total_steps) * 0.95, 2)
                    SEARCH_PROGRESS["message"] = (
                        f"Searching {len(model_embeddings)} sampled embeddings...\n\n"
                        f"Model: {model_name}\n"
                        f"Rosbag: {rosbag_name}\n"
                        f"Topic: {topic_folder}\n"
                        f"Index: {index.__class__.__name__}\n\n"
                        f"(Searching every {accuracy}th embedding)"
                    )
                    similarityScores, indices = index.search(query_embedding.reshape(1, -1), MAX_K)
                    if len(indices) == 0 or len(similarityScores) == 0:
                        continue
                    model_results = []
                    topic_name = topic_folder.replace("__", "/")
                    for i, idx in enumerate(indices[0][:MAX_K]):
                        similarityScore = float(similarityScores[0][i])
                        if math.isnan(similarityScore) or math.isinf(similarityScore):
                            continue
                        embedding_path = str(embedding_paths[idx])
                        path_of_interest = str(os.path.basename(embedding_path))
                        # result_timestamp from filename
                        result_timestamp = path_of_interest[-32:-13]
                        dt_result = datetime.utcfromtimestamp(int(result_timestamp) / 1e9)
                        minute_of_day = dt_result.strftime("%H:%M")
                        model_results.append({
                            'rank': i + 1,
                            'rosbag': rosbag_name,
                            'embedding_path': embedding_path,
                            'similarityScore': similarityScore,
                            'topic': topic_name,
                            'timestamp': result_timestamp,
                            'minuteOfDay': minute_of_day,
                            'model': model_name
                        })

                        matching_reference_timestamps = aligned_data.loc[
                            aligned_data.isin([result_timestamp]).any(axis=1),
                            'Reference Timestamp'
                        ].tolist()
                        match_indices = []
                        for ref_ts in matching_reference_timestamps:
                            indices_ = aligned_data.index[aligned_data['Reference Timestamp'] == ref_ts].tolist()
                            match_indices.extend(indices_)
                        key = (model_name, rosbag_name, topic_name)
                        if key not in marks:
                            marks[key] = set()
                        for index_val in match_indices:
                            marks[key].add(index_val)
                    all_results.extend(model_results)
            del model
            torch.cuda.empty_cache()
    except Exception as e:
        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()
        SEARCH_PROGRESS["status"] = "error"
        SEARCH_PROGRESS["progress"] = 0.0
        SEARCH_PROGRESS["message"] = f"Error: {e}"
        return jsonify({'error': str(e)}), 500

    # Flatten and sort all results
    all_results = sorted([r for r in all_results if isinstance(r, dict)], key=lambda x: x['similarityScore'])
    for rank, result in enumerate(all_results, 1):
        result['rank'] = rank
    filtered_results = all_results

    # --- Construct categorizedSearchResults ---
    categorizedSearchResults = {}
    for result in all_results:
        model = result['model']
        rosbag = result['rosbag']
        topic = result['topic']
        minute_of_day = result['minuteOfDay']
        rank = result['rank']
        similarity_score = result['similarityScore']
        timestamp = result['timestamp']
        categorizedSearchResults.setdefault(model, {}).setdefault(rosbag, {}).setdefault(topic, {
            'marks': [],
            'results': []
        })
        categorizedSearchResults[model][rosbag][topic]['results'].append({
            'minuteOfDay': minute_of_day,
            'rank': rank,
            'similarityScore': similarity_score
        })

    # Populate marks per topic only once per mark
    for key, indices in marks.items():
        model, rosbag, topic = key
        if model in categorizedSearchResults and rosbag in categorizedSearchResults[model] and topic in categorizedSearchResults[model][rosbag]:
            for index_val in indices:
                categorizedSearchResults[model][rosbag][topic]['marks'].append({'value': index_val})
    # Flatten marks for response (for compatibility)
    flat_marks = [{'value': idx} for indices in marks.values() for idx in indices]
    SEARCH_PROGRESS["status"] = "done"
    return jsonify({'query': query_text, 'results': filtered_results, 'marks': flat_marks, 'categorizedSearchResults': categorizedSearchResults})

# Perform semantic search using CLIP embedding against precomputed image features
@app.route('/api/search', methods=['GET'])
def search():
    """
    Perform semantic search using CLIP embedding against precomputed image features.
    Logic:
      - Loads the selected CLIP model and tokenizer.
      - Computes query embedding for the input query text.
      - Loads the FAISS index and embedding paths for the selected model and Rosbag.
      - If multiple searched rosbags are provided, creates a temporary FAISS index from sampled embeddings.
      - Searches the index for the top MAX_K results.
      - For each result:
          - Extracts the topic and timestamp from the embedding path.
          - Appends search results, including similarity score and model.
          - For each result timestamp, finds all reference timestamps in ALIGNED_DATA that map to it (used for UI marks).
    """
    query_text = request.args.get('query', default=None, type=str)
    models = request.args.get('models', default=None, type=str)
    rosbags = request.args.get('rosbags', default=None, type=str)
    timeRange = request.args.get('timeRange', default=None, type=str)
    accuracy = request.args.get('accuracy', default=None, type=str)

    if query_text is None:
        return jsonify({'error': 'No query text provided'}), 400
    """if models is None:
        return jsonify({'error': 'No models provided'}), 400
    if rosbags is None:
        return jsonify({'error': 'No rosbags provided'}), 400
    if timeRange is None:
        return jsonify({'error': 'No time range provided'}), 400
    if accuracy is None:
        return jsonify({'error': 'No accuracy provided'}), 400
"""

    # Insert logic for multi-rosbag search using sampled embeddings
    results = []
    marks = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Load model, tokenizer, and compute query embedding
        name, pretrained = SELECTED_MODEL.split('__')
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained, device=device, cache_dir="/mnt/data/openclip_cache")
        tokenizer = open_clip.get_tokenizer(name)
        query_embedding = get_text_embedding(query_text, model, tokenizer, device)

        # --- Refactored searchedRosbags logic ---
        searched_rosbags = SEARCHED_ROSBAGS
        if not searched_rosbags:
            return jsonify({"error": "No rosbags selected"}), 400

        if len(searched_rosbags) == 1:
            rosbag_name = os.path.basename(searched_rosbags[0]).replace('.db3', '')
            subdir = f"{name}__{pretrained}"
            index_path = os.path.join(INDICES_DIR, subdir, rosbag_name, "faiss_index.index")
            embedding_paths_file = os.path.join(INDICES_DIR, subdir, rosbag_name, "embedding_paths.npy")

            if not os.path.exists(index_path) or not os.path.exists(embedding_paths_file):
                del model
                torch.cuda.empty_cache()
                return jsonify({"error": "Index or embedding paths not found"}), 404

            index = faiss.read_index(index_path)
            embedding_paths = np.load(embedding_paths_file)

        else:
            all_embeddings = []
            all_paths = []

            for rosbag_path in searched_rosbags:
                rosbag = os.path.basename(rosbag_path)
                subdir = f"{name}__{pretrained}"
                embedding_folder_path = os.path.join(EMBEDDINGS_DIR, subdir, rosbag)

                for root, _, files in os.walk(embedding_folder_path):
                    for file in files:
                        if file.lower().endswith('_embedding.pt'):
                            input_file_path = os.path.join(root, file)
                            try:
                                embedding = torch.load(input_file_path, weights_only=True)
                                all_embeddings.append(embedding.cpu().numpy())
                                all_paths.append(input_file_path)
                            except Exception as e:
                                print(f"Error loading {input_file_path}: {e}")

            if not all_embeddings:
                del model
                torch.cuda.empty_cache()
                return jsonify({"error": "No embeddings found for selected rosbags"}), 404

            stacked_embeddings = np.vstack(all_embeddings[::10]).astype('float32')
            index = faiss.IndexFlatL2(stacked_embeddings.shape[1])
            index.add(stacked_embeddings)
            embedding_paths = np.array(all_paths[::10])

        # (Removed obsolete loading of FAISS index and embedding paths here)

        # Perform nearest neighbor search in the embedding space
        similarityScores, indices = index.search(query_embedding.reshape(1, -1), MAX_K)
        if len(indices) == 0 or len(similarityScores) == 0:
            del model
            torch.cuda.empty_cache()
            return jsonify({'error': 'No results found for the query'}), 200

        # For each result, extract topic/timestamp and match back to reference timestamps for UI highlighting
        for i, idx in enumerate(indices[0][:MAX_K]):
            similarityScore = float(similarityScores[0][i])
            if math.isnan(similarityScore) or math.isinf(similarityScore):
                continue
            embedding_path = str(embedding_paths[idx])
            path_of_interest = str(os.path.basename(embedding_path))
            relative_path = os.path.relpath(embedding_path, EMBEDDINGS_DIR)
            rosbag_name = os.path.basename(os.path.dirname(relative_path))
            result_timestamp = path_of_interest[-32:-13]
            result_topic = path_of_interest[:-33].replace("__", "/")
            results.append({
                'rank': i + 1,
                'rosbag': rosbag_name,
                'embedding_path': embedding_path,
                'similarityScore': similarityScore,
                'topic': result_topic,
                'timestamp': result_timestamp,
                'model': SELECTED_MODEL
            })

            if rosbag_name != os.path.basename(SELECTED_ROSBAG):
                # If the result is from a different Rosbag, skip alignment
                continue
            # For each result, find all reference timestamps that align to this result timestamp (for UI marks)
            matching_reference_timestamps = ALIGNED_DATA.loc[
                ALIGNED_DATA.isin([result_timestamp]).any(axis=1),
                'Reference Timestamp'
            ].tolist()
            match_indices = []
            for ref_ts in matching_reference_timestamps:
                indices = ALIGNED_DATA.index[ALIGNED_DATA['Reference Timestamp'] == ref_ts].tolist()
                match_indices.extend(indices)
            for index in match_indices:
                marks.append({'value': index})

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()
        return jsonify({'error': str(e)}), 500

    return jsonify({'query': query_text, 'results': results, 'marks': marks})


# Export portion of a Rosbag containing selected topics and time range
@app.route('/api/export-rosbag', methods=['POST'])
def export_rosbag():
    """
    Export a portion of a Rosbag containing selected topics and time range.
    Starts a background thread to perform the export and updates EXPORT_PROGRESS.
    """
    try:
        data = request.json
        new_rosbag_name = data.get('new_rosbag_name')
        topics = data.get('topics')
        start_timestamp = int(data.get('start_timestamp'))
        end_timestamp = int(data.get('end_timestamp'))

        if not new_rosbag_name or not topics:
            return jsonify({"error": "Rosbag name and topics are required"}), 400

        if not os.path.exists(SELECTED_ROSBAG):
            return jsonify({"error": "Rosbag not found"}), 404

        EXPORT_PATH = os.path.join(EXPORT_DIR, new_rosbag_name)
        EXPORT_PROGRESS["status"] = "starting"
        EXPORT_PROGRESS["progress"] = -1

        def run_export():
            export_rosbag_with_topics(SELECTED_ROSBAG, EXPORT_PATH, topics, start_timestamp, end_timestamp)

        Thread(target=run_export).start()
        return jsonify({"message": "Export started", "exported_path": str(EXPORT_PATH)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Canvas configuration endpoints for restoring panel layouts
@app.route("/api/load-canvases", methods=["GET"])
def api_load_canvases():
    return jsonify(load_canvases())

# Canvas configuration endpoints for saving panel layouts
@app.route("/api/save-canvas", methods=["POST"])
def api_save_canvas():
    data = request.json
    name = data.get("name")
    canvas_data = data.get("canvas")
    
    # Load existing canvases
    canvases = load_canvases()
    
    # Update or add the single canvas
    canvases[name] = canvas_data
    
    # Save back to file
    save_canvases(canvases)
    
    return jsonify({"message": f"Canvas '{name}' saved successfully"})

# Canvas configuration endpoints for deleting panel layouts
@app.route("/api/delete-canvas", methods=["POST"])
def api_delete_canvas():
    data = request.json
    name = data.get("name")
    canvases = load_canvases()

    if name in canvases:
        del canvases[name]
        save_canvases(canvases)
        return jsonify({"message": f"Canvas '{name}' deleted successfully"})
    
    return jsonify({"error": "Canvas not found"}), 404

# Start Flask server (debug mode enabled for development)
# Note: debug=True is for local development only. In production, use a WSGI server and set debug=False.
if __name__ == '__main__':
    app.run(debug=True)
