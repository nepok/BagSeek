from csv import reader
import json
import os
from pathlib import Path
from rosbags.rosbag2 import Reader, Writer  # type: ignore
from rosbags.typesys import Stores, get_typestore, get_types_from_msg # type: ignore
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file # type: ignore
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
from threading import Thread

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
ROSBAGS_DIR = '/mnt/data/rosbags'
BASE_DIR = '/mnt/data/bagseek/flask-backend/src'
TOPICS_DIR = os.path.join(BASE_DIR, "topics")
IMAGES_DIR = os.path.join(BASE_DIR, 'extracted_images')
LOOKUP_TABLES_DIR = os.path.join(BASE_DIR, 'lookup_tables')
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings')
CANVASES_FILE = os.path.join(BASE_DIR, 'canvases.json')
INDICES_DIR = os.path.join(BASE_DIR, 'faiss_indices')
EXPORT_DIR = os.path.join(ROSBAGS_DIR, 'EXPORTED')
SELECTED_ROSBAG = os.path.join(ROSBAGS_DIR, 'output_bag')

# ALIGNED_DATA: DataFrame mapping reference timestamps to per-topic timestamps for alignment
ALIGNED_DATA = pd.read_csv(os.path.join(LOOKUP_TABLES_DIR, os.path.basename(SELECTED_ROSBAG) + '.csv'), dtype=str)

# EXPORT_PROGRESS: Dictionary to track progress and status of export jobs
EXPORT_PROGRESS = {"status": "idle", "progress": -1}

# SELECTED_MODEL: Default CLIP model for semantic search (format: <model_name>__<pretrained_name>)
SELECTED_MODEL = 'ViT-B-16-quickgelu__openai'

# MAX_K: Number of top results to return for semantic search
MAX_K = 100

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
        for model in os.listdir(EMBEDDINGS_DIR):
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
        ros_files = []
        for root, dirs, files in os.walk(ROSBAGS_DIR):
            if 'metadata.yaml' in files:
                rosbag_path = os.path.dirname(os.path.join(root, 'metadata.yaml'))
                if "EXCLUDED" in rosbag_path:
                    continue
                ros_files.append(rosbag_path)
        return jsonify({"paths": ros_files}), 200
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

# Returns density of valid data fields per timestamp (used for heatmap intensity visualization)
@app.route('/api/get-timestamp-density', methods=['GET'])
def get_timestamp_density():
    density_array = ALIGNED_DATA.drop(columns=["Reference Timestamp"]).notnull().sum(axis=1).tolist()
    return jsonify({'timestampDensity': density_array})
    
@app.route('/images/<path:filename>')
def serve_image(filename):
    """
    Serve an extracted image file from the backend image directory.
    """
    response = send_from_directory(IMAGES_DIR, filename)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

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


# Perform semantic search using CLIP embedding against precomputed image features
@app.route('/api/search', methods=['GET'])
def search():
    """
    Perform semantic search using CLIP embedding against precomputed image features.
    Logic:
      - Loads the selected CLIP model and tokenizer.
      - Computes query embedding for the input query text.
      - Loads the FAISS index and embedding paths for the selected model and Rosbag.
      - Searches the index for the top MAX_K results.
      - For each result:
          - Extracts the topic and timestamp from the embedding path.
          - Appends search results, including similarity score and model.
          - For each result timestamp, finds all reference timestamps in ALIGNED_DATA that map to it (used for UI marks).
    """
    query_text = request.args.get('query', default=None, type=str)
    if query_text is None:
        return jsonify({'error': 'No query text provided'}), 400

    rosbag_name = os.path.basename(SELECTED_ROSBAG).replace('.db3', '')
    results = []
    marks = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Load model, tokenizer, and compute query embedding
        name, pretrained = SELECTED_MODEL.split('__')
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained, device=device, cache_dir="/mnt/data/openclip_cache")
        tokenizer = open_clip.get_tokenizer(name)
        query_embedding = get_text_embedding(query_text, model, tokenizer, device)
        subdir = f"{name}__{pretrained}"
        index_path = os.path.join(INDICES_DIR, subdir, rosbag_name, "faiss_index.index")
        embedding_paths_file = os.path.join(INDICES_DIR, subdir, rosbag_name, "embedding_paths.npy")

        # Load the FAISS index and embedding paths
        if not Path(index_path).exists() or not Path(embedding_paths_file).exists():
            del model
            torch.cuda.empty_cache()
            return jsonify({'error': 'Missing index or embeddings for selected model'}), 500

        index_cpu = faiss.read_index(index_path)
        if device == "cuda" and faiss.get_num_gpus() > 0:
            # Move index to GPU(s) for faster search if available
            num_gpus = faiss.get_num_gpus()
            gpu_resources = [faiss.StandardGpuResources() for _ in range(num_gpus)]
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True  # Enable sharding across GPUs
            index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index_cpu, co)
            total_vectors = index.ntotal
        else:
            index = index_cpu

        embedding_paths = np.load(embedding_paths_file)

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
            result_timestamp = path_of_interest[-32:-13]
            result_topic = path_of_interest[:-33].replace("__", "/")
            results.append({
                'rank': i + 1,
                'embedding_path': embedding_path,
                'similarityScore': similarityScore,
                'topic': result_topic,
                'timestamp': result_timestamp,
                'model': SELECTED_MODEL
            })
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