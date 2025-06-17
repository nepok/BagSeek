#from __future__ import annotations
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
#from sensor_msgs.msg import PointCloud2
import struct
import torch
import faiss
import traceback
from typing import TYPE_CHECKING, cast
from rosbags.interfaces import ConnectionExtRosbag2  # type: ignore
import time
import threading
import open_clip
import yaml  # Add yaml import for metadata.yaml parsing
import math

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

EXPORT_PROGRESS = {"status": "idle", "progress": 0.0, "message": ""}

# Define the base path
BASE_DIR = '/mnt/data/bagseek/flask-backend/src'
IMAGES_DIR = os.path.join(BASE_DIR, 'extracted_images')
LOOKUP_TABLES_DIR = os.path.join(BASE_DIR, 'lookup_tables')
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings')
INDICES_DIR = os.path.join(BASE_DIR, 'faiss_indices')
ROSBAGS_DIR = "/mnt/data/rosbags"
EXPORT_DIR = "/mnt/data/rosbags/EXPORTED"
SELECTED_ROSBAG = os.path.join(ROSBAGS_DIR, 'output_bag')
CANVASES_FILE = os.path.join(BASE_DIR, 'canvases.json')
TOPICS_DIR = os.path.join(BASE_DIR, "topics")

# Create a typestore
typestore = get_typestore(Stores.ROS2_HUMBLE)

# import custom message types from the novatel_oem7_msgs package
msg_folder = Path('/opt/ros/humble/share/novatel_oem7_msgs/msg')
custom_types = {}

for msg_file in msg_folder.glob('*.msg'):
    try:
        text = msg_file.read_text()
        typename = f"novatel_oem7_msgs/msg/{msg_file.stem}"
        result = get_types_from_msg(text, typename)
        custom_types.update(result)
    except Exception as e:
        logging.warning(f"Failed to parse {msg_file.name}: {e}")

# Registriere alle geladenen Typen
typestore.register(custom_types)

# Store the available timestamps globally
timestamps = []

# Load the CSV into a pandas DataFrame (global so it can be accessed by the API)
aligned_data = pd.read_csv('../src/lookup_tables/rosbag2_2024_08_01-16_00_23.csv', dtype=str)

topic_cache = {}

selected_model = "ViT-B-16-quickgelu__openai"  # <-- oder was auch immer dein Default-Model sein soll


# Reference timestamp mapping globals
current_reference_timestamp = None
mapped_timestamps = {}

@app.route('/api/set-reference-timestamp', methods=['POST'])
def set_reference_timestamp():
    global current_reference_timestamp, mapped_timestamps
    try:
        data = request.get_json()
        referenceTimestamp = data.get('referenceTimestamp')
        
        if not referenceTimestamp:
            return jsonify({"error": "Missing referenceTimestamp"}), 400

        row = aligned_data[aligned_data["Reference Timestamp"] == str(referenceTimestamp)]
        if row.empty:
            return jsonify({"error": "Reference timestamp not found in CSV"}), 404

        current_reference_timestamp = referenceTimestamp
        mapped_timestamps = row.iloc[0].to_dict()
        # Clean/sanitize mapped_timestamps before returning
        cleaned = {
            k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in mapped_timestamps.items()
        }
        return jsonify({"mappedTimestamps": cleaned, "message": "Reference timestamp updated"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to compute text embedding using CLIP
def get_text_embedding(text, model, tokenizer, device):
    with torch.no_grad():
        tokens = tokenizer([text])
        tokens = tokens.to(device)
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

# Load canvases from file
def load_canvases():
    if os.path.exists(CANVASES_FILE):
        with open(CANVASES_FILE, "r") as f:
            return json.load(f)
    return {}

# Save canvases to file
def save_canvases(data):
    with open(CANVASES_FILE, "w") as f:
        json.dump(data, f, indent=4)

# logic for exporting rosbag
def export_rosbag_with_topics(src: Path, dst: Path, includedTopics, start_timestamp, end_timestamp) -> None:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(src) as reader, Writer(dst, version=9) as writer:
        conn_map = {}
        total_msgs = sum(1 for _ in reader.messages())  # Count total messages
        msg_counter = 0

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

        #from api import EXPORT_PROGRESS
        EXPORT_PROGRESS["status"] = "running"
        EXPORT_PROGRESS["progress"] = 0.0
        EXPORT_PROGRESS["message"] = "Export started"

        for conn, timestamp, data in reader.messages():
            if conn.id not in conn_map:
                continue
            msg_counter += 1
            EXPORT_PROGRESS["progress"] = round(msg_counter / total_msgs, 3)
            EXPORT_PROGRESS["message"] = f"Exporting message {msg_counter} of {total_msgs}"

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
        EXPORT_PROGRESS["message"] = "Export completed"
        EXPORT_PROGRESS["progress"] = 1.0

@app.route('/api/export-status', methods=['GET'])
def get_export_status():
    return jsonify(EXPORT_PROGRESS)

@app.route('/api/set-model', methods=['POST'])
def post_model():
    try:
        data = request.get_json()  # Get the JSON payload
        model_value = data.get('model')  # The path value from the JSON

        global selected_model
        selected_model = model_value

        # Reload the model and processor
        # reload_model_and_processor()

        return jsonify({"message": f"Model {model_value} successfully posted."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/get-models', methods=['GET'])
def get_models():
    try:
        # List all files in the directoryx
        models = []
        for model in os.listdir(EMBEDDINGS_DIR):
            if not model in ['.DS_Store', 'README.md']:
                models.append(model)

        return jsonify({"models": models}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500


@app.route('/api/set-file-paths', methods=['POST'])
def post_file_paths():
    try:
        data = request.get_json()  # Get the JSON payload
        path_value = data.get('path')  # The path value from the JSON

        global SELECTED_ROSBAG
        SELECTED_ROSBAG = path_value

        # Clear topic cache when selected rosbag changes
        topic_cache.clear()

        global aligned_data
        
        csv_path = LOOKUP_TABLES_DIR + "/" + os.path.basename(SELECTED_ROSBAG) + ".csv"
        aligned_data = pd.read_csv(csv_path, dtype=str)
        return jsonify({"message": f"File path updated successfully to {path_value}."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-file-paths', methods=['GET'])
def get_file_paths():
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
    
@app.route('/api/get-selected-rosbag', methods=['GET'])
def get_selected_rosbag():
    try:
        selectedRosbag = os.path.basename(SELECTED_ROSBAG)
        return jsonify({"selectedRosbag": selectedRosbag}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500
    
# Endpoint to get available topics from pre-generated JSON file
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


# Endpoint to get available topics from pre-generated JSON file
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
    
@app.route('/api/get-available-timestamps', methods=['GET'])
def get_available_timestamps():
    # Extract the first column (Reference Timestamp) from the PD Frame
    availableTimestamps = aligned_data['Reference Timestamp'].astype(str).tolist()
    return jsonify({'availableTimestamps': availableTimestamps})

@app.route('/images/<path:filename>')
def serve_image(filename):
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

@app.route('/api/content', methods=['GET'])
def get_ros():
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
                        # Extract point cloud data
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

                    case _:
                        # If the message type is something else, return msg.data as a list
                        return jsonify({'type': 'text', 'text': str(msg), 'timestamp': timestamp})

    return jsonify({'error': 'No message found for the provided timestamp and topic'})



@app.route('/api/search', methods=['GET'])
def search():

    query_text = request.args.get('query', default=None, type=str)
    if query_text is None:
        return jsonify({'error': 'No query text provided'}), 400

    rosbag_name = os.path.basename(SELECTED_ROSBAG).replace('.db3', '')
    results = []
    marks = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        name, pretrained = selected_model.split('__')
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained, device=device, cache_dir="/mnt/data/openclip_cache")
        tokenizer = open_clip.get_tokenizer(name)
        
        # Compute query embedding
        query_embedding = get_text_embedding(query_text, model, tokenizer, device)
        subdir = f"{name}__{pretrained}"
        index_path = os.path.join(INDICES_DIR, subdir, rosbag_name, "faiss_index.index")
        embedding_paths_file = os.path.join(INDICES_DIR, subdir, rosbag_name, "embedding_paths.npy")

        if not Path(index_path).exists() or not Path(embedding_paths_file).exists():
            del model
            torch.cuda.empty_cache()
            return jsonify({'error': 'Missing index or embeddings for selected model'}), 500

        index_cpu = faiss.read_index(index_path)
        if device == "cuda" and faiss.get_num_gpus() > 0:
            num_gpus = faiss.get_num_gpus()
            gpu_resources = [faiss.StandardGpuResources() for _ in range(num_gpus)]

            co = faiss.GpuMultipleClonerOptions()
            co.shard = True  # Enable sharding across GPUs

            index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index_cpu, co)
            total_vectors = index.ntotal
            
        else:
            index = index_cpu

        embedding_paths = np.load(embedding_paths_file)
        distances, indices = index.search(query_embedding.reshape(1, -1), 10)

        if len(indices) == 0 or len(distances) == 0:
            del model
            torch.cuda.empty_cache()
            return jsonify({'error': 'No results found for the query'}), 200

        for i, idx in enumerate(indices[0][:10]):
            embedding_path = str(embedding_paths[idx])
            path_of_interest = str(os.path.basename(embedding_path))
            result_timestamp = path_of_interest[-32:-13]
            result_topic = path_of_interest[:-33].replace("__", "/")
            results.append({
                'rank': i + 1,
                'embedding_path': embedding_path,
                'distance': float(distances[0][i]),
                'topic': result_topic,
                'timestamp': result_timestamp,
                'model': selected_model
            })
            # New logic: map from result_timestamp back to all reference timestamps using aligned_data
            matching_reference_timestamps = aligned_data.loc[
                aligned_data.isin([result_timestamp]).any(axis=1),
                'Reference Timestamp'
            ].tolist()

            logging.warning(matching_reference_timestamps)

            match_indices = []
            for ref_ts in matching_reference_timestamps:
                indices = aligned_data.index[aligned_data['Reference Timestamp'] == ref_ts].tolist()
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

    return jsonify({'query': query_text, 'results': results[:10], 'marks': marks})


@app.route('/api/export-rosbag', methods=['POST'])
def export_rosbag():
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

        export_rosbag_with_topics(SELECTED_ROSBAG, EXPORT_PATH, topics, start_timestamp, end_timestamp)
        return jsonify({"message": "Export successful", "exported_path": str(EXPORT_PATH)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/load-canvases", methods=["GET"])
def api_load_canvases():
    return jsonify(load_canvases())

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

if __name__ == '__main__':
    app.run(debug=True)