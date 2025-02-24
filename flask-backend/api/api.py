import json
import os
from pathlib import Path
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_typestore
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import logging
import pandas as pd
#from sensor_msgs.msg import PointCloud2
import struct
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the base path
BASE_DIR = '../src'
IMAGES_DIR = os.path.join(BASE_DIR, 'extracted_images')
INDICES_DIR = os.path.join(BASE_DIR, 'faiss_indices')
ROSBAGS_DIR = os.path.join(BASE_DIR, 'rosbags')
EXPORT_DIR = os.path.join(BASE_DIR, 'exported_rosbags')
SELECTED_ROSBAG = os.path.join(ROSBAGS_DIR, 'rosbag2_2024_08_01-16_00_23')
CANVASES_FILE = os.path.join(BASE_DIR, 'canvases.json')

# Create a typestore
typestore = get_typestore(Stores.LATEST)

# Store the available timestamps globally
timestamps = []

# Load the CSV into a pandas DataFrame (global so it can be accessed by the API)
aligned_data = pd.read_csv('../src/lookup_tables/rosbag2_2024_08_01-16_00_23.csv', dtype=str)

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to compute text embedding using CLIP
def get_text_embedding(text):
    # Preprocess the text using CLIPProcessor (tokenizes and converts to tensors)
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Generate embedding
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize
    return text_features.cpu().numpy().flatten()

# Function to perform similarity search in FAISS TODO: warum error, wenn nicht genutzt?
# TODO: eher binary search implementieren? oder mehr 
def search_faiss_index(query_embedding, index, k=5):
    # Perform the search
    _, distances, indices = index.range_search(query_embedding.reshape(1, -1), 1.5)
    return distances, indices

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

@app.route('/api/set-file-paths', methods=['POST'])
def post_file_paths():
    try:
        data = request.get_json()  # Get the JSON payload
        path_value = data.get('path')  # The path value from the JSON

        global SELECTED_ROSBAG
        SELECTED_ROSBAG = path_value

        global aligned_data
        csv_path = str(path_value).replace("rosbags", "lookup_tables") + ".csv"
        aligned_data = pd.read_csv(csv_path, dtype=str)
        return jsonify({"message": "File path updated successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-file-paths', methods=['GET'])
def get_file_paths():
    try:
        # List all files in the directory
        files = os.listdir(ROSBAGS_DIR)
        ros_files = [os.path.join(ROSBAGS_DIR, file) for file in files]
        return jsonify({"paths": ros_files}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/get-selected-rosbag', methods=['GET'])
def get_selected_rosbag():
    try:
        selectedRosbag = os.path.basename(SELECTED_ROSBAG)
        return jsonify({"selectedRosbag": selectedRosbag}), 200
    except Exception as e:
        # Handle any errors that occur (e.g., directory not found, permission issues)
        return jsonify({"error": str(e)}), 500
    
# Endpoint to get available topics from the CSV file
@app.route('/api/topics', methods=['GET'])
def get_rosbag_topics():
    try:
        with Reader(SELECTED_ROSBAG) as reader:
            # Extract all topics using a set to avoid duplicates
            topics = sorted({conn.topic for conn in reader.connections})
    except Exception as error:
        # Handle the error gracefully by returning an empty list
        print(f"Error reading rosbag: {error}")
        topics = []
    
    return jsonify({'topics': topics}), 200

@app.route('/api/timestamps', methods=['GET'])
def get_timestamps():
    # Extract the first column (Reference Timestamp) from the PD Frame
    timestamps = aligned_data['Reference Timestamp'].astype(str).tolist()
    return jsonify({'timestamps': timestamps})

@app.route('/images/<path:filename>')
def serve_image(filename):
    response = send_from_directory(IMAGES_DIR, filename)
    response.headers["Cache-Control"] = "public, max-age=86400"  # Cache for 1 day
    return response

@app.route('/api/ros', methods=['GET'])
def get_ros():
    timestamp = request.args.get('timestamp', default=None, type=str)  # Get timestamp from query params
    topic = request.args.get('topic', default=None, type=str)  # Get topic from query params
    mode = request.args.get('mode', default='show', type=str)  # Default mode is "show"

    if not timestamp:
        return jsonify({'error': 'No timestamp provided'})
    if not topic:
        return jsonify({'error': 'No topic provided'})
    if mode not in ['show', 'search']:
        return jsonify({'error': f'Invalid mode: {mode}. Supported modes are "show" and "search".'})
        
    if mode == 'show':
        # Validate timestamp and topic in the DataFrame
        if timestamp not in aligned_data['Reference Timestamp'].values:
            return jsonify({'error': 'Timestamp not found in aligned_data'})
        if topic not in aligned_data.columns:
            return jsonify({'error': f'Topic {topic} not found in aligned_data'})

        # Lookup the real timestamp in the DataFrame
        row = aligned_data[aligned_data['Reference Timestamp'] == timestamp]
        realTimestamp = row[topic].iloc[0]
    else:
        realTimestamp = timestamp


    # Open the rosbag to find the message at the requested timestamp and topic
    with Reader(SELECTED_ROSBAG) as reader:
        connections = [x for x in reader.connections if x.topic == topic]

        for connection, msg_timestamp, rawdata in reader.messages(connections=connections):
            if str(msg_timestamp) == realTimestamp:
                # Deserialize the message based on the connection's message type
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                match connection.msgtype:

                    case 'sensor_msgs/msg/Image':
                        print("Image")
                        
                        """if hasattr(msg, 'encoding'):
                            if msg.encoding == 'rgb8' or msg.encoding == 'bgr8':
                                image_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                                if msg.encoding == 'rgb8':
                                    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

                                # Convert the image to a byte stream with WebP compression (set quality to 75)
                                _, img_bytes = cv2.imencode('.webp', image_data, [int(cv2.IMWRITE_WEBP_QUALITY), 75])

                                # Convert to base64
                                img_base64 = base64.b64encode(img_bytes.tobytes()).decode('utf-8')

                                return jsonify({'image': img_base64, 'realTimestamp': realTimestamp}) """
                    

                    case 'sensor_msgs/msg/PointCloud2':
                        # Extract point cloud data
                        points = []
                        point_step = msg.point_step
                        for i in range(0, len(msg.data), point_step):
                            x, y, z = struct.unpack_from('fff', msg.data, i)
                            points.extend([x, y, z])  # Add the x, y, z coordinates as a flat list

                        return jsonify({'points': points, 'realTimestamp': realTimestamp})
                    
                    case 'sensor_msgs/msg/NavSatFix':
                        return jsonify({'gpsData': {'latitude': msg.latitude, 'longitude': msg.longitude, 'altitude': msg.altitude},'realTimestamp': realTimestamp})

                    case _:
                        # If the message type is something else, return msg.data as a list
                        return jsonify({'text': str(msg), 'realTimestamp': realTimestamp})

    return jsonify({'error': 'No message found for the provided timestamp and topic'})


# New API endpoint to perform text-based search using FAISS and CLIP
@app.route('/api/search', methods=['GET'])
def search():
    query_text = request.args.get('query', default=None, type=str)
    if query_text is None:
        return jsonify({'error': 'No query text provided'}), 400
    
    # Compute the query embedding
    query_embedding = get_text_embedding(query_text)

    # Define paths for the specified rosbag
    rosbag_name = os.path.basename(SELECTED_ROSBAG).replace('.db3', '')#.replace('.bag', '')
    index_path = os.path.join(INDICES_DIR, rosbag_name, "faiss_index.index")
    embedding_paths_file = os.path.join(INDICES_DIR, rosbag_name, "embedding_paths.npy")

    # Check if the index and embedding paths exist
    if not Path(index_path).exists() or not Path(embedding_paths_file).exists():
        print(f"FAISS index or embedding paths not found for rosbag: {rosbag_name}")
        return jsonify({'error': f'FAISS index or embedding paths not found for rosbag: {rosbag_name}'}), 404
    
    # Load FAISS index and embedding paths
    index = faiss.read_index(index_path)
    embedding_paths = np.load(embedding_paths_file)

    # Perform the FAISS search
    distances, indices = search_faiss_index(query_embedding, index)
    # Prepare the results
    results = []
    marks = []

    if len(indices) == 0 or len(distances) == 0:
        return jsonify({'query': query_text, 'results': [], 'marks': []})  # Return empty lists if no results are found

    for i, idx in enumerate(indices):

        if i >= 20:
            break

        # Prepare the result object
        embedding_path = str(embedding_paths[idx])
        path_of_interest = str(os.path.basename(embedding_path))
        result_timestamp = path_of_interest[-32:-13]
        result_topic = path_of_interest[:-33].replace("__", "/")
                
        results.append({
            'rank': i + 1,
            'embedding_path': embedding_path,
            'distance': float(distances[i]),
            'topic': result_topic,
            'timestamp': result_timestamp
        })

        # Find matching reference timestamp and store in the dictionary
        #match = aligned_data.loc[aligned_data[result_topic] == result_timestamp, "Reference Timestamp"]
        match = aligned_data[aligned_data['Reference Timestamp'] == result_timestamp].index
        for index in match:
            marks.append({
                'value': index,    # Use the index as the "value"
                #'label': str(timestamp)  # Use the timestamp as the "label"
            })

        if len(marks) >= 20:
            break

    return jsonify({'query': query_text, 'results': results, 'marks': marks})

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

        export_path = os.path.join(EXPORT_DIR, f"{new_rosbag_name}_exported")
        typestore = get_typestore(Stores.LATEST)

        with Reader(SELECTED_ROSBAG) as reader, Writer(export_path) as writer:
            for connection, timestamp, rawdata in reader.messages():
                logging.warning(type(start_timestamp)), type(end_timestamp), type(timestamp)
                #writer.add_connection(connection)
                if connection.topic in topics and start_timestamp <= timestamp <= end_timestamp:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    writer.write(connection, timestamp, msg)

        return jsonify({"message": "Rosbag exported successfully", "export_path": export_path}), 200

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/load-canvases", methods=["GET"])
def api_load_canvases():
    return jsonify(load_canvases())

@app.route("/api/save-canvases", methods=["POST"])
def api_save_canvases():
    data = request.json
    save_canvases(data)
    return jsonify({"message": "Canvas saved successfully"})

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