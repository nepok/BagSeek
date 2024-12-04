import os
import base64
import csv
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import pandas as pd
from sensor_msgs.msg import PointCloud2
import struct
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the rosbag file path
rosbag_path = '/home/ubuntu/Documents/Bachelor/testcode/rosbag2_2024_08_01-16_00_23'

# Create a typestore and get the Image message class
typestore = get_typestore(Stores.LATEST)

# Store the available timestamps globally
timestamps = []

# Load the CSV into a pandas DataFrame (global so it can be accessed by the API)
aligned_data = pd.read_csv('/home/ubuntu/Documents/Bachelor/bagseek/aligned_data_with_max_distance.csv', dtype=str)

# FAISS and CLIP model loading
index_path = "/home/ubuntu/Documents/Bachelor/faiss_index/faiss_index.index"
embedding_paths_file = "/home/ubuntu/Documents/Bachelor/faiss_index/embedding_paths.npy"

# Load FAISS index
index = faiss.read_index(index_path)
embedding_paths = np.load(embedding_paths_file)

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

# Function to perform similarity search in FAISS
def search_faiss_index(query_embedding, k=5):
    # Perform the search (returning k nearest neighbors)
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    return distances, indices

# Endpoint to get available topics from the CSV file
@app.route('/api/topics', methods=['GET'])
def get_rosbag_topics():
    with Reader(rosbag_path) as reader:
        # Extract all topics using a set to avoid duplicates
        topics = sorted({conn.topic for conn in reader.connections})
    return jsonify({'topics': topics}), 200

@app.route('/api/timestamps', methods=['GET'])
def get_timestamps():
    # Extract the first column (Reference Timestamp) from the PD Frame
    timestamps = aligned_data['Reference Timestamp'].astype(str).tolist()

    return jsonify({'timestamps': timestamps})

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
    with Reader(rosbag_path) as reader:
        connections = [x for x in reader.connections if x.topic == topic]

        for connection, msg_timestamp, rawdata in reader.messages(connections=connections):
            if str(msg_timestamp) == realTimestamp:
                # Deserialize the message based on the connection's message type
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                if connection.msgtype == 'sensor_msgs/msg/Image':
                    if hasattr(msg, 'encoding'):
                        if msg.encoding == 'rgb8' or msg.encoding == 'bgr8':
                            image_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                            if msg.encoding == 'rgb8':
                                image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)

                            # Convert the image to a byte stream with WebP compression (set quality to 75)
                            _, img_bytes = cv2.imencode('.webp', image_data, [int(cv2.IMWRITE_WEBP_QUALITY), 75])

                            # Convert to base64
                            img_base64 = base64.b64encode(img_bytes.tobytes()).decode('utf-8')

                            return jsonify({'image': img_base64, 'realTimestamp': realTimestamp})

                elif connection.msgtype == 'sensor_msgs/msg/PointCloud2':
                    # Extract point cloud data
                    points = []
                    point_step = msg.point_step
                    for i in range(0, len(msg.data), point_step):
                        x, y, z = struct.unpack_from('fff', msg.data, i)
                        points.extend([x, y, z])  # Add the x, y, z coordinates as a flat list

                    return jsonify({'points': points, 'realTimestamp': realTimestamp})

                else:
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

    # Perform the FAISS search
    distances, indices = search_faiss_index(query_embedding)

    # Prepare the results
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'rank': i + 1,
            'path': embedding_paths[idx],
            'distance': float(distances[0][i]),
        })

    return jsonify({'query': query_text, 'results': results})

if __name__ == '__main__':
    app.run(debug=True)