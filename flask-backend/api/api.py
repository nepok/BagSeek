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
from sensor_msgs.msg import PointCloud2  # Import PointCloud2 message type
import struct

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

    if timestamp is None:
        return jsonify({'error': 'No timestamp provided'})
    if topic is None:
        return jsonify({'error': 'No topic provided'})

    # Check if timestamp exists in aligned_data
    if timestamp not in aligned_data['Reference Timestamp'].values:
        return jsonify({'error': 'Timestamp not found in aligned_data'})

    # Check if the topic exists as a column in aligned_data
    if topic not in aligned_data.columns:
        return jsonify({'error': f'Topic {topic} not found in aligned_data'})

    row = aligned_data[aligned_data['Reference Timestamp'] == timestamp]
    realTimestamp = row[topic].iloc[0]

    # Open the rosbag to find the image at the requested timestamp and topic
    with Reader(rosbag_path) as reader:
        connections = [x for x in reader.connections if x.topic == topic]

        for connection, msg_timestamp, rawdata in reader.messages(connections=connections):
            if str(msg_timestamp) == realTimestamp:
                # Deserialize the message based on the connection's message type
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                # Check if the message is an image (sensor_msgs/msg/Image)
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
                    else:
                        print(f"Unsupported encoding {msg.encoding}")
                else:
                    #TODO: überbrückung, sollte dann 3d daten übergeben , damit es angezeigt werden kann
                    float_values = [struct.unpack('f', bytes(msg.data[i:i+4]))[0] for i in range(0, len(msg.data) - 3, 4)]
                    grouped_values = [float_values[i:i + 3] for i in range(0, len(float_values), 3)]
                    formatted_values = "\n".join([", ".join(map(str, group)) for group in grouped_values])

                    return jsonify({'text': formatted_values, 'realTimestamp': realTimestamp})

    return jsonify({'error': 'No message found for the provided timestamp and topic'})

if __name__ == '__main__':
    app.run(debug=True)