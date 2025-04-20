import os
import csv
import numpy as np
from rosbags.rosbag2 import Reader  # type: ignore
from collections import defaultdict

# Define constants for paths
BASE_DIR = "/mnt/data/bagseek/flask-backend/src"
ROSBAGS_DIR = "/mnt/data/rosbags"
LOOKUP_TABLES_DIR = os.path.join(BASE_DIR, "lookup_tables/lookup_tables_new_test")

def process_rosbag(rosbag_path, csv_path):
    topic_data = defaultdict(list)
    
    try:
        with Reader(rosbag_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic in ["/tf", "/tf_static"]:
                    continue
                topic_data[connection.topic].append(timestamp)
    except Exception as e:
        print(f"Error reading .db3 bag: {e}")
        return
    
    if not topic_data:
        print(f"No topics found in {rosbag_path}")
        return
    
    # Find the topic with the maximum number of timestamps
    max_topic = max(topic_data, key=lambda topic: len(topic_data[topic]))
    max_size = len(topic_data[max_topic])
    
    # Create a reference timeline of `max_size` evenly spaced timestamps
    reference_timestamps = np.array(sorted(set(topic_data[max_topic])))

    # Align each topic to the reference timeline
    aligned_data = {}
    for topic, timestamps in topic_data.items():
        aligned_list = []
        index = 0
        for ref_time in reference_timestamps:
            if index >= len(timestamps):
                aligned_list.append(None)
                continue
            if index == 0 or abs(timestamps[index] - ref_time) < abs(timestamps[index - 1] - ref_time):
                aligned_list.append(timestamps[index])
                index += 1
            else:
                aligned_list.append(None)
        aligned_data[topic] = aligned_list
    
    # Prepare CSV data
    csv_data = []
    header = ['Reference Timestamp'] + list(aligned_data.keys()) + ['Max Distance']
    for i, ref_time in enumerate(reference_timestamps):
        row = [int(ref_time)]
        row_values = []
        
        for topic in aligned_data.keys():
            ts = aligned_data[topic][i]
            row.append(str(ts) if ts is not None else "None")
            if ts is not None:
                row_values.append(ts)
        
        max_distance = max(row_values) - min(row_values) if len(row_values) > 1 else None
        row.append(str(max_distance) if max_distance is not None else "None")
        csv_data.append(row)
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(csv_data)
    
    print(f"CSV file generated: {csv_path}")

def main():
    for rosbag_name in os.listdir(ROSBAGS_DIR):
        rosbag_path = os.path.join(ROSBAGS_DIR, rosbag_name)
        csv_path = os.path.join(LOOKUP_TABLES_DIR, f"{rosbag_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"Processing rosbag: {rosbag_name}")
            process_rosbag(rosbag_path, csv_path)
        else:
            print(f"Skipping already processed rosbag: {rosbag_name}")

if __name__ == "__main__":
    main()