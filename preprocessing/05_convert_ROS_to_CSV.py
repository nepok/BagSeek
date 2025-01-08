import os
import csv
import json
import numpy as np
from rosbags.rosbag2 import Reader
from collections import defaultdict


# Define constants for paths
BASE_DIR = "/home/ubuntu/Documents/Bachelor/bagseek/flask-backend/src"
ROSBAGS_DIR = os.path.join(BASE_DIR, "rosbags")
LOOKUP_TABLES_DIR = os.path.join(BASE_DIR, "lookup_tables")

def process_rosbag(rosbag_path, csv_path):
    topic_data = defaultdict(list)

    # Read data from the ROS bag
    try:
        with Reader(rosbag_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                topic_data[connection.topic].append(timestamp)

        # Find the topic with the maximum number of timestamps
        max_topic = max(topic_data, key=lambda topic: len(topic_data[topic]))
        max_size = len(topic_data[max_topic])
        
        # Create a reference timeline of `max_size` evenly spaced timestamps
        reference_timestamps = np.linspace(
            topic_data[max_topic][0],
            topic_data[max_topic][-1],
            max_size
        )

        # Align each topic to the reference timeline
        aligned_data = {}
        for topic, timestamps in topic_data.items():
            aligned_list = []
            index = 0
            for ref_time in reference_timestamps:
                # Check if we have exhausted the timestamps for this topic
                if index >= len(timestamps):
                    aligned_list.append(None)
                    continue

                # If the current timestamp is closer than the previous one
                if index == 0 or abs(timestamps[index] - ref_time) < abs(timestamps[index - 1] - ref_time):
                    aligned_list.append(timestamps[index])
                    index += 1  # Use this timestamp and move to the next
                else:
                    aligned_list.append(None)  # No valid match for this reference time

            aligned_data[topic] = aligned_list

        # Prepare CSV data
        csv_data = []
        header = ['Reference Timestamp'] + list(aligned_data.keys()) + ['Max Distance']
        for i, ref_time in enumerate(reference_timestamps):
            row = [int(ref_time)]   # Format reference timestamp
            row_values = []
            
            for topic in aligned_data.keys():
                ts = aligned_data[topic][i]
                row.append(str(ts) if ts is not None else "None")
                if ts is not None:
                    row_values.append(ts)
            
            # Calculate max distance for this row
            if len(row_values) > 1:
                max_distance = max(row_values) - min(row_values)
            else:
                max_distance = None
            
            row.append(str(max_distance) if max_distance is not None else "None")
            csv_data.append(row)

        # Write to CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write the header
            writer.writerows(csv_data)  # Write all rows

        print(f"CSV file generated: {csv_path}")

    except Exception as e:
        print(f"Error: {e}")

# Run the processing
def main():
    """Main function to iterate over all rosbags and process them."""
    for rosbag_name in os.listdir(ROSBAGS_DIR):
        rosbag_path = os.path.join(ROSBAGS_DIR, rosbag_name)
        if os.path.isdir(rosbag_path):
            csv_path = os.path.join(LOOKUP_TABLES_DIR, f"{rosbag_name}.csv")
            if not os.path.exists(csv_path):
                print(f"Processing rosbag: {rosbag_name}")
                process_rosbag(rosbag_path, csv_path)
            else:
                print(f"Skipping already processed rosbag: {rosbag_name}")

if __name__ == "__main__":
    main()