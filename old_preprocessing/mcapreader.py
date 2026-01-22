import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# Hardcoded path to the ROS 2 bag folder
MCAP_FOLDER = "/mnt/data/bagseek/flask-backend/src/rosbags/rosbag2_2025_02_21-09_17_40"

def read_messages(input_bag: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),  # Open the entire bag folder
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"Topic {topic_name} not in bag")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(typename(topic))
        msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp
    del reader

def main():
    print(f"Reading from bag: {MCAP_FOLDER}")
    i = 0
    for topic, msg, timestamp in read_messages(MCAP_FOLDER):
        #print(f"{topic} ({type(msg).__name__}) [{timestamp}]: '{msg}'")
        print(f"{timestamp}, i: {i}")
        i += 1

if __name__ == "__main__":
    main()
