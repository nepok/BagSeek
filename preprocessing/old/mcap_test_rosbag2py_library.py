import argparse
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String
import rosbag2_py
from dotenv import load_dotenv
from pathlib import Path
import os

PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Define constants for paths
ROSBAGS_DIR_NAS = os.getenv("ROSBAGS_DIR_NAS")

def read_messages(input_bag: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(typename(topic))
        msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp
    del reader

def main():

    if not ROSBAGS_DIR_NAS:
        raise RuntimeError("ROSBAGS_DIR_NAS is not set in the environment")

    def iter_bag_dirs(base_dir: Path):
        for dirpath, _dirnames, filenames in os.walk(base_dir):
            if "metadata.yaml" in filenames:
                yield Path(dirpath)

    rosbag_root = Path(ROSBAGS_DIR_NAS)
    if not rosbag_root.exists():
        raise FileNotFoundError(f"ROSBAGS_DIR_NAS path does not exist: {rosbag_root}")

    for bag_dir in iter_bag_dirs(rosbag_root):
        print(f"-- Reading bag: {bag_dir}")
        for topic, msg, timestamp in read_messages(str(bag_dir)):
            if isinstance(msg, String):
                print(f"{topic} [{timestamp}]: '{msg.data}'")
            else:
                print(f"{topic} [{timestamp}]: ({type(msg).__name__})")


if __name__ == "__main__":
    main()
