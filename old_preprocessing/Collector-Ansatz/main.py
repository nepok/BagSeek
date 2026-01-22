# main.py
import os
import sys
import subprocess
import re
from pathlib import Path
from typing import Optional, List

from mcap.exceptions import McapError
from completion import completion_done, update_completion
from classes.timestamp_alignment import TimestampAlignment
from classes.topics import TopicsCollector

from dotenv import load_dotenv
from mcap.reader import SeekingReader
from mcap_ros2.decoder import DecoderFactory

PARENT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)


def get_rosbags(rosbags_dir: Path):
    """
    Get all rosbags from the rosbags directory.
    
    Handles multi-part rosbags: if a folder ends with '_multi_parts',
    it goes one level deeper and extracts all Part_N folders.
    
    Args:
        rosbags_dir: Path to the rosbags directory
        
    Returns:
        List of Path objects, one for each rosbag, sorted by base name then Part number
    """
    rosbags = []
    
    # Iterate over all items in rosbags_dir
    for item in rosbags_dir.iterdir():
        # Skip non-directories and EXCLUDED folders
        if not item.is_dir() or "EXCLUDED" in item.name:
            continue
        
        # Check if this is a multi-part rosbag container
        if item.name.endswith("_multi_parts"):
            # Go one level deeper and collect all Part_N folders
            for part_dir in item.iterdir():
                if not part_dir.is_dir():
                    continue
                
                # Check if it's a Part_N folder (Part_1, Part_2, etc.)
                if part_dir.name.startswith("Part_"):
                    part_suffix = part_dir.name[5:]  # Remove "Part_" prefix
                    # Verify it's a valid Part_N format (N is a number)
                    if part_suffix.isdigit():
                        rosbags.append(part_dir)
        else:
            # Regular rosbag - include the folder itself
            rosbags.append(item)
    
    # Sort rosbags: primary by base name (without _multi_parts), secondary by Part number
    def sort_key(path: Path):
        """Extract base name and part number for sorting."""
        if path.name.startswith("Part_"):
            # Multi-part rosbag: extract base name from parent and Part number
            parent_name = path.parent.name
            base_name = parent_name.replace("_multi_parts", "")
            part_num = int(path.name[5:])  # Extract Part number
        else:
            # Regular rosbag: use folder name as base, Part number is 0
            base_name = path.name
            part_num = 0
        
        return (base_name, part_num)
    
    rosbags.sort(key=sort_key)
    return rosbags


def get_mcaps(rosbag: Path):
    """
    Get all mcaps from the rosbag, sorted numerically by the number in the filename.
    
    Args:
        rosbag: Path to the rosbag directory
        
    Returns:
        List of Path objects, one for each .mcap file, sorted by numeric order
    """
    # Use glob to find all .mcap files
    mcaps = list(rosbag.glob("*.mcap"))
    
    # Sort by extracting the numeric part from the filename
    # Filename format: rosbag2_YYYY_MM_DD-HH_MM_SS_N.mcap
    def extract_number(path: Path):
        """Extract the number from the mcap filename for sorting."""
        stem = path.stem  # filename without extension
        # Extract number after the last underscore
        parts = stem.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        # If no number found, return a large number to put it at the end
        return float('inf')
    
    mcaps.sort(key=extract_number)
    return mcaps

def main():

    rosbags_dir = Path(os.getenv("ROSBAGS"))
    lookup_tables_dir = Path(os.getenv("BASE") + os.getenv("LOOKUP_TABLES"))
    topics_dir = Path(os.getenv("BASE") + os.getenv("TOPICS"))


    # completion check
    # fallback: actual checking of the files
    # update completion if necessary

    # build the pipeline: (so it runs the pipeline but checks completion of each step before)

    # run the pipeline:
    # iterate over all rosbags that are not already completed completely

    for rosbag in get_rosbags(rosbags_dir):
        # Create TopicsCollector - it will determine the JSON path internally
        # The collector will check completion internally and skip if already done
        topics_collector = TopicsCollector(input_path=rosbag, output_dir=topics_dir)
        topics_collector.finalize()
        
        
        """
        for mcap in get_mcaps(rosbag):
            rosbag_name = mcap.parent.relative_to(rosbags_dir)
            mcap_number = mcap.stem.rsplit('_', 1)[-1] if '_' in mcap.stem else mcap.stem

            csv_path = lookup_tables_dir / rosbag_name / f"{mcap_number}.csv"
            
            # Pass all_topics to ensure consistent CSV columns across all MCAPs
            timestamp_alignment = TimestampAlignment(csv_path=csv_path, all_topics=all_topics)
            
            with open(mcap, "rb") as f:
                reader = SeekingReader(f, decoder_factories=[DecoderFactory()], record_size_limit=8 * 1024 * 1024 * 1024)
                
                for schema, channel, message, ros2_msg in reader.iter_decoded_messages(log_time_order=True, reverse=False):
                    msg_type = schema.name if schema else "unknown"

                    for p in preprocessors:
                        if p.enabled and p.wants(channel.topic, msg_type):
                            p.on_message(
                                topic=channel.topic,
                                msg=ros2_msg,
                                timestamp_ns=message.log_time
                            )
                    
                for p in preprocessors:
                    if p.enabled:
                        p.finalize()

                
                break
        """
        #break

    # group all the models into preprocessing groups

    # per rosbag: 

    # if completion of topics for this rosbag is not done: create topics.json using metadata.yaml data or from mcap info (fallback)
    # if representative_previews for all image topics of a rosbag isnt done: collect count of mcaps to know the fenceposts (split through 8 and take the first image from each image topic from the 1st to 7th part (fenceposts)). if len < 8 then use the message timestamps from start to end of the whole rosbag and create fenceposts from that. then search for the nearest message of a topic to create the fenceposts.

        # then: iterate over all mcaps that are not already completed completely

        # per mcap: 
        # use collectors to collect data from the mcap while iterating over the mcap
        # Timestamp Alignment Collector: collect all timestamps of all topics
        # Image Collector: collect all image data of all image topics
        # Bestpos Collector: collect all bestpos data (only lat, lon) of all bestpos topics
        
        # finalize
        # out of images: convert to PIL, preprocess using the preprocessing groups, create embeddings using the right preprocessed image, extend shards.
        # create adjacent similarities out of embeddings



if __name__ == "__main__":
    main()