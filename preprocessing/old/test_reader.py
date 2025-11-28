from rosbags.rosbag2 import Reader
from tqdm import tqdm
import os

rosbag_path = "/home/nepomuk/sflnas/DataReadOnly334/tractor_data/autorecord/rosbag2_2025_07_25-10_14_58"
#rosbag_path = "/mnt/data/rosbags/output_bag"

try:
    with Reader(rosbag_path) as reader:
        print(f"✅ Successfully opened rosbag: {rosbag_path}")
        print(f"Topics: {reader.topics}")
        
        message_count = 0
        for connection, timestamp, rawdata in tqdm(reader.messages(), desc="Reading rosbag"):
            message_count += 1
            if message_count <= 5:  # Print first 5 messages
                print(f"  Topic: {connection.topic}, Timestamp: {timestamp}")
        
        print(f"\n✅ Successfully read {message_count} messages from rosbag")
        
except Exception as e:
    # Print detailed error info including rosbag path and MCAP/DB3 files
    print(f"\n{'='*80}")
    print(f"❌ ERROR reading rosbag: {rosbag_path}")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    
    # List all MCAP and DB3 files in this rosbag to identify the problematic one
    try:
        files = os.listdir(rosbag_path)
        mcap_files = [f for f in files if f.endswith('.mcap')]
        
        print(f"\nFiles in rosbag directory:")
        if mcap_files:
            print(f"  MCAP files ({len(mcap_files)}): {mcap_files}")
        
        # Show file sizes to identify potentially incomplete files
        print(f"\nFile sizes:")
        for f in mcap_files:
            fpath = os.path.join(rosbag_path, f)
            size = os.path.getsize(fpath)
            size_mb = size / (1024 * 1024)
            print(f"  {f}: {size_mb:.2f} MB ({size:,} bytes)")
            
    except Exception as list_err:
        print(f"Could not list files: {list_err}")
    
    print(f"{'='*80}\n")
    raise