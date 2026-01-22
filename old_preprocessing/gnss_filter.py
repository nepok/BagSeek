import os
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# Load environment variables
PARENT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Define paths
ROSBAGS_DIR_NAS = os.getenv("ROSBAGS_DIR_NAS")
IMAGES_PER_TOPIC_DIR = os.getenv("IMAGES_PER_TOPIC_DIR")
GNSS_FILTER = "/home/nepomuk/sflnas/DataReadOnly334/gps_filter/gps_polygon.json"
POLYGON_BUFFER_METERS = 30  # Shrink polygon by 30 meters inward for safety
OUTPUT_DIR = Path("/home/nepomuk/sflnas/DataReadWrite334/0_shared/Feldschwarm/bagseek/src/gnss_aligned_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create typestore for message deserialization
typestore = get_typestore(Stores.LATEST)

print(f"ROSBAGS_DIR_NAS: {ROSBAGS_DIR_NAS}")
print(f"IMAGES_PER_TOPIC_DIR: {IMAGES_PER_TOPIC_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")



# GPS Polygon Filter

def load_gps_polygon(polygon_file):
    """Load GPS polygon from JSON file."""
    try:
        with open(polygon_file, 'r') as f:
            data = json.load(f)
        
        # Extract coordinates - handle multiple formats
        if "coordinates" in data:
            # GeoJSON format: {"coordinates": [[[lon, lat], [lon, lat], ...]] }
            coords = data["coordinates"]
            # Handle nested lists (GeoJSON MultiPolygon or Polygon)
            if isinstance(coords[0][0], list):
                # Nested: take first polygon
                coords = coords[0]
            # Convert [lon, lat] to (lat, lon)
            polygon = [(coord[1], coord[0]) for coord in coords]
        elif "polygon" in data:
            # Simple format: [{"lat": ..., "lon": ...}, ...]
            polygon = [(p["lat"], p["lon"]) for p in data["polygon"]]
        else:
            # Try to parse as list directly
            polygon = data if isinstance(data, list) else []
        
        print(f"✓ Loaded GPS polygon with {len(polygon)} points")
        return polygon
    except Exception as e:
        print(f"⚠ Warning: Could not load GPS polygon: {e}")
        return None

def point_in_polygon(lat, lon, polygons):
    """
    Check if a point (lat, lon) is inside any of the polygons using ray-casting algorithm.
    
    Args:
        lat: Latitude of the point
        lon: Longitude of the point
        polygons: List of polygons, where each polygon is a list of (lat, lon) tuples
    
    Returns:
        bool: True if point is inside any polygon, False otherwise
    """
    # Handle both single polygon and list of polygons
    if not polygons:
        return False
    
    # If it's a single polygon (list of tuples), wrap it in a list
    if len(polygons) > 0 and isinstance(polygons[0], tuple) and len(polygons[0]) == 2:
        polygons = [polygons]
    
    # Check if point is inside ANY of the polygons
    for polygon in polygons:
        if not polygon or len(polygon) < 3:
            continue
        
        n = len(polygon)
        inside = False
        
        p1_lat, p1_lon = polygon[0]
        for i in range(1, n + 1):
            p2_lat, p2_lon = polygon[i % n]
            
            # Check if point is on an edge (ray crosses the edge)
            if lon > min(p1_lon, p2_lon):
                if lon <= max(p1_lon, p2_lon):
                    if lat <= max(p1_lat, p2_lat):
                        if p1_lon != p2_lon:
                            x_intersection = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                        if p1_lat == p2_lat or lat <= x_intersection:
                            inside = not inside
            
            p1_lat, p1_lon = p2_lat, p2_lon
        
        # If point is inside this polygon, return True immediately
        if inside:
            return True
    
    # Not inside any polygon
    return False

def buffer_polygon_inward(polygon, buffer_meters):
    """
    Shrink polygon inward by specified meters using shapely.
    Can return multiple polygons if buffer splits the original polygon.
    
    Args:
        polygon: List of (lat, lon) tuples
        buffer_meters: Distance in meters to shrink inward (positive = shrink, negative = expand)
    
    Returns:
        List of polygons, where each polygon is a list of (lat, lon) tuples
    """
    try:
        from shapely.geometry import Polygon
        
        # Convert to (lon, lat) for shapely
        coords_lonlat = [(lon, lat) for lat, lon in polygon]
        poly = Polygon(coords_lonlat)
        
        # Approximate conversion: meters to degrees
        # At latitude ~51°: 1 degree lat ≈ 111km, 1 degree lon ≈ 70km
        avg_lat = sum(lat for lat, lon in polygon) / len(polygon)
        meters_per_degree_lat = 111000
        meters_per_degree_lon = 111000 * np.cos(np.radians(avg_lat))
        
        # Convert meters to degrees (negative to shrink inward)
        buffer_degrees = -buffer_meters / ((meters_per_degree_lat + meters_per_degree_lon) / 2)
        
        # Apply buffer
        buffered_poly = poly.buffer(buffer_degrees)
        
        # Convert back to (lat, lon) list
        if buffered_poly.is_empty:
            print(f"⚠ Warning: Buffer of {buffer_meters}m too large, polygon disappeared. Using original.")
            return [polygon]
        
        # Handle MultiPolygon (keep ALL polygons, not just the largest)
        polygons = []
        if buffered_poly.geom_type == 'MultiPolygon':
            # Keep all polygon components
            for poly_component in buffered_poly.geoms:
                buffered_coords = list(poly_component.exterior.coords)
                polygons.append([(lat, lon) for lon, lat in buffered_coords])
            print(f"  Note: Buffer split polygon into {len(polygons)} separate areas")
        else:
            # Single Polygon
            buffered_coords = list(buffered_poly.exterior.coords)
            polygons.append([(lat, lon) for lon, lat in buffered_coords])
        
        return polygons
        
    except ImportError:
        print("⚠ shapely not installed. Install it to use polygon buffering")
        print("  Using original polygon without buffer")
        return [polygon]
    except Exception as e:
        print(f"⚠ Error applying buffer: {e}")
        return [polygon]

# Load polygon if file exists
GPS_POLYGON = None
if Path(GNSS_FILTER).exists():
    original_polygon = load_gps_polygon(Path(GNSS_FILTER))
    
    # Apply inward buffer if configured
    if original_polygon and POLYGON_BUFFER_METERS > 0:
        original_points = len(original_polygon)
        GPS_POLYGON = buffer_polygon_inward(original_polygon, POLYGON_BUFFER_METERS)
        total_points = sum(len(poly) for poly in GPS_POLYGON)
        print(f"✓ Applied {POLYGON_BUFFER_METERS}m inward buffer ({original_points} pts → {len(GPS_POLYGON)} polygon(s), {total_points} total pts)")
    else:
        # No buffer, wrap single polygon in list
        GPS_POLYGON = [original_polygon] if original_polygon else None
else:
    print(f"⚠ GPS polygon file not found: {GNSS_FILTER}")
    print("  Records will not be filtered by location.")

# Alignment functions (from 01_generate_alignment_and_metadata.py)

def determine_reference_topic(topic_timestamps):
    """Determine the topic with the most messages to use as the reference timeline."""
    return max(topic_timestamps.items(), key=lambda x: len(x[1]))[0]

def create_reference_timestamps(timestamps, factor=2):
    """Create a refined reference timeline with smaller intervals to improve alignment accuracy."""
    timestamps = sorted(set(timestamps))
    diffs = np.diff(timestamps)
    mean_interval = np.mean(diffs)
    refined_interval = mean_interval / factor
    ref_start = timestamps[0]
    ref_end = timestamps[-1]
    return np.arange(ref_start, ref_end, refined_interval).astype(np.int64)

def align_topic_to_reference(topic_ts, ref_ts, max_diff=int(1e8)):
    """Align timestamps of a topic to the closest reference timestamps within max_diff."""
    aligned = []
    topic_idx = 0
    for rt in ref_ts:
        while topic_idx + 1 < len(topic_ts) and abs(topic_ts[topic_idx + 1] - rt) < abs(topic_ts[topic_idx] - rt):
            topic_idx += 1
        closest = topic_ts[topic_idx]
        if abs(closest - rt) <= max_diff:
            aligned.append(closest)
        else:
            aligned.append(None)
    return aligned

def topic_to_directory_name(topic: str) -> str:
    """Convert topic name to directory-safe name."""
    return topic.replace("/", "__")

print("✓ Alignment functions loaded")

# Message data extraction functions

def extract_gnss_data(msg):
    """Extract GNSS/GPS data from NavSatFix message."""
    try:
        return {
            "latitude": float(msg.latitude),
            "longitude": float(msg.longitude),
            "altitude": float(msg.altitude),
            "status": int(msg.status.status) if hasattr(msg, 'status') else None,
            "service": int(msg.status.service) if hasattr(msg, 'status') else None,
        }
    except Exception as e:
        return {"error": str(e)}

def extract_tf_data(msg):
    """Extract TF (Transform) data from TFMessage."""
    try:
        transforms = []
        for transform in msg.transforms:
            tf_data = {
                "child_frame_id": transform.child_frame_id,
                "header": {
                    "frame_id": transform.header.frame_id,
                    "stamp": {
                        "sec": int(transform.header.stamp.sec),
                        "nanosec": int(transform.header.stamp.nanosec)
                    }
                },
                "transform": {
                    "translation": {
                        "x": float(transform.transform.translation.x),
                        "y": float(transform.transform.translation.y),
                        "z": float(transform.transform.translation.z)
                    },
                    "rotation": {
                        "x": float(transform.transform.rotation.x),
                        "y": float(transform.transform.rotation.y),
                        "z": float(transform.transform.rotation.z),
                        "w": float(transform.transform.rotation.w)
                    }
                }
            }
            transforms.append(tf_data)
        return {"transforms": transforms}
    except Exception as e:
        return {"error": str(e)}

def get_image_path(rosbag_name, topic, timestamp):
    """Get the relative path to the extracted image file."""
    topic_dir = topic_to_directory_name(topic)
    # Return relative path: extracted_images_per_topic/rosbag_name/topic_dir/timestamp.webp
    relative_path = f"extracted_images_per_topic/{rosbag_name}/{topic_dir}/{timestamp}.webp"
    
    # Check if file actually exists (using absolute path)
    absolute_path = Path(IMAGES_PER_TOPIC_DIR) / rosbag_name / topic_dir / f"{timestamp}.webp"
    if absolute_path.exists():
        return relative_path
    else:
        return None

print("✓ Data extraction functions loaded")

def classify_topic_type(topic_name, msg_type):
    """Classify topic as Image, GNSS, or TF based on message type."""
    # Only accept the front left camera
    if "Image" in msg_type:
        # Filter for only the front left camera
        if "zed_node" in topic_name and "left" in topic_name:
            return "image"
        return None  # Reject other image topics
    elif "NavSatFix" in msg_type or "GPS" in msg_type or "GNSS" in msg_type:
        return "gnss"
    elif topic_name == "/tf" or "TFMessage" in msg_type:
        return "tf"
    return None

def process_rosbag_with_data(rosbag_path, output_jsonl_path):
    """
    Process rosbag and extract aligned data for images, GNSS, and IMU.
    
    Args:
        rosbag_path: Path to the rosbag directory
        output_jsonl_path: Path to output JSONL file
    """
    rosbag_name = os.path.basename(rosbag_path)
    print(f"\n{'='*80}")
    print(f"Processing: {rosbag_name}")
    print(f"{'='*80}")
    
    # Store timestamps and messages by topic
    topic_data = defaultdict(list)
    topic_messages = defaultdict(dict)  # topic -> {timestamp: message_data}
    topic_types = {}
    topic_classifications = {}
    
    # Read all messages from rosbag
    print("Reading rosbag...")
    try:
        with Reader(rosbag_path) as reader:
            for connection, timestamp, rawdata in tqdm(reader.messages(), desc="Reading messages"):
                topic = connection.topic
                msg_type = connection.msgtype
                
                # Store message type
                if topic not in topic_types:
                    topic_types[topic] = msg_type
                    topic_classifications[topic] = classify_topic_type(topic, msg_type)
                
                # Only process image, GNSS, and IMU topics
                topic_class = topic_classifications[topic]
                if topic_class is None:
                    continue
                
                # Store timestamp
                topic_data[topic].append(timestamp)
                
                # Deserialize and extract data based on topic type
                try:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    
                    if topic_class == "gnss":
                        topic_messages[topic][timestamp] = extract_gnss_data(msg)
                    elif topic_class == "tf":
                        topic_messages[topic][timestamp] = extract_tf_data(msg)
                    # For images, we just store the timestamp (path is computed later)
                    
                except Exception as e:
                    # Skip messages that fail to deserialize
                    continue
    
    except Exception as e:
        print(f"Error reading rosbag: {e}")
        return
    
    if not topic_data:
        print("No relevant topics found in rosbag")
        return
    
    # Separate topics by classification
    image_topics = [t for t, c in topic_classifications.items() if c == "image"]
    gnss_topics = [t for t, c in topic_classifications.items() if c == "gnss"]
    tf_topics = [t for t, c in topic_classifications.items() if c == "tf"]
    
    print(f"\nFound:")
    print(f"  Image topics: {len(image_topics)}")
    print(f"  GNSS topics: {len(gnss_topics)}")
    print(f"  TF topics: {len(tf_topics)}")
    
    # Determine reference topic (use topic with most messages)
    ref_topic = determine_reference_topic(topic_data)
    print(f"\nReference topic: {ref_topic} ({len(topic_data[ref_topic])} messages)")
    
    # Create reference timeline
    print("Creating reference timeline...")
    ref_ts = create_reference_timestamps(topic_data[ref_topic])
    print(f"Reference timeline: {len(ref_ts)} timestamps")
    
    # Align all topics to reference
    print("\nAligning topics to reference timeline...")
    aligned_data = {}
    for topic, timestamps in tqdm(topic_data.items(), desc="Aligning topics"):
        aligned_data[topic] = align_topic_to_reference(sorted(timestamps), ref_ts)
    
    print(f"\n✓ Alignment complete")
    return ref_ts, aligned_data, topic_classifications, topic_messages, rosbag_name

print("✓ Main processing function loaded")

def write_aligned_jsonl(ref_ts, aligned_data, topic_classifications, topic_messages, rosbag_name, output_path):
    """
    Write aligned data to JSONL file.
    
    Each line in the JSONL file represents one reference timestamp with all aligned data.
    """
    print(f"\nWriting aligned data to {output_path}...")
    
    # Separate topics by type
    image_topics = [t for t, c in topic_classifications.items() if c == "image"]
    gnss_topics = [t for t, c in topic_classifications.items() if c == "gnss"]
    tf_topics = [t for t, c in topic_classifications.items() if c == "tf"]
    
    records_written = 0
    
    with open(output_path, 'w') as f:
        for i, ref_time in enumerate(tqdm(ref_ts, desc="Writing JSONL")):
            record = {
                "reference_timestamp": int(ref_time),
                "images": {},
                "gnss": {},
                "tf": {}
            }
            
            # Add aligned image data
            for topic in image_topics:
                aligned_ts = aligned_data[topic][i]
                if aligned_ts is not None:
                    image_path = get_image_path(rosbag_name, topic, aligned_ts)
                    record["images"][topic] = {
                        "timestamp": int(aligned_ts),
                        "path": image_path
                    }
            
            # Add aligned GNSS data
            for topic in gnss_topics:
                aligned_ts = aligned_data[topic][i]
                if aligned_ts is not None:
                    gnss_data = topic_messages[topic].get(aligned_ts)
                    if gnss_data:
                        record["gnss"][topic] = {
                            "timestamp": int(aligned_ts),
                            "data": gnss_data
                        }
            
            # Add aligned TF data
            for topic in tf_topics:
                aligned_ts = aligned_data[topic][i]
                if aligned_ts is not None:
                    tf_data = topic_messages[topic].get(aligned_ts)
                    if tf_data:
                        record["tf"][topic] = {
                            "timestamp": int(aligned_ts),
                            "data": tf_data
                        }
            
            # Check if location is within polygon (using first GNSS topic if available)
            in_area = False
            if GPS_POLYGON and record["gnss"]:
                # Get first GNSS topic data
                first_gnss = next(iter(record["gnss"].values()), None)
                if first_gnss and "data" in first_gnss:
                    gnss_data = first_gnss["data"]
                    if "latitude" in gnss_data and "longitude" in gnss_data:
                        lat = gnss_data["latitude"]
                        lon = gnss_data["longitude"]
                        in_area = point_in_polygon(lat, lon, GPS_POLYGON)
            
            record["in_area"] = in_area
            
            # Write record as JSON line
            f.write(json.dumps(record) + '\n')
            records_written += 1
    
    print(f"✓ Wrote {records_written} aligned records to {output_path}")
    return records_written

print("✓ JSONL writer function loaded")
