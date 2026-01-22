#!/usr/bin/env python3
"""
Rosbag GPS Lookup Table Creator

This script creates a lookup table mapping each rosbag to its GPS places
(rounded coordinates) with visit counts. The lookup is exported as JSON for
easy consumption by downstream tooling.

Usage:
1. Ensure all rosbags have been processed and CSV files exist
2. Run: python3 create_rosbag_gps_lookup.py
"""

import csv
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
PARENT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=PARENT_ENV)

# Configuration
ROSBAGS = Path(os.getenv("ROSBAGS"))

BASE_STR = os.getenv("BASE")

POSITIONS_STR = os.getenv("POSITIONS")
GPS_LOOKUP_TABLE_STR = os.getenv("GPS_LOOKUP_TABLE")

POSITIONS = Path(BASE_STR + POSITIONS_STR)
GPS_LOOKUP_TABLE = Path(BASE_STR + GPS_LOOKUP_TABLE_STR)

# Data source topic name
DATA_SOURCE = "novatel_oem7_bestpos"

# Sampling configuration
SAMPLE_EVERY_N = 1  # Sample every Nth entry for performance

# Visit frequency calculation
GRID_RESOLUTION = 0.0001  # Grid size in degrees (~11 meters)


def is_in_excluded_folder(path: Path) -> bool:
    """Check if a path is inside an EXCLUDED folder (case-insensitive)."""
    current = path.resolve().parent
    while current != current.parent:
        if current.name.upper() == "EXCLUDED":
            return True
        current = current.parent
    return False


def is_rosbag_excluded(rosbag_name: str) -> bool:
    """Check if a rosbag is in an EXCLUDED folder under ROSBAGS directory."""
    if ROSBAGS is None:
        return False
    
    # Check if rosbag path exists in ROSBAGS and is in EXCLUDED folder
    rosbag_path_in_rosbags = ROSBAGS / rosbag_name
    if rosbag_path_in_rosbags.exists():
        if is_in_excluded_folder(rosbag_path_in_rosbags):
            return True
    
    # Search for rosbag in ROSBAGS (might be in subdirectory)
    found_rosbag = None
    for rosbag_path in ROSBAGS.rglob(rosbag_name):
        if rosbag_path.is_dir():
            found_rosbag = rosbag_path
            break
    
    if found_rosbag:
        if is_in_excluded_folder(found_rosbag):
            return True
    else:
        # Also check if rosbag name appears in any EXCLUDED path (case-insensitive partial match)
        for excluded_path in ROSBAGS.rglob("*EXCLUDED*"):
            if excluded_path.is_dir():
                # Check if rosbag_name is in this EXCLUDED directory
                for item in excluded_path.iterdir():
                    if item.is_dir() and rosbag_name.lower() in item.name.lower():
                        return True
    
    return False


def extract_gps_data_from_csv(csv_path: Path, mcap_id: str) -> List[Tuple[float, float, str]]:
    """
    Extract GPS coordinates (lat, lon) from a CSV file.
    Returns list of (latitude, longitude, mcap_id) tuples.
    """
    coordinates = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            
            # Common delimiters to try
            delimiters = [',', ';', '\t', '|']
            detected_delimiter = ','
            
            for delimiter in delimiters:
                if delimiter in sample:
                    detected_delimiter = delimiter
                    break
            
            reader = csv.DictReader(f, delimiter=detected_delimiter)
            
            # Find latitude and longitude columns
            lat_col = None
            lon_col = None
            
            for col in reader.fieldnames:
                col_lower = col.lower().strip()
                if col_lower == 'lat' or col_lower == 'latitude':
                    lat_col = col
                elif col_lower == 'lon' or col_lower == 'longitude' or col_lower == 'lng':
                    lon_col = col
            
            if not lat_col or not lon_col:
                print(f"⚠️  Could not find lat/lon columns in {csv_path.name}")
                return coordinates
            
            # Read data with sampling
            row_count = 0
            for row in reader:
                if row_count % SAMPLE_EVERY_N == 0:
                    try:
                        lat = float(row[lat_col])
                        lon = float(row[lon_col])
                        
                        # Basic validation (rough bounds for Germany/Europe)
                        if 40.0 <= lat <= 60.0 and 0.0 <= lon <= 20.0:
                            coordinates.append((lat, lon, mcap_id))
                    except (ValueError, KeyError):
                        continue
                
                row_count += 1
                
    except Exception as e:
        print(f"❌ Error reading {csv_path.name}: {e}")
    
    return coordinates


def round_to_grid(lat: float, lon: float) -> Tuple[float, float]:
    """Round GPS coordinates to grid cells."""
    lat_grid = round(lat / GRID_RESOLUTION) * GRID_RESOLUTION
    lon_grid = round(lon / GRID_RESOLUTION) * GRID_RESOLUTION
    return lat_grid, lon_grid


def get_gps_data_for_rosbag(rosbag_name: str) -> List[Tuple[float, float, str]]:
    """Get GPS data for a specific rosbag from the positions directory.
    
    Structure: positions/rosbag_name/topic_name/mcapid.csv
    Returns list of (latitude, longitude, mcap_id) tuples.
    """
    all_coordinates = []
    rosbag_dir = POSITIONS / rosbag_name
    if not rosbag_dir.exists():
        return all_coordinates
    
    # Look for the data source topic directory
    topic_dir = rosbag_dir / DATA_SOURCE
    if not topic_dir.exists():
        return all_coordinates
    
    # Find all CSV files (mcapid.csv) in the topic directory
    csv_files = sorted(topic_dir.glob("*.csv"))
    if csv_files:
        print(f"📊 {rosbag_name}: Processing {len(csv_files)} {DATA_SOURCE} CSV files...")
        for csv_file in csv_files:
            # Extract MCAP ID from filename (e.g., "0.csv" -> "0")
            mcap_id = csv_file.stem
            coords = extract_gps_data_from_csv(csv_file, mcap_id)
            all_coordinates.extend(coords)
    
    return all_coordinates


def parse_rosbag_timestamp(rosbag_name: str) -> Optional[datetime]:
    """Parse timestamp from rosbag directory name: rosbag2_YYYY_MM_DD-HH_MM_SS"""
    try:
        # Remove 'rosbag2_' prefix if present
        if rosbag_name.startswith('rosbag2_'):
            timestamp_str = rosbag_name[8:]  # Remove 'rosbag2_' prefix
        else:
            timestamp_str = rosbag_name
        
        # Parse format: YYYY_MM_DD-HH_MM_SS
        return datetime.strptime(timestamp_str, "%Y_%m_%d-%H_%M_%S")
    except ValueError:
        return None


def create_lookup_table():
    """Create JSON lookup table mapping rosbags to GPS places with visit counts."""
    print("🗺️  Rosbag GPS Lookup Table Creator")
    print("=" * 70)
    
    # Find all processed rosbags
    rosbag_dirs = []
    if POSITIONS.exists():
        for dir_path in POSITIONS.iterdir():
            if dir_path.is_dir():
                # Check if this rosbag has the data source topic directory
                topic_dir = dir_path / DATA_SOURCE
                if topic_dir.exists() and topic_dir.is_dir():
                    rosbag_dirs.append(dir_path.name)
    
    # Sort rosbags by timestamp if parseable, otherwise by name
    def sort_key(name: str) -> Tuple[datetime, str]:
        ts = parse_rosbag_timestamp(name)
        if ts:
            return (ts, name)
        return (datetime.min, name)
    
    rosbag_dirs.sort(key=sort_key)
    
    print(f"📁 Found {len(rosbag_dirs)} processed rosbag directories with {DATA_SOURCE} data")
    
    if not rosbag_dirs:
        print("❌ No processed rosbags found!")
        return
    
    # Process each rosbag and create lookup table
    lookup_data = {}
    
    for rosbag_name in rosbag_dirs:
        # Check if rosbag is in EXCLUDED folder
        if is_rosbag_excluded(rosbag_name):
            print(f"⏭️  Skipping {rosbag_name}: found in EXCLUDED folder")
            continue
        
        print(f"\n🔍 Processing {rosbag_name}...")
        coords = get_gps_data_for_rosbag(rosbag_name)
        
        if not coords:
            print(f"⚠️  No GPS data found for {rosbag_name}")
            lookup_data[rosbag_name] = {}
            continue
        
        # Count visits per rounded GPS location per MCAP
        location_counts = defaultdict(lambda: defaultdict(int))
        for lat, lon, mcap_id in coords:
            lat_grid, lon_grid = round_to_grid(lat, lon)
            lat_lon_key = f"{lat_grid:.6f},{lon_grid:.6f}"
            location_counts[lat_lon_key][mcap_id] += 1
        
        # Convert to final structure with total and mcaps dict
        final_location_data = {}
        for lat_lon_key, mcap_counts in location_counts.items():
            total_count = sum(mcap_counts.values())
            final_location_data[lat_lon_key] = {
                "total": total_count,
                "mcaps": dict(mcap_counts)
            }
        
        # Add to lookup data
        lookup_data[rosbag_name] = final_location_data
        
        print(f"✅ Processed {len(coords)} GPS points, {len(location_counts)} unique locations")
    
    # Write JSON file
    print(f"\n💾 Writing lookup table to {GPS_LOOKUP_TABLE}...")
    GPS_LOOKUP_TABLE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(GPS_LOOKUP_TABLE, 'w', encoding='utf-8') as f:
        json.dump(lookup_data, f, indent=2, ensure_ascii=False)
    
    total_locations = sum(len(locations) for locations in lookup_data.values())
    print(f"✅ Lookup table created for {len(lookup_data)} rosbags with {total_locations} total locations")
    print(f"📂 File saved to: {GPS_LOOKUP_TABLE}")


if __name__ == "__main__":
    create_lookup_table()

