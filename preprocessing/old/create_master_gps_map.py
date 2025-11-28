#!/usr/bin/env python3
"""
Master GPS Map Creator

This script creates a comprehensive map showing all GPS positions from processed rosbags.
It reads CSV files from bestpos and fix directories, extracts GPS coordinates,
and creates a satellite map visualization with real satellite imagery background.

Features:
- High-quality satellite imagery background (Esri WorldImagery)
- GPS points color-coded by altitude
- High-resolution output (300 DPI)
- Configurable sampling and visualization options

Usage:
1. Ensure all rosbags have been processed and CSV files exist
2. Install required packages: pip install contextily
3. Run: python3 create_master_gps_map.py
"""

import csv
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
try:
    import contextily as ctx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False
    print("⚠️  contextily not available. Install with: pip install contextily")
    print("   Satellite background will be disabled.")

from matplotlib.patches import Rectangle

# after importing contextily (if available)
try:
    ctx.set_cache_dir(str(Path.home() / ".cache" / "contextily"))
except Exception:
    pass

# Configuration
POSITIONAL_DATA_DIR = Path("/home/nepomuk/sflnas/DataReadWrite334/0_shared/Feldschwarm/bagseek/src/positional_data")
OUTPUT_MAP = Path("/mnt/data/master_gps_map")

# Select which data source to use: "fix" or "bestpos"
DATA_SOURCE = "bestpos"  # "fix" or "bestpos"

if DATA_SOURCE not in ("fix", "bestpos"):
    raise SystemExit("DATA_SOURCE must be 'fix' or 'bestpos'")

SELECTED_DIR = POSITIONAL_DATA_DIR / DATA_SOURCE

print(SELECTED_DIR)

# Sampling configuration
SAMPLE_EVERY_N = 1  # Sample every 100th entry for performance
MIN_POINTS_PER_ROSBAG = 1  # Minimum points required to include a rosbag

# Map configuration
MAP_SIZE_INCHES = (20, 15)  # Width, Height in inches
DPI = 300  # High resolution for detailed map
POINT_SIZE = 5.0  # Size of GPS points (increased for better visibility)
ALPHA = 0.1  # Transparency of points (higher for more vibrant colors)

# Satellite imagery configuration
USE_SATELLITE_BACKGROUND = True  # Set to False for plain background

if CONTEXTILY_AVAILABLE:
    SATELLITE_PROVIDER = ctx.providers.Esri.WorldImagery  # High-quality satellite imagery
    # Alternative providers:
    # SATELLITE_PROVIDER = ctx.providers.OpenStreetMap.Mapnik  # Street map
    # ctx.providers.CartoDB.Positron  # Light theme
    # ctx.providers.CartoDB.DarkMatter  # Dark theme
else:
    SATELLITE_PROVIDER = None


def find_rosbag_directories() -> List[str]:
    """Find all rosbag directories for the selected data source."""
    rosbag_dirs = set()
    if SELECTED_DIR.exists():
        for dir_path in SELECTED_DIR.iterdir():
            if dir_path.is_dir():
                rosbag_dirs.add(dir_path.name)
    return sorted(list(rosbag_dirs))


def extract_gps_data_from_csv(csv_path: Path) -> List[Tuple[float, float, float]]:
    """
    Extract GPS coordinates (lat, lon, alt) from a CSV file.
    Returns list of (latitude, longitude, altitude) tuples.
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
            
            # Find latitude, longitude, altitude columns
            lat_col = None
            lon_col = None
            alt_col = None
            
            #print(f"🔍 Available columns: {list(reader.fieldnames)}")
            
            for col in reader.fieldnames:
                col_lower = col.lower().strip()
                if col_lower == 'lat' or col_lower == 'latitude':
                    lat_col = col
                elif col_lower == 'lon' or col_lower == 'longitude' or col_lower == 'lng':
                    lon_col = col
                elif col_lower == 'hgt' or col_lower == 'alt' or col_lower == 'altitude' or col_lower == 'elevation':
                    alt_col = col
            
            #print(f"🔍 Detected columns - Lat: {lat_col}, Lon: {lon_col}, Alt: {alt_col}")
            
            if not lat_col or not lon_col:
                print(f"⚠️  Could not find lat/lon columns in {csv_path.name}")
                return coordinates
            
            # Read data with sampling
            row_count = 0
            valid_points = 0
            for row in reader:
                if row_count % SAMPLE_EVERY_N == 0:
                    try:
                        lat = float(row[lat_col])
                        lon = float(row[lon_col])
                        alt = float(row[alt_col]) if alt_col and row[alt_col] else 0.0
                    
                        
                        # Basic validation (rough bounds for Germany/Europe)
                        if 40.0 <= lat <= 60.0 and 0.0 <= lon <= 20.0:
                            coordinates.append((lat, lon, alt))
                            valid_points += 1
                    except (ValueError, KeyError) as e:
                        continue
                
                row_count += 1
            
            # Ensure at least one point is selected if any valid data exists
            if valid_points == 0 and row_count > 0:
                # Reset and take every point if sampling resulted in no points
                f.seek(0)
                reader = csv.DictReader(f, delimiter=detected_delimiter)
                for row in reader:
                    try:
                        lat = float(row[lat_col])
                        lon = float(row[lon_col])
                        alt = float(row[alt_col]) if alt_col and row[alt_col] else 0.0
                        
                        # Basic validation (rough bounds for Germany/Europe)
                        if 40.0 <= lat <= 60.0 and 0.0 <= lon <= 20.0:
                            coordinates.append((lat, lon, alt))
                            valid_points += 1
                            # Take only the first valid point to avoid too much data
                            break
                    except (ValueError, KeyError) as e:
                        continue
            
            #print(f"📊 Processed {row_count} rows, found {len(coordinates)} valid coordinates")
                
    except Exception as e:
        print(f"❌ Error reading {csv_path.name}: {e}")
    
    return coordinates


def get_gps_data_for_rosbag(rosbag_name: str) -> List[Tuple[float, float, float]]:
    """Get GPS data for a specific rosbag from the selected directory (fix or bestpos)."""
    all_coordinates = []
    source_dir = SELECTED_DIR / rosbag_name
    if source_dir.exists():
        csv_files = sorted(source_dir.glob("*.csv"))
        if csv_files:
            print(f"📊 {rosbag_name}: Processing {len(csv_files)} {DATA_SOURCE} CSV files...")
            for csv_file in csv_files:
                coords = extract_gps_data_from_csv(csv_file)
                all_coordinates.extend(coords)
                # print per-file points only if needed
                # print(f"   📄 {csv_file.name}: {len(coords)} points")
    return all_coordinates


def create_satellite_map(coordinates: List[Tuple[float, float, float]], rosbag_data: Dict[str, List[Tuple[float, float, float]]], total_csv_files: int = 0):
    """Create a satellite map visualization of GPS coordinates."""
    
    if not coordinates:
        print("❌ No GPS coordinates found!")
        return
    
    # Convert to numpy arrays for easier handling
    lats = np.array([coord[0] for coord in coordinates])
    lons = np.array([coord[1] for coord in coordinates])
    alts = np.array([coord[2] for coord in coordinates])
    
    print(f"🗺️  Creating map with {len(coordinates)} GPS points")
    print(f"📍 Latitude range: {lats.min():.6f} to {lats.max():.6f}")
    print(f"📍 Longitude range: {lons.min():.6f} to {lons.max():.6f}")
    print(f"📍 Altitude range: {alts.min():.1f}m to {alts.max():.1f}m")
    print(f"🎨 Using neon colormap for maximum visibility against satellite imagery")

    # Create figure
    fig, ax = plt.subplots(figsize=MAP_SIZE_INCHES, dpi=DPI)
    
    # Set the extent of the plot based on GPS data with proper aspect ratio
    margin_lat = 0.001  # Small margin around the data (latitude)
    margin_lon = 0.001  # Small margin around the data (longitude)
    
    # Calculate proper longitude margin based on latitude (longitude degrees get smaller at higher latitudes)
    center_lat = (lats.min() + lats.max()) / 2
    lat_correction = np.cos(np.radians(center_lat))
    margin_lon_corrected = margin_lon / lat_correction
    
    ax.set_xlim(lons.min() - margin_lon_corrected, lons.max() + margin_lon_corrected)
    ax.set_ylim(lats.min() - margin_lat, lats.max() + margin_lat)
    
    # Set equal aspect ratio for proper geographic scaling
    ax.set_aspect('equal')
    
    # Add satellite background if enabled
    if USE_SATELLITE_BACKGROUND and CONTEXTILY_AVAILABLE:
        try:
            print("🛰️  Adding high-resolution satellite background...")
            # Convert to Web Mercator projection for contextily with high resolution
            ax_proj = ctx.add_basemap(ax, 
                                    crs='EPSG:4326',  # WGS84 (GPS coordinates)
                                    source=SATELLITE_PROVIDER,
                                    alpha=0.9,  # Less transparency for sharper imagery
                                    zoom=18)  # High zoom level for sharp imagery
            print("✅ High-resolution satellite background added successfully")
        except Exception as e:
            print(f"⚠️  Could not add satellite background: {e}")
            print("   Continuing with plain background...")
    elif USE_SATELLITE_BACKGROUND and not CONTEXTILY_AVAILABLE:
        print("⚠️  Satellite background requested but contextily not available")
        print("   Install with: pip install contextily")
        print("   Continuing with plain background...")
    
    # Create custom neon colormap for high visibility against satellite imagery
    # Colors that contrast well with green/brown satellite imagery
    neon_colors = ['#8A2BE2', '#FF1493' ]  # blue violet, hot pink
    neon_cmap = LinearSegmentedColormap.from_list('neon', neon_colors, N=256)
    
    # Plot GPS points with altitude as color using neon colormap
    scatter = ax.scatter(lons, lats, c=alts, cmap=neon_cmap, s=POINT_SIZE, alpha=ALPHA, edgecolors='none')
    
    # Add colorbar for altitude with proper labels and height
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.2, aspect=20)
    cbar.set_label('Altitude (m)', rotation=270, labelpad=25, fontsize=12)
    
    # Set colorbar ticks to show actual altitude values
    cbar.set_ticks(np.linspace(alts.min(), alts.max(), 6))
    cbar.set_ticklabels([f'{alt:.0f}' for alt in np.linspace(alts.min(), alts.max(), 6)])
    cbar.ax.tick_params(labelsize=10)
    
    # Customize plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Master GPS Map - All Visited Positions', fontsize=16, fontweight='bold')
    
    # Add grid (only if no satellite background)
    if not USE_SATELLITE_BACKGROUND:
        ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""
Total Points: {len(coordinates):,}
Rosbags Processed: {len(rosbag_data)}
CSV Files Processed: {total_csv_files:,}
Sampling Rate: Every {SAMPLE_EVERY_N}th point
    """.strip()
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Tight layout
    plt.tight_layout()
    
    # Save the map
    OUTPUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_MAP, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"💾 Map saved to: {OUTPUT_MAP}")
    
    # Show plot
    plt.show()


def main():
    """Main function to create the master GPS map."""
    print("🗺️  Master GPS Map Creator")
    print("=" * 50)
    
    # Find all processed rosbags
    rosbag_dirs = find_rosbag_directories()
    print(f"📁 Found {len(rosbag_dirs)} processed rosbag directories ({DATA_SOURCE})")
    
    if not rosbag_dirs:
        print("❌ No processed rosbags found!")
        return
    
    # Collect GPS data from all rosbags
    all_coordinates = []
    rosbag_data = {}
    total_csv_files = 0
    
    for rosbag_name in rosbag_dirs:
        print(f"\n🔍 Processing {rosbag_name}...")
        coords = get_gps_data_for_rosbag(rosbag_name)
        
        # Count CSV files processed for this rosbag (selected source only)
        rosbag_csv_count = len(list((SELECTED_DIR / rosbag_name).glob("*.csv"))) if (SELECTED_DIR / rosbag_name).exists() else 0
        total_csv_files += rosbag_csv_count
        
        if len(coords) >= MIN_POINTS_PER_ROSBAG:
            all_coordinates.extend(coords)
            rosbag_data[rosbag_name] = coords
            print(f"✅ Added {len(coords)} points from {rosbag_name} ({rosbag_csv_count} CSV files)")
        else:
            print(f"⚠️  Skipped {rosbag_name}: Only {len(coords)} points (minimum: {MIN_POINTS_PER_ROSBAG}) from {rosbag_csv_count} CSV files")
    
    print(f"\n📊 Summary: Processed {total_csv_files} CSV files from {len(rosbag_dirs)} rosbags")
    
    # Create the map
    if all_coordinates:
        create_satellite_map(all_coordinates, rosbag_data, total_csv_files)
    else:
        print("❌ No valid GPS coordinates found!")


if __name__ == "__main__":
    main()
