#!/usr/bin/env python3
"""
Interactive GPS Map Creator with Visit Frequency Heatmap

This script creates an enhanced GPS map visualization showing visit frequency
as a heatmap, with better outlier handling and interactive HTML output.

Features:
- Visit frequency heatmap (primary visualization)
- Geographic outlier handling with percentile-based focus
- Interactive HTML map with zoom, pan, and layer toggle
- Static PNG output with optional inset map for outliers
- Click markers showing visit counts
- Satellite/street map toggle

Usage:
1. Ensure all rosbags have been processed and CSV files exist
2. Install required packages: pip install contextily folium
3. Run: python3 create_interactive_gps_map.py
"""

import csv
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Tuple, Dict, Optional
import pandas as pd
from datetime import datetime
import json
from collections import defaultdict

try:
    import contextily as ctx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False
    print("⚠️  contextily not available. Install with: pip install contextily")
    print("   Satellite background will be disabled.")

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("⚠️  folium not available. Install with: pip install folium")
    print("   Interactive HTML map will be disabled.")

# After importing contextily (if available)
if CONTEXTILY_AVAILABLE:
    try:
        ctx.set_cache_dir(str(Path.home() / ".cache" / "contextily"))
    except Exception:
        pass

# Configuration
POSITIONAL_DATA_DIR = Path("/home/nepomuk/sflnas/DataReadWrite334/0_shared/Feldschwarm/bagseek/src/positional_data")
OUTPUT_MAP_PNG = Path("/mnt/data/master_gps_map_interactive.png")
OUTPUT_MAP_HTML = Path("/mnt/data/master_gps_map_interactive.html")
LOOKUP_JSON = Path("/mnt/data/bagseek/flask-backend/src/rosbag_gps_lookup.json")

# Select which data source to use: "fix" or "bestpos"
DATA_SOURCE = "bestpos"  # "fix" or "bestpos"

if DATA_SOURCE not in ("fix", "bestpos"):
    raise SystemExit("DATA_SOURCE must be 'fix' or 'bestpos'")

SELECTED_DIR = POSITIONAL_DATA_DIR / DATA_SOURCE

# Sampling configuration
SAMPLE_EVERY_N = 1  # Sample every Nth entry for performance
MIN_POINTS_PER_ROSBAG = 0  # Minimum points required to include a rosbag

# Visit frequency calculation
GRID_RESOLUTION = 0.0001  # Grid size in degrees (~11 meters)

# Outlier handling
FOCUS_ON_CORE = True  # Focus on core area instead of all data
CORE_PERCENTILE = 0.95  # Percentile threshold for core area (0.95 = exclude top/bottom 5%)
SHOW_INSET_MAP = True  # Show inset map with full extent including outliers

# Map configuration
MAP_SIZE_INCHES = (20, 15)  # Width, Height in inches
DPI = 300  # High resolution for detailed map
POINT_SIZE = 5.0  # Size of GPS points (for overlay, not heatmap)
ALPHA = 0.6  # Transparency of heatmap overlay

# Satellite imagery configuration
USE_SATELLITE_BACKGROUND = True  # Set to False for plain background
CREATE_INTERACTIVE_MAP = True  # Generate interactive HTML map
SHOW_ALTITUDE = False  # Deprioritize altitude visualization

if CONTEXTILY_AVAILABLE:
    SATELLITE_PROVIDER = ctx.providers.Esri.WorldImagery  # High-quality satellite imagery
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


def extract_gps_data_from_csv(csv_path: Path) -> List[Tuple[float, float, float, Optional[datetime]]]:
    """
    Extract GPS coordinates (lat, lon, alt) and timestamps from a CSV file.
    Returns list of (latitude, longitude, altitude, timestamp) tuples.
    Timestamp can be None if not found in CSV.
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
            
            # Find latitude, longitude, altitude, and timestamp columns
            lat_col = None
            lon_col = None
            alt_col = None
            timestamp_sec_col = None
            timestamp_nsec_col = None
            timestamp_col = None
            
            for col in reader.fieldnames:
                col_lower = col.lower().strip()
                if col_lower == 'lat' or col_lower == 'latitude':
                    lat_col = col
                elif col_lower == 'lon' or col_lower == 'longitude' or col_lower == 'lng':
                    lon_col = col
                elif col_lower == 'hgt' or col_lower == 'alt' or col_lower == 'altitude' or col_lower == 'elevation':
                    alt_col = col
                elif col_lower == 'header.stamp.sec' or col_lower == 'stamp.sec' or col_lower == 'timestamp_sec':
                    timestamp_sec_col = col
                elif col_lower == 'header.stamp.nanosec' or col_lower == 'stamp.nanosec' or col_lower == 'timestamp_nsec':
                    timestamp_nsec_col = col
                elif col_lower == 'timestamp' or col_lower == 'time' or col_lower == 'header.stamp':
                    timestamp_col = col
            
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
                            # Parse timestamp
                            timestamp = None
                            if timestamp_sec_col and timestamp_nsec_col:
                                try:
                                    sec = int(float(row[timestamp_sec_col]))
                                    nsec = int(float(row[timestamp_nsec_col])) if timestamp_nsec_col in row and row[timestamp_nsec_col] else 0
                                    # Convert ROS2 timestamp (sec + nanosec) to datetime
                                    timestamp = datetime.fromtimestamp(sec + nsec / 1e9)
                                except (ValueError, KeyError):
                                    pass
                            elif timestamp_col:
                                try:
                                    # Try parsing as Unix timestamp (seconds or nanoseconds)
                                    ts_val = float(row[timestamp_col])
                                    if ts_val > 1e12:  # Likely nanoseconds
                                        timestamp = datetime.fromtimestamp(ts_val / 1e9)
                                    else:  # Likely seconds
                                        timestamp = datetime.fromtimestamp(ts_val)
                                except (ValueError, KeyError):
                                    try:
                                        # Try parsing as ISO format string
                                        timestamp = datetime.fromisoformat(row[timestamp_col].replace('Z', '+00:00'))
                                    except (ValueError, KeyError):
                                        pass
                            
                            coordinates.append((lat, lon, alt, timestamp))
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
                            # Parse timestamp (same logic as above)
                            timestamp = None
                            if timestamp_sec_col and timestamp_nsec_col:
                                try:
                                    sec = int(float(row[timestamp_sec_col]))
                                    nsec = int(float(row[timestamp_nsec_col])) if timestamp_nsec_col in row and row[timestamp_nsec_col] else 0
                                    timestamp = datetime.fromtimestamp(sec + nsec / 1e9)
                                except (ValueError, KeyError):
                                    pass
                            elif timestamp_col:
                                try:
                                    ts_val = float(row[timestamp_col])
                                    if ts_val > 1e12:
                                        timestamp = datetime.fromtimestamp(ts_val / 1e9)
                                    else:
                                        timestamp = datetime.fromtimestamp(ts_val)
                                except (ValueError, KeyError):
                                    try:
                                        timestamp = datetime.fromisoformat(row[timestamp_col].replace('Z', '+00:00'))
                                    except (ValueError, KeyError):
                                        pass
                            
                            coordinates.append((lat, lon, alt, timestamp))
                            valid_points += 1
                            # Take only the first valid point to avoid too much data
                            break
                    except (ValueError, KeyError) as e:
                        continue
                
    except Exception as e:
        print(f"❌ Error reading {csv_path.name}: {e}")
    
    return coordinates


def get_gps_data_for_rosbag(rosbag_name: str) -> List[Tuple[float, float, float, Optional[datetime]]]:
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
    return all_coordinates


def calculate_visit_frequencies(coordinates: List[Tuple[float, float, float, Optional[datetime]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate visit frequencies using grid-based binning.
    Works with both old format (lat, lon, alt) and new format (lat, lon, alt, timestamp).
    
    Returns:
        lats: Array of unique latitude grid cells
        lons: Array of unique longitude grid cells  
        frequencies: Array of visit counts per grid cell
        indices: Mapping from original coordinates to grid cells
    """
    lats = np.array([coord[0] for coord in coordinates])
    lons = np.array([coord[1] for coord in coordinates])
    
    # Round coordinates to grid cells
    lat_grid = np.round(lats / GRID_RESOLUTION) * GRID_RESOLUTION
    lon_grid = np.round(lons / GRID_RESOLUTION) * GRID_RESOLUTION
    
    # Create unique grid cells and count frequencies
    grid_cells = np.column_stack((lat_grid, lon_grid))
    unique_cells, indices, frequencies = np.unique(grid_cells, axis=0, return_inverse=True, return_counts=True)
    
    unique_lats = unique_cells[:, 0]
    unique_lons = unique_cells[:, 1]
    
    return unique_lats, unique_lons, frequencies, indices


def calculate_percentile_bounds(lats: np.ndarray, lons: np.ndarray, percentile: float = 0.95) -> Tuple[float, float, float, float]:
    """
    Calculate percentile-based bounds to focus on core area.
    
    Returns:
        lat_min, lat_max, lon_min, lon_max within percentile bounds
    """
    lower_percentile = (1 - percentile) / 2
    upper_percentile = 1 - lower_percentile
    
    lat_min = np.percentile(lats, lower_percentile * 100)
    lat_max = np.percentile(lats, upper_percentile * 100)
    lon_min = np.percentile(lons, lower_percentile * 100)
    lon_max = np.percentile(lons, upper_percentile * 100)
    
    return lat_min, lat_max, lon_min, lon_max


def create_static_map(coordinates: List[Tuple[float, float, float, Optional[datetime]]], rosbag_data: Dict[str, List[Tuple[float, float, float, Optional[datetime]]]], total_csv_files: int = 0):
    """Create a static PNG map visualization with visit frequency heatmap."""
    
    if not coordinates:
        print("❌ No GPS coordinates found!")
        return
    
    # Convert to numpy arrays
    lats = np.array([coord[0] for coord in coordinates])
    lons = np.array([coord[1] for coord in coordinates])
    alts = np.array([coord[2] for coord in coordinates])
    
    # Calculate visit frequencies
    print("📊 Calculating visit frequencies...")
    unique_lats, unique_lons, frequencies, indices = calculate_visit_frequencies(coordinates)
    
    print(f"🗺️  Creating map with {len(coordinates)} GPS points")
    print(f"📍 Latitude range: {lats.min():.6f} to {lats.max():.6f}")
    print(f"📍 Longitude range: {lons.min():.6f} to {lons.max():.6f}")
    print(f"📍 Visit frequency range: {frequencies.min()} to {frequencies.max()}")
    print(f"📍 Unique locations: {len(unique_lats)}")
    
    # Calculate bounds (core area or full extent)
    if FOCUS_ON_CORE:
        lat_min, lat_max, lon_min, lon_max = calculate_percentile_bounds(lats, lons, CORE_PERCENTILE)
        full_lat_min, full_lat_max, full_lon_min, full_lon_max = lats.min(), lats.max(), lons.min(), lons.max()
        print(f"📍 Core area (percentile {CORE_PERCENTILE:.0%}): lat {lat_min:.6f}-{lat_max:.6f}, lon {lon_min:.6f}-{lon_max:.6f}")
    else:
        lat_min, lat_max, lon_min, lon_max = lats.min(), lats.max(), lons.min(), lons.max()
        full_lat_min, full_lat_max, full_lon_min, full_lon_max = lat_min, lat_max, lon_min, lon_max
    
    # Create figure with optional inset
    if FOCUS_ON_CORE and SHOW_INSET_MAP:
        fig = plt.figure(figsize=MAP_SIZE_INCHES, dpi=DPI)
        ax_main = fig.add_axes([0.1, 0.1, 0.75, 0.85])  # Main map area
        ax_inset = fig.add_axes([0.67, 0.02, 0.3, 0.25])  # Inset map area
    else:
        fig, ax_main = plt.subplots(figsize=MAP_SIZE_INCHES, dpi=DPI)
        ax_inset = None
    
    # Calculate margins
    margin_lat = (lat_max - lat_min) * 0.02
    margin_lon = (lon_max - lon_min) * 0.02
    center_lat = (lat_min + lat_max) / 2
    lat_correction = np.cos(np.radians(center_lat))
    margin_lon_corrected = margin_lon / lat_correction
    
    # Set main map bounds
    ax_main.set_xlim(lon_min - margin_lon_corrected, lon_max + margin_lon_corrected)
    ax_main.set_ylim(lat_min - margin_lat, lat_max + margin_lat)
    ax_main.set_aspect('equal')
    
    # Add satellite background to main map
    if USE_SATELLITE_BACKGROUND and CONTEXTILY_AVAILABLE:
        try:
            print("🛰️  Adding satellite background to main map...")
            ctx.add_basemap(ax_main, 
                          crs='EPSG:4326',
                          source=SATELLITE_PROVIDER,
                          alpha=0.9,
                          zoom='auto')
            print("✅ Satellite background added successfully")
        except Exception as e:
            print(f"⚠️  Could not add satellite background: {e}")
    
    # Filter coordinates to core area for heatmap
    if FOCUS_ON_CORE:
        mask = (lats >= lat_min) & (lats <= lat_max) & \
               (lons >= lon_min) & (lons <= lon_max)
        plot_lats = lats[mask]
        plot_lons = lons[mask]
    else:
        plot_lats = lats
        plot_lons = lons
    
    # Create heatmap using hexbin (automatically counts density)
    print("🎨 Creating visit frequency heatmap...")
    hexbin = ax_main.hexbin(plot_lons, plot_lats, 
                           gridsize=50, cmap='YlOrRd', 
                           mincnt=1, alpha=ALPHA, edgecolors='none')
    
    # Add colorbar
    cbar = plt.colorbar(hexbin, ax=ax_main, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Visit Frequency', rotation=270, labelpad=25, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Customize main plot
    ax_main.set_xlabel('Longitude', fontsize=12)
    ax_main.set_ylabel('Latitude', fontsize=12)
    title = 'GPS Visit Frequency Heatmap'
    if FOCUS_ON_CORE:
        title += f' (Core Area, {CORE_PERCENTILE:.0%} percentile)'
    ax_main.set_title(title, fontsize=16, fontweight='bold')
    
    if not USE_SATELLITE_BACKGROUND:
        ax_main.grid(True, alpha=0.3)
    
    # Add inset map showing full extent
    if ax_inset is not None:
        ax_inset.set_xlim(full_lon_min, full_lon_max)
        ax_inset.set_ylim(full_lat_min, full_lat_max)
        ax_inset.set_aspect('equal')
        
        # Draw all points as light dots in inset
        ax_inset.scatter(lons, lats, c='blue', s=0.5, alpha=0.1, edgecolors='none')
        
        # Draw rectangle showing core area
        rect = Rectangle((lon_min, lat_min), 
                        lon_max - lon_min, lat_max - lat_min,
                        linewidth=2, edgecolor='red', facecolor='none')
        ax_inset.add_patch(rect)
        
        ax_inset.set_title('Full Extent', fontsize=8)
        ax_inset.tick_params(labelsize=6)
    
    # Add statistics text
    stats_text = f"""
Total Points: {len(coordinates):,}
Unique Locations: {len(unique_lats):,}
Max Visits: {frequencies.max()}
Rosbags: {len(rosbag_data)}
CSV Files: {total_csv_files:,}
Grid Size: {GRID_RESOLUTION*111000:.0f}m
    """.strip()
    
    ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the map
    OUTPUT_MAP_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_MAP_PNG, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"💾 Static map saved to: {OUTPUT_MAP_PNG}")
    
    plt.close()


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


def load_lookup_table() -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """Load the JSON lookup table and return lookup dict and sorted rosbag list."""
    if not LOOKUP_JSON.exists():
        print(f"⚠️  Lookup JSON not found at {LOOKUP_JSON}")
        print("   Please run create_rosbag_gps_lookup.py first to generate it.")
        return {}, []
    
    try:
        with open(LOOKUP_JSON, 'r', encoding='utf-8') as f:
            raw_lookup = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"❌ Failed to parse lookup JSON: {exc}")
        return {}, []
    
    lookup: Dict[str, Dict[str, int]] = {}
    
    for rosbag_name, locations in raw_lookup.items():
        if not isinstance(locations, dict):
            lookup[rosbag_name] = {}
            continue
        
        sanitized_locations: Dict[str, int] = {}
        for lat_lon_key, count in locations.items():
            try:
                sanitized_locations[str(lat_lon_key)] = int(count)
            except (TypeError, ValueError):
                # Skip entries that can't be interpreted as integers
                continue
        lookup[rosbag_name] = sanitized_locations
    
    # Sort rosbags by timestamp
    def sort_key(name: str) -> Tuple[datetime, str]:
        ts = parse_rosbag_timestamp(name)
        if ts:
            return (ts, name)
        return (datetime.min, name)
    
    sorted_rosbags = sorted(lookup.keys(), key=sort_key)
    
    return lookup, sorted_rosbags


def create_interactive_map(coordinates: List[Tuple[float, float, float, Optional[datetime]]], rosbag_data: Dict[str, List[Tuple[float, float, float, Optional[datetime]]]], total_csv_files: int = 0):
    """Create an interactive HTML map with Folium and rosbag filtering."""
    
    if not FOLIUM_AVAILABLE:
        print("⚠️  Folium not available, skipping interactive map creation")
        return
    
    if not coordinates:
        print("❌ No GPS coordinates found!")
        return
    
    print("🌐 Creating interactive HTML map...")
    
    # Load lookup table
    print("📊 Loading rosbag GPS lookup table...")
    lookup_table, sorted_rosbags = load_lookup_table()
    
    if not lookup_table:
        print("❌ No lookup table data available!")
        return
    
    print(f"✅ Loaded lookup table with {len(sorted_rosbags)} rosbags")
    
    # Convert to numpy arrays
    lats = np.array([coord[0] for coord in coordinates])
    lons = np.array([coord[1] for coord in coordinates])
    
    # Calculate visit frequencies for all data (initial view)
    unique_lats, unique_lons, frequencies, indices = calculate_visit_frequencies(coordinates)
    
    # Calculate center and bounds
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    if FOCUS_ON_CORE:
        lat_min, lat_max, lon_min, lon_max = calculate_percentile_bounds(lats, lons, CORE_PERCENTILE)
    else:
        lat_min, lat_max, lon_min, lon_max = lats.min(), lats.max(), lons.min(), lons.max()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None  # We'll add tiles as layers
    )
    
    # Add tile layers
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Prepare heatmap data
    # Format: [[lat, lon, weight], ...]
    heatmap_data = [[lat, lon, int(freq)] for lat, lon, freq in zip(unique_lats, unique_lons, frequencies)]
    
    # Create heatmap layer (will be updated by JavaScript)
    heatmap_layer = HeatMap(
        heatmap_data,
        min_opacity=0.2,
        max_zoom=18,
        radius=15,
        blur=15,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},
        name='Visit Frequency Heatmap',
        show=True,
        overlay=True,
        control=True
    )
    heatmap_layer.add_to(m)
    
    # Add markers for high-frequency locations (top 5% by visit count)
    top_percentile = np.percentile(frequencies, 95)
    top_mask = frequencies >= top_percentile
    top_lats = unique_lats[top_mask]
    top_lons = unique_lons[top_mask]
    top_freqs = frequencies[top_mask]
    
    # Create marker cluster for high-frequency locations
    marker_cluster = MarkerCluster(name='High Frequency Locations', overlay=True, control=True).add_to(m)
    
    for lat, lon, freq in zip(top_lats, top_lons, top_freqs):
        popup_text = f"<b>Visit Count:</b> {freq}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=f"Visited {freq} times",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Fit bounds to data
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])
    
    # Add title/legend
    title_html = f'''
    <div id="map-info-panel" style="position: fixed; 
                top: 10px; left: 50px; width: 300px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius:5px; padding: 10px">
    <h4 style="margin-top:0">GPS Visit Frequency Map</h4>
    <p id="stats-display" style="margin:0"><b>Total Points:</b> {len(coordinates):,}<br>
    <b>Unique Locations:</b> {len(unique_lats):,}<br>
    <b>Max Visits:</b> {frequencies.max()}</p>
    <p id="rosbag-display" style="margin:5px 0 0 0; font-size:12px; color:#666;">Select a rosbag below</p>
    </div>
    '''
    
    # Rosbag slider HTML at bottom
    rosbag_slider_html = f'''
    <div id="rosbag-slider-panel" style="position: fixed; 
                bottom: 20px; left: 50%; transform: translateX(-50%); width: 80%; max-width: 1200px;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius:5px; padding: 15px; box-shadow: 0 -2px 10px rgba(0,0,0,0.3)">
    <h4 style="margin-top:0; margin-bottom:10px;">Select Rosbag (sorted by time)</h4>
    <div style="display: flex; align-items: center; gap: 15px;">
        <label for="rosbag-slider" style="min-width: 120px; font-weight: bold;">Rosbag:</label>
        <input type="range" id="rosbag-slider" min="0" max="{len(sorted_rosbags) - 1}" value="0" 
               style="flex: 1; height: 8px;" step="1">
        <span id="rosbag-name-display" style="min-width: 250px; font-family: monospace; font-size: 12px;">
            {sorted_rosbags[0] if sorted_rosbags else 'No rosbags'}
        </span>
        <button id="show-all-rosbags" style="padding: 5px 15px; cursor: pointer; font-size: 12px;">Show All</button>
    </div>
    <div id="rosbag-stats" style="margin-top: 8px; font-size: 11px; color: #666;"></div>
    </div>
    '''
    
    # JavaScript for rosbag filtering
    js_code = f'''
    <script>
    (function() {{
        // Embed lookup table data
        const lookupTable = {json.dumps(lookup_table)};
        const sortedRosbags = {json.dumps(sorted_rosbags)};
        const GRID_RESOLUTION = {GRID_RESOLUTION};
        
        // Reference to map and layers
        let mapInstance = null;
        let currentHeatmapLayer = null;
        let currentMarkerCluster = null;
        let showingAll = true;
        
        // Get map instance
        function getMap() {{
            if (mapInstance) return mapInstance;
            
            // Try different ways to get the map
            if (typeof window._map !== 'undefined') {{
                mapInstance = window._map;
            }} else {{
                // Find map container
                const mapContainer = document.querySelector('.folium-map');
                if (mapContainer && mapContainer._leaflet_id) {{
                    mapInstance = L.map._instances[mapContainer._leaflet_id];
                }} else {{
                    // Last resort: find first map instance
                    const maps = Object.values(L.map._instances || {{}});
                    if (maps.length > 0) {{
                        mapInstance = maps[0];
                    }}
                }}
            }}
            return mapInstance;
        }}
        
        // Remove all heatmap and marker cluster layers
        function removeAllHeatmapAndMarkerLayers() {{
            const map = getMap();
            if (!map) {{
                console.warn('Cannot remove layers: map not found');
                return;
            }}
            
            const layersToRemove = [];
            
            // Collect layers to remove (can't remove while iterating)
            map.eachLayer(function(layer) {{
                // Check for heat layers - Leaflet.heat adds _heat property
                if (layer._heat) {{
                    layersToRemove.push(layer);
                }}
                // Check for marker clusters
                if (layer.hasOwnProperty('_maxZoom') && layer.hasOwnProperty('_featureGroup')) {{
                    // MarkerClusterGroup detection
                    layersToRemove.push(layer);
                }}
            }});
            
            // Remove collected layers
            layersToRemove.forEach(function(layer) {{
                try {{
                    if (map.hasLayer(layer)) {{
                        map.removeLayer(layer);
                    }}
                }} catch (e) {{
                    console.warn('Error removing layer:', e);
                }}
            }});
        }}
        
        function updateMapForRosbag(rosbagIndex) {{
            const map = getMap();
            if (!map) {{
                console.error('Map not found');
                return;
            }}
            
            if (rosbagIndex < 0 || rosbagIndex >= sortedRosbags.length) {{
                console.error('Invalid rosbag index:', rosbagIndex);
                return;
            }}
            
            const selectedRosbag = sortedRosbags[rosbagIndex];
            const rosbagData = lookupTable[selectedRosbag] || {{}};
            
            // Update display
            document.getElementById('rosbag-name-display').textContent = selectedRosbag;
            
            // Remove existing layers
            removeAllHeatmapAndMarkerLayers();
            
            // Small delay to ensure layers are removed
            setTimeout(function() {{
                // Convert lookup data to heatmap format
                const heatmapData = [];
                let totalVisits = 0;
                let maxVisits = 0;
                
                for (const [latLonKey, count] of Object.entries(rosbagData)) {{
                    const [lat, lon] = latLonKey.split(',').map(parseFloat);
                    heatmapData.push([lat, lon, count]);
                    totalVisits += count;
                    maxVisits = Math.max(maxVisits, count);
                }}
                
                if (heatmapData.length === 0) {{
                    document.getElementById('rosbag-stats').innerHTML = 
                        '<span style="color: red;">No GPS data for this rosbag</span>';
                    document.getElementById('stats-display').innerHTML = 
                        '<b>Selected Rosbag:</b> ' + selectedRosbag + '<br>' +
                        '<b>GPS Places:</b> 0';
                    return;
                }}
                
                // Add heatmap layer
                if (typeof L.heatLayer !== 'undefined') {{
                    currentHeatmapLayer = L.heatLayer(heatmapData, {{
                        minOpacity: 0.2,
                        maxZoom: 18,
                        radius: 15,
                        blur: 15,
                        gradient: {{0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}}
                    }});
                    currentHeatmapLayer.addTo(map);
                }}
                
                // Add markers for top locations
                const sortedData = heatmapData.sort((a, b) => b[2] - a[2]);
                const topPercentileIndex = Math.floor(sortedData.length * 0.05);
                const topThreshold = sortedData[topPercentileIndex] ? sortedData[topPercentileIndex][2] : 0;
                
                if (typeof L.markerClusterGroup !== 'undefined') {{
                    currentMarkerCluster = L.markerClusterGroup({{name: 'High Frequency Locations'}});
                    for (const [lat, lon, count] of sortedData) {{
                        if (count >= topThreshold) {{
                            const marker = L.marker([lat, lon], {{
                                icon: L.icon({{
                                    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
                                    iconSize: [25, 41],
                                    iconAnchor: [12, 41],
                                    popupAnchor: [1, -34]
                                }})
                            }});
                            marker.bindPopup('<b>Visit Count:</b> ' + count + '<br>Lat: ' + lat.toFixed(6) + '<br>Lon: ' + lon.toFixed(6));
                            marker.bindTooltip('Visited ' + count + ' times');
                            currentMarkerCluster.addLayer(marker);
                        }}
                    }}
                    currentMarkerCluster.addTo(map);
                }}
                
                // Update stats
                document.getElementById('rosbag-stats').innerHTML = 
                    '<b>GPS Places:</b> ' + heatmapData.length + ' | ' +
                    '<b>Total Visits:</b> ' + totalVisits + ' | ' +
                    '<b>Max Visits:</b> ' + maxVisits;
                document.getElementById('stats-display').innerHTML = 
                    '<b>Selected Rosbag:</b> ' + selectedRosbag + '<br>' +
                    '<b>GPS Places:</b> ' + heatmapData.length.toLocaleString() + '<br>' +
                    '<b>Total Visits:</b> ' + totalVisits.toLocaleString() + '<br>' +
                    '<b>Max Visits:</b> ' + maxVisits;
            }}, 100);
        }}
        
        function showAllRosbags() {{
            const map = getMap();
            if (!map) return;
            
            showingAll = true;
            document.getElementById('rosbag-name-display').textContent = 'All Rosbags';
            document.getElementById('rosbag-stats').innerHTML = '<span style="color: #666;">Showing all rosbags combined</span>';
            
            // Remove existing layers
            removeAllHeatmapAndMarkerLayers();
            
            // Reload page or recreate initial view
            // For now, just show message - user can reload page to see all data
            document.getElementById('stats-display').innerHTML = 
                '<b>Showing:</b> All Rosbags<br>' +
                '<b>Total Points:</b> {len(coordinates):,}<br>' +
                '<b>Unique Locations:</b> {len(unique_lats):,}<br>' +
                '<b>Max Visits:</b> {frequencies.max()}';
        }}
        
        // Initialize when DOM is ready
        function initRosbagSlider() {{
            setTimeout(function() {{
                const slider = document.getElementById('rosbag-slider');
                const showAllBtn = document.getElementById('show-all-rosbags');
                
                if (slider) {{
                    slider.addEventListener('input', function() {{
                        showingAll = false;
                        updateMapForRosbag(parseInt(this.value));
                    }});
                }}
                
                if (showAllBtn) {{
                    showAllBtn.addEventListener('click', showAllRosbags);
                }}
            }}, 500);
        }}
        
        // Initialize when page loads
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initRosbagSlider);
        }} else {{
            initRosbagSlider();
        }}
    }})();
    </script>
    '''
    
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(rosbag_slider_html))
    m.get_root().html.add_child(folium.Element(js_code))
    
    # Save HTML file
    OUTPUT_MAP_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUTPUT_MAP_HTML))
    
    # Inject Leaflet.heat plugin if not already present
    html_content = OUTPUT_MAP_HTML.read_text()
    
    # Add Leaflet.heat plugin if needed
    if 'leaflet-heat' not in html_content.lower() and 'leaflet.heat' not in html_content.lower():
        leaflet_heat_script = '''
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    '''
        html_content = html_content.replace('</head>', leaflet_heat_script + '</head>')
    
    OUTPUT_MAP_HTML.write_text(html_content)
    
    print(f"💾 Interactive map saved to: {OUTPUT_MAP_HTML}")
    print(f"📂 Open in browser: file://{OUTPUT_MAP_HTML.absolute()}")
    print(f"📊 Rosbag filtering enabled with {len(sorted_rosbags)} rosbags")


def main():
    """Main function to create the enhanced GPS map."""
    print("🗺️  Interactive GPS Map Creator with Visit Frequency Heatmap")
    print("=" * 70)
    
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
        
        # Count CSV files processed for this rosbag
        rosbag_csv_count = len(list((SELECTED_DIR / rosbag_name).glob("*.csv"))) if (SELECTED_DIR / rosbag_name).exists() else 0
        total_csv_files += rosbag_csv_count
        
        if len(coords) >= MIN_POINTS_PER_ROSBAG:
            all_coordinates.extend(coords)
            rosbag_data[rosbag_name] = coords
            print(f"✅ Added {len(coords)} points from {rosbag_name} ({rosbag_csv_count} CSV files)")
        else:
            print(f"⚠️  Skipped {rosbag_name}: Only {len(coords)} points (minimum: {MIN_POINTS_PER_ROSBAG}) from {rosbag_csv_count} CSV files")
    
    print(f"\n📊 Summary: Processed {total_csv_files} CSV files from {len(rosbag_dirs)} rosbags")
    
    # Create maps
    if all_coordinates:
        create_static_map(all_coordinates, rosbag_data, total_csv_files)
        
        if CREATE_INTERACTIVE_MAP:
            create_interactive_map(all_coordinates, rosbag_data, total_csv_files)
    else:
        print("❌ No valid GPS coordinates found!")


if __name__ == "__main__":
    main()

