"""
Step 3: Positional Lookup

Builds positional lookup tables from position/GPS messages.
Hybrid processor that collects per MCAP and aggregates per rosbag.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from ..abstract import HybridProcessor
from ..core import McapProcessingContext, RosbagProcessingContext
from ..utils import CompletionTracker, PipelineLogger, get_logger


class PositionalLookupProcessor(HybridProcessor):
    """
    Build aggregated positional lookup table for a rosbag.
    
    Hybrid processor that:
    - Collects position messages during MCAP iteration
    - Aggregates positions per rosbag after all MCAPs
    - Finalizes by combining all rosbag JSONs into one big file
    """
    
    def __init__(self, output_dir: Path, positional_grid_resolution: float = 0.0001):
        """
        Initialize positional lookup step.
        
        Args:
            output_dir: Full path to positional lookup table JSON file
            positional_grid_resolution: Step size in degrees (for spatial indexing, ~11 meters)
        """
        super().__init__("positional_lookup_processor")
        self.output_dir = Path(output_dir)  # Now expects full file path
        self.positional_grid_resolution = positional_grid_resolution  # 0.0001 degrees (~11 meters)
        self.logger: PipelineLogger = get_logger()
        self.completion_tracker = CompletionTracker(self.output_dir.parent, processor_name="positional_lookup_processor")
        self.boundaries_tracker = CompletionTracker(self.output_dir.parent, processor_name="positional_boundaries_processor")
        self.positions: List[Dict[str, Any]] = []  # Store collected position data (accumulated across MCAPs)
        self.current_mcap_id: Optional[str] = None  # Track which MCAP we're currently collecting from
        self.combined_data: Dict[str, Dict] = {}  # Accumulate all rosbag data in memory for single file output
        self.processed_rosbags: set = set()  # Track which rosbags have been processed
        self.mcap_message_counts: Dict[str, int] = defaultdict(int)  # Track message counts per MCAP
        self.collected_topic: Optional[str] = None  # Track which topic we're collecting

    def wants_message(self, topic: str, msg_type: str) -> bool:
        """Collect position messages."""
        if "bestpos" in topic.lower():
            if self.collected_topic is None:
                self.collected_topic = topic
            return True
        return False

    def collect_message(self, message: Any, channel: Any, schema: Any, ros2_msg: Any) -> None:
        """
        Collect a position message from bestpos topic.
        
        Stores position with MCAP ID to track which MCAP each position came from.
        
        Args:
            message: MCAP message (decoded ROS2 message)
            channel: MCAP channel info
            schema: MCAP schema info
            ros2_msg: Decoded ROS2 message
        """
        # Extract position data from message
        try:
            lat = ros2_msg.lat
            lon = ros2_msg.lon

            if lat == 0.0 or lon == 0.0:
                return  
            
        except (AttributeError, ValueError) as e:
            self.logger.warning(f"Failed to extract position from message: {e}")
            return
        
        # Store position with MCAP ID for aggregation
        # Note: current_mcap_id should be set before collection via reset() or process_mcap()
        self.positions.append({
            'latitude': lat,
            'longitude': lon,
            'mcap_id': self.current_mcap_id  # Track which MCAP this came from
        })
        
        # Track message count per MCAP
        if self.current_mcap_id:
            self.mcap_message_counts[self.current_mcap_id] += 1
    
    def get_data(self) -> Dict[str, List]:
        """
        Get collected position messages.
        
        Returns:
            Dictionary with collected positions
        """
        return {"positions": self.positions}

    def process_rosbag_before_mcaps(self, context: RosbagProcessingContext) -> None:
        """
        Initialize processing for a new rosbag.
        
        Called before MCAP iteration starts. Prepares state for collection.
        Note: Completion is checked in process_rosbag_after_mcaps after verifying data exists.
        
        Args:
            context: RosbagProcessingContext
        """
        # Clear positions for this rosbag (will accumulate across MCAPs)
        self.positions = []
        self.current_mcap_id = None
        self.mcap_message_counts = defaultdict(int)
        self.collected_topic = None
        
        # Log that we're starting collection
        self.logger.info(f"Collecting bestpos messages for positional lookup...")
    
    def reset(self) -> None:
        """
        Reset collector state before each MCAP iteration.
        
        Note: We DON'T clear positions - we want to accumulate across all MCAPs
        for rosbag-level aggregation. Only reset the current MCAP ID.
        """
        # Don't clear self.positions - accumulate across MCAPs
        # Just reset the current MCAP ID (will be set via context in collect_message or process_mcap)
        self.current_mcap_id = None
    
    def process_mcap(self, context: McapProcessingContext) -> None:
        """
        Mark MCAP as completed after message collection.
        
        Called after messages from this MCAP have been collected.
        Marks completion per MCAP while still aggregating into rosbag-level JSON.
        
        Args:
            context: MCAP processing context
        """
        # Get rosbag name and mcap name
        rosbag_name = str(context.get_relative_path())
        mcap_name = context.get_mcap_name()
        
        # Mark this MCAP as completed (no output_files at MCAP level - they contribute to rosbag-level file)
        self.completion_tracker.mark_completed(
            rosbag_name=rosbag_name,
            mcap_name=mcap_name,
            status="completed"
        )
    
    def is_mcap_completed(self, context: McapProcessingContext) -> bool:
        """
        Check if a specific MCAP is completed by checking completion.json.
        
        For PositionalLookupProcessor, completion.json is the source of truth since
        we cannot reliably determine if a specific MCAP was processed by looking at
        the positional_lookup_table.json (all MCAPs write to the same aggregated file).
        
        Args:
            context: MCAP processing context
            
        Returns:
            True if this MCAP is marked as completed in completion.json, False otherwise
        """
        rosbag_name = str(context.get_relative_path())
        mcap_name = context.get_mcap_name()
        
        # Use new unified interface
        return self.completion_tracker.is_mcap_completed(rosbag_name, mcap_name)
    
    def _verify_mcap_data_in_json(self, rosbag_name: str, mcap_id: str) -> bool:
        """
        Verify that a specific MCAP's data exists in the JSON file.
        
        Args:
            rosbag_name: Name of the rosbag
            mcap_id: ID of the MCAP (e.g., "0", "1", "669")
            
        Returns:
            True if this MCAP's data is present in the JSON file
        """
        if not self.output_dir.exists():
            return False
        
        try:
            with open(self.output_dir, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if rosbag entry exists
            if rosbag_name not in data:
                return False
            
            rosbag_data = data[rosbag_name]
            if not isinstance(rosbag_data, dict):
                return False
            
            # Check if any location entry has this MCAP ID in its mcaps dict
            for location_data in rosbag_data.values():
                if isinstance(location_data, dict) and "mcaps" in location_data:
                    mcaps = location_data["mcaps"]
                    if isinstance(mcaps, dict) and mcap_id in mcaps:
                        # Found this MCAP's data
                        return True
            
            return False
        except (json.JSONDecodeError, IOError):
            return False
    
    def get_output_path(self, context: Union[RosbagProcessingContext, McapProcessingContext]) -> Path:
        """Get the expected output path for this context (combined file)."""
        return self.output_dir
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected messages per MCAP.
        
        Returns:
            Dictionary with total positions, counts per MCAP, and topic name
        """
        return {
            "total_positions": len(self.positions),
            "mcap_counts": dict(self.mcap_message_counts),
            "topic": self.collected_topic or "bestpos (unknown)"
        }
    
    def process_rosbag_after_mcaps(self, context: RosbagProcessingContext) -> Dict:
        """
        Process collected position messages from ALL MCAPs and build aggregated lookup.
        
        Aggregates positions from all MCAPs into one rosbag-level lookup table.
        Accumulates data in memory instead of writing per-rosbag files.
        Structure: {"lat,lon": {"total": X, "mcaps": {"0": Y, "4": Z}}}
        
        Args:
            context: Rosbag processing context
        
        Returns:
            Aggregated positional lookup table with grid-based location counts
        """
        # Get rosbag name for storing in combined data
        rosbag_name = context.get_relative_path().as_posix()  # Use relative path as key
        
        # Read existing data from JSON file
        existing_data = {}
        if self.output_dir.exists():
            try:
                with open(self.output_dir, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Error reading combined file: {e}")
                # Continue with empty dict - will create new file
        
        # If no positions collected, skip processing
        if len(self.positions) == 0:
            self.logger.warning(f"No positions collected for {rosbag_name}, skipping aggregation")
            return {}
        
        # Calculate breakdown by MCAP
        mcap_breakdown = {}
        for pos in self.positions:
            mcap_id = pos.get('mcap_id')
            if mcap_id is None:
                mcap_id = 'unknown'
            else:
                mcap_id = str(mcap_id)  # Convert to string for consistency
            mcap_breakdown[mcap_id] = mcap_breakdown.get(mcap_id, 0) + 1
        
        # Log detailed summary
        topic_name = self.collected_topic or "bestpos"
        self.logger.info(f"Building aggregated positional lookup for {context.get_relative_path()}...")
        self.logger.info(f"  Collected {len(self.positions)} positions from {len(mcap_breakdown)} MCAP(s) via {topic_name}")
        if len(mcap_breakdown) > 0:
            breakdown_str = ", ".join([f"MCAP {mcap_id}: {count}" for mcap_id, count in sorted(mcap_breakdown.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)])
            self.logger.info(f"  Breakdown: {breakdown_str}")
        
        # Aggregate counts per location per MCAP
        # Structure: {lat_lon_key: {mcap_id: count}}
        location_counts = defaultdict(lambda: defaultdict(int))
        
        for pos in self.positions:
            lat_grid, lon_grid = self._round_to_grid(pos['latitude'], pos['longitude'])
            lat_lon_key = f"{lat_grid:.6f},{lon_grid:.6f}"
            # Ensure mcap_id is always a string (convert None to 'unknown')
            mcap_id = pos.get('mcap_id')
            if mcap_id is None:
                mcap_id = 'unknown'
            else:
                mcap_id = str(mcap_id)  # Convert to string
            location_counts[lat_lon_key][mcap_id] += 1
        
        # Convert to final structure with total and mcaps dict
        final_location_data = {}
        for lat_lon_key, mcap_counts in location_counts.items():
            total_count = sum(mcap_counts.values())
            final_location_data[lat_lon_key] = {
                "total": total_count,
                "mcaps": dict(mcap_counts)  # Convert defaultdict to regular dict
            }
        
        # Add new rosbag data
        existing_data[rosbag_name] = final_location_data
        
        # Write back immediately (atomic write)
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        # Verify the write succeeded by reading it back
        try:
            with open(self.output_dir, 'r', encoding='utf-8') as f:
                verify_data = json.load(f)
            if rosbag_name not in verify_data or not verify_data[rosbag_name]:
                raise ValueError(f"Data verification failed for {rosbag_name}")
        except Exception as e:
            self.logger.error(f"Failed to verify written data for {rosbag_name}: {e}")
            return {}
        
        # Also keep in memory for tracking (optional, for finalize summary)
        self.combined_data[rosbag_name] = final_location_data
        self.processed_rosbags.add(rosbag_name)

        self.logger.success(f"Built aggregated lookup with {len(location_counts)} grid cells from {len(self.positions)} positions")
        self.logger.info(f"  Incrementally written to {self.output_dir}")

        # Compute convex hull boundary for fast spatial pre-filtering
        self._compute_and_write_boundaries(rosbag_name, final_location_data)
        
        # Mark rosbag as completed (output file is at rosbag level)
        self.completion_tracker.mark_completed(
            rosbag_name=rosbag_name,
            status="completed",
            output_files=[self.output_dir]
        )
        
        # Clear positions after processing this rosbag (prepare for next rosbag)
        self.positions = []
        self.current_mcap_id = None
        self.mcap_message_counts = defaultdict(int)
        self.collected_topic = None
        
        return final_location_data
    
    def finalize(self) -> None:
        """
        Finalize positional lookup processing.
        
        Called after all rosbags have been processed.
        Since data is written incrementally after each rosbag, this method
        just provides a summary of what was processed.
        """
        if not self.output_dir.exists():
            self.logger.warning("No positional lookup file found to finalize")
            return
        
        # Read the file to get final count
        try:
            with open(self.output_dir, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
            
            total_rosbags = len(final_data)
            total_locations = sum(len(rosbag_data) for rosbag_data in final_data.values())
            
            self.logger.info(f"Finalizing positional lookup: {total_rosbags} rosbag(s) with {total_locations} total location entries in {self.output_dir}")
            self.logger.success(f"Positional lookup complete: {self.output_dir}")
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error reading combined file for finalization summary: {e}")


    def _compute_concave_hull(self, points: List[Tuple[float, float]]) -> List[List[List[float]]]:
        """
        Compute concave hull (alpha shape) using Delaunay triangulation.

        Filters out triangles whose longest edge exceeds an alpha threshold
        based on grid resolution, then extracts all boundary loops.
        Returns a multi-polygon (list of polygons) to handle disconnected
        coverage areas (e.g. separate field paths).

        Args:
            points: List of (lat, lon) tuples

        Returns:
            List of polygon components, each as [[lat, lon], ...]
        """
        import numpy as np
        from collections import defaultdict

        pts = sorted(set(points))
        if len(pts) <= 2:
            return [[[p[0], p[1]] for p in pts]]

        if len(pts) < 4:
            return [[[p[0], p[1]] for p in pts]]

        try:
            from scipy.spatial import Delaunay
        except ImportError:
            self.logger.warning("scipy not available, falling back to convex hull")
            return [self._compute_convex_hull_simple(pts)]

        coords = np.array(pts)
        tri = Delaunay(coords)

        # Alpha = 2.5x grid resolution.
        # Grid diagonal is ~1.41x grid_res, so this keeps direct neighbors
        # and skip-one horizontal/vertical (2x grid_res) but cuts skip-one
        # diagonal (2.83x grid_res), producing tight concave shapes.
        alpha = self.positional_grid_resolution * 2.5

        # Filter triangles: keep those where all edges are shorter than alpha
        boundary_edges = defaultdict(int)
        for simplex in tri.simplices:
            verts = coords[simplex]
            edges = [
                np.linalg.norm(verts[0] - verts[1]),
                np.linalg.norm(verts[1] - verts[2]),
                np.linalg.norm(verts[2] - verts[0]),
            ]
            if max(edges) > alpha:
                continue

            for i in range(3):
                edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                boundary_edges[edge] += 1

        # Extract boundary: edges that appear exactly once
        boundary = [e for e, count in boundary_edges.items() if count == 1]

        if not boundary:
            return [self._compute_convex_hull_simple(pts)]

        # Build adjacency graph from boundary edges
        adj = defaultdict(set)
        for a, b in boundary:
            adj[a].add(b)
            adj[b].add(a)

        # Extract ALL connected boundary components (not just one walk)
        visited_nodes = set()
        components = []

        for start_node in adj:
            if start_node in visited_nodes:
                continue

            ordered = [start_node]
            visited_nodes.add(start_node)
            current = start_node

            while True:
                next_node = None
                for n in adj[current]:
                    if n not in visited_nodes:
                        next_node = n
                        break
                if next_node is None:
                    break
                ordered.append(next_node)
                visited_nodes.add(next_node)
                current = next_node

            if len(ordered) >= 3:
                hull = [[float(coords[i][0]), float(coords[i][1])] for i in ordered]

                # Simplify with Douglas-Peucker if too many vertices
                if len(hull) > 80:
                    lats = [p[0] for p in hull]
                    lons = [p[1] for p in hull]
                    diag = ((max(lats) - min(lats)) ** 2 + (max(lons) - min(lons)) ** 2) ** 0.5
                    hull = self._simplify_polygon(hull, epsilon=diag * 0.01)

                components.append(hull)

        if not components:
            return [self._compute_convex_hull_simple(pts)]

        # Sort by vertex count (largest first)
        components.sort(key=lambda c: len(c), reverse=True)

        return components

    @staticmethod
    def _simplify_polygon(points: List[List[float]], epsilon: float) -> List[List[float]]:
        """Ramer-Douglas-Peucker polygon simplification."""
        if len(points) < 3:
            return points

        def _perp_dist(pt, line_start, line_end):
            dx = line_end[0] - line_start[0]
            dy = line_end[1] - line_start[1]
            if dx == 0 and dy == 0:
                return ((pt[0] - line_start[0]) ** 2 + (pt[1] - line_start[1]) ** 2) ** 0.5
            t = ((pt[0] - line_start[0]) * dx + (pt[1] - line_start[1]) * dy) / (dx * dx + dy * dy)
            t = max(0, min(1, t))
            proj_x = line_start[0] + t * dx
            proj_y = line_start[1] + t * dy
            return ((pt[0] - proj_x) ** 2 + (pt[1] - proj_y) ** 2) ** 0.5

        def _rdp(pts, eps):
            if len(pts) <= 2:
                return pts
            d_max = 0
            idx = 0
            for i in range(1, len(pts) - 1):
                d = _perp_dist(pts[i], pts[0], pts[-1])
                if d > d_max:
                    d_max = d
                    idx = i
            if d_max > eps:
                left = _rdp(pts[:idx + 1], eps)
                right = _rdp(pts[idx:], eps)
                return left[:-1] + right
            return [pts[0], pts[-1]]

        # Close the ring for simplification, then remove duplicate end
        closed = points + [points[0]]
        simplified = _rdp(closed, epsilon)
        if len(simplified) > 1 and simplified[0] == simplified[-1]:
            simplified = simplified[:-1]
        return simplified

    @staticmethod
    def _compute_convex_hull_simple(pts: List[Tuple[float, float]]) -> List[List[float]]:
        """Convex hull fallback using Andrew's monotone chain."""
        if len(pts) <= 2:
            return [[p[0], p[1]] for p in pts]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        return [[p[0], p[1]] for p in hull]

    def _compute_and_write_boundaries(self, rosbag_name: str, location_data: Dict) -> None:
        """
        Compute concave hull + bbox from grid cell coordinates and write to boundaries file.

        The boundaries file sits next to the positional lookup table and stores
        a tight polygon per rosbag for fast spatial pre-filtering.

        Args:
            rosbag_name: Rosbag identifier (relative path)
            location_data: Grid cell data {\"lat,lon\": {...}, ...}
        """
        boundaries_path = self.output_dir.parent / "positional_boundaries.json"

        # Extract all grid cell coordinates
        coords: List[Tuple[float, float]] = []
        for lat_lon_key in location_data.keys():
            try:
                lat_str, lon_str = lat_lon_key.split(',')
                coords.append((float(lat_str), float(lon_str)))
            except (ValueError, TypeError):
                continue

        if not coords:
            return

        # Compute bounding box
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        bbox = [min(lats), min(lons), max(lats), max(lons)]

        # Compute concave hull (alpha shape)
        hull = self._compute_concave_hull(coords)

        # Read existing boundaries file
        existing = {}
        if boundaries_path.exists():
            try:
                with open(boundaries_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = {}

        existing[rosbag_name] = {
            "concave_hull": hull,
            "bbox": bbox,
        }

        boundaries_path.parent.mkdir(parents=True, exist_ok=True)
        with open(boundaries_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        # Track in completion.json (output_files stored once at processor level)
        if not getattr(self, '_boundaries_status_set', False):
            self.boundaries_tracker.mark_processor_status("in_progress")
            self._boundaries_status_set = True
        self.boundaries_tracker.mark_completed(
            rosbag_name=rosbag_name,
            status="completed",
            output_files=[boundaries_path],
        )

        total_verts = sum(len(c) for c in hull)
        self.logger.info(f"  Written boundary ({len(hull)} component(s), {total_verts} vertices) to {boundaries_path}")

    def ensure_boundaries(self, rosbag_name: str) -> bool:
        """
        Ensure boundaries exist for a rosbag, regenerating from grid data if needed.

        Called by main.py for already-completed rosbags that may lack boundaries
        (e.g., preprocessed before the boundaries feature existed).

        Args:
            rosbag_name: Rosbag identifier (relative path)

        Returns:
            True if boundaries were generated, False if already complete or no data
        """
        # Check completion.json — single source of truth
        if self.boundaries_tracker.is_rosbag_completed(rosbag_name):
            return False

        # Read grid data from the positional lookup table
        if not self.output_dir.exists():
            return False

        try:
            with open(self.output_dir, 'r', encoding='utf-8') as f:
                lookup_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return False

        location_data = lookup_data.get(rosbag_name)
        if not location_data:
            return False

        # _compute_and_write_boundaries writes the file AND marks completion.json
        self._compute_and_write_boundaries(rosbag_name, location_data)
        return True

    def _round_to_grid(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Round GPS coordinates to grid cells.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
        
        Returns:
            Tuple of (rounded_latitude, rounded_longitude)
        """
        lat_grid = round(lat / self.positional_grid_resolution) * self.positional_grid_resolution
        lon_grid = round(lon / self.positional_grid_resolution) * self.positional_grid_resolution
        return lat_grid, lon_grid