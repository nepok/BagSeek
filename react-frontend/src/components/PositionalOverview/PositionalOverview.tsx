import { Alert, Box, Button, CircularProgress, IconButton, InputAdornment, MenuItem, Paper, Popper, Slider, TextField, Tooltip, Typography } from '@mui/material';
import MapIcon from '@mui/icons-material/Map';
import SatelliteAltIcon from '@mui/icons-material/SatelliteAlt';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import Switch from '@mui/material/Switch';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.heat';
import './PositionalOverview.css';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { extractRosbagName } from '../../utils/rosbag';
import { inflatePathsD, PathsD, PathD, JoinType, EndType } from 'clipper2-ts';
import { formatNsToTime, McapRangeMeta } from '../McapRangeFilter/McapRangeFilter';

type RosbagPoint = {
  lat: number;
  lon: number;
  count: number;
};

type RosbagMeta = {
  name: string;
  timestamp: number | null;
};

type PolygonPoint = {
  lat: number;
  lon: number;
  id: string;
};

type Polygon = {
  id: string;
  points: PolygonPoint[];
  isClosed: boolean;
};

type McapLocationPoint = {
  lat: number;
  lon: number;
  mcaps: { [mcap_id: string]: number }; // mcap_id -> count
};

type McapInfo = {
  id: string;
  totalCount: number;
};

type RosbagBoundary = {
  concave_hull: [number, number][][]; // Array of polygon components, each [[lat, lon], ...]
  bbox: [number, number, number, number]; // [min_lat, min_lon, max_lat, max_lon]
};

const TILE_LAYER_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const TILE_LAYER_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

// Global heatmap configuration
const HEATMAP_BASE_RADIUS = 30;
const HEATMAP_BASE_BLUR = 30;
const HEATMAP_MAX_ZOOM = 18;
const HEATMAP_MIN_ZOOM_FACTOR = 0.6;

// Calculate zoom factor for heatmap scaling
const calculateHeatmapZoomFactor = (currentZoom: number): number => {
  return Math.max(HEATMAP_MIN_ZOOM_FACTOR, currentZoom / HEATMAP_MAX_ZOOM);
};

// HTML escape helper to prevent XSS in Leaflet popups and labels
const escapeHtml = (str: string | number): string => {
  const s = String(str);
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
};

// Helper to find closest point on a line segment
const closestPointOnSegment = (
  px: number, py: number,  // Point to project
  ax: number, ay: number,  // Segment start
  bx: number, by: number   // Segment end
): { x: number; y: number; t: number } => {
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;

  if (lenSq === 0) {
    // Segment is a point
    return { x: ax, y: ay, t: 0 };
  }

  // Project point onto line, clamped to segment
  let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));

  return {
    x: ax + t * dx,
    y: ay + t * dy,
    t,
  };
};

const ROSBAG_TIMESTAMP_REGEX = /^rosbag2_(\d{4})_(\d{2})_(\d{2})-(\d{2})_(\d{2})_(\d{2})(?:_short)?$/;

const parseRosbagTimestamp = (name: string): number | null => {
  // Handle multipart rosbags: extract base name before _multi_parts or /Part_
  let baseName = name.trim();
  
  // If it contains _multi_parts, extract the part before it
  if (baseName.includes('_multi_parts')) {
    baseName = baseName.split('_multi_parts')[0];
  }
  // If it contains /Part_, extract the part before it (for paths like "rosbag2_.../Part_2")
  else if (baseName.includes('/Part_')) {
    baseName = baseName.split('/Part_')[0];
  }
  // If it's just a path with slashes, use the last part (e.g., "rosbag2_.../Part_2" -> "Part_2")
  // But we want the base name, so check if the last part starts with "Part_"
  else if (baseName.includes('/')) {
    const parts = baseName.split('/');
    const lastPart = parts[parts.length - 1];
    // If last part is "Part_X", go up one level
    if (lastPart.startsWith('Part_')) {
      baseName = parts[parts.length - 2] || baseName;
    } else {
      // Otherwise use the last part (leaf name)
      baseName = lastPart;
    }
  }
  
  const match = ROSBAG_TIMESTAMP_REGEX.exec(baseName);
  if (!match) {
    return null;
  }

  const [, year, month, day, hour, minute, second] = match;
  const date = new Date(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second),
  );

  return Number.isNaN(date.getTime()) ? null : date.getTime();
};

const formatTimestampLabel = (timestamp: number | null, fallback: string) => {
  if (!timestamp) {
    return fallback;
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(timestamp));
};

// Point-in-polygon algorithm (ray casting algorithm)
const pointInPolygon = (point: { lat: number; lon: number }, polygon: PolygonPoint[]): boolean => {
  if (polygon.length < 3) {
    return false;
  }

  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].lon;
    const yi = polygon[i].lat;
    const xj = polygon[j].lon;
    const yj = polygon[j].lat;

    const intersect =
      (yi > point.lat) !== (yj > point.lat) &&
      point.lon < ((xj - xi) * (point.lat - yi)) / (yj - yi) + xi;
    if (intersect) {
      inside = !inside;
    }
  }

  return inside;
};

// Check if two axis-aligned bounding boxes overlap
const bboxOverlaps = (
  a: [number, number, number, number], // [min_lat, min_lon, max_lat, max_lon]
  b: [number, number, number, number],
): boolean => {
  return a[0] <= b[2] && a[2] >= b[0] && a[1] <= b[3] && a[3] >= b[1];
};

// Check if two line segments intersect
const segmentsIntersect = (
  a1x: number, a1y: number, a2x: number, a2y: number,
  b1x: number, b1y: number, b2x: number, b2y: number,
): boolean => {
  const d1x = a2x - a1x, d1y = a2y - a1y;
  const d2x = b2x - b1x, d2y = b2y - b1y;
  const cross = d1x * d2y - d1y * d2x;
  if (Math.abs(cross) < 1e-12) return false;
  const t = ((b1x - a1x) * d2y - (b1y - a1y) * d2x) / cross;
  const u = ((b1x - a1x) * d1y - (b1y - a1y) * d1x) / cross;
  return t >= 0 && t <= 1 && u >= 0 && u <= 1;
};

// Check if a rosbag boundary (multi-polygon concave hull) overlaps with any user polygon
const boundaryOverlapsPolygons = (
  boundary: RosbagBoundary,
  closedPolygons: Polygon[],
  offsetDistance: number = 0,
): boolean => {
  if (closedPolygons.length === 0) return true;
  if (!boundary.concave_hull || boundary.concave_hull.length === 0) return false;

  const polygonsToCheck = getPolygonsForFiltering(closedPolygons, offsetDistance);

  for (const userPoly of polygonsToCheck) {
    // Quick reject: bbox check
    const polyLats = userPoly.map((p) => p.lat);
    const polyLons = userPoly.map((p) => p.lon);
    const polyBbox: [number, number, number, number] = [
      Math.min(...polyLats), Math.min(...polyLons),
      Math.max(...polyLats), Math.max(...polyLons),
    ];
    if (!bboxOverlaps(boundary.bbox, polyBbox)) continue;

    // Check each hull component — overlap with ANY component means overlap
    for (const component of boundary.concave_hull) {
      if (component.length < 3) continue;

      const componentAsPolygonPoints: PolygonPoint[] = component.map(([lat, lon]) => ({
        lat, lon, id: '',
      }));

      // Check if any hull vertex is inside user polygon
      for (const [lat, lon] of component) {
        if (pointInPolygon({ lat, lon }, userPoly)) return true;
      }

      // Check if any user polygon vertex is inside this hull component
      for (const pt of userPoly) {
        if (pointInPolygon({ lat: pt.lat, lon: pt.lon }, componentAsPolygonPoints)) return true;
      }

      // Check edge intersections
      for (let i = 0; i < component.length; i++) {
        const h1 = component[i];
        const h2 = component[(i + 1) % component.length];
        for (let j = 0; j < userPoly.length; j++) {
          const p1 = userPoly[j];
          const p2 = userPoly[(j + 1) % userPoly.length];
          if (segmentsIntersect(h1[1], h1[0], h2[1], h2[0], p1.lon, p1.lat, p2.lon, p2.lat)) {
            return true;
          }
        }
      }
    }
  }

  return false;
};

// Format MCAP IDs with consecutive ranges (e.g., [0,1,2,3,4,7,9,10,11] -> "0 - 4, 7, 9 - 11")
const formatMcapRanges = (mcapIds: string[]): string => {
  if (mcapIds.length === 0) {
    return '';
  }

  // Sort MCAP IDs numerically if possible, otherwise alphabetically
  const sorted = [...mcapIds].sort((a, b) => {
    const aNum = parseInt(a, 10);
    const bNum = parseInt(b, 10);
    if (!isNaN(aNum) && !isNaN(bNum)) {
      return aNum - bNum;
    }
    return a.localeCompare(b);
  });

  const ranges: string[] = [];
  let rangeStart: string | null = null;
  let rangeEnd: string | null = null;

  for (let i = 0; i < sorted.length; i++) {
    const current = sorted[i];
    const currentNum = parseInt(current, 10);
    const prevNum = rangeEnd !== null ? parseInt(rangeEnd, 10) : null;

    // Check if current is consecutive to previous
    if (rangeStart !== null && prevNum !== null && !isNaN(currentNum) && !isNaN(prevNum) && currentNum === prevNum + 1) {
      // Continue the range
      rangeEnd = current;
    } else {
      // Finish previous range if exists
      if (rangeStart !== null) {
        if (rangeEnd !== null && rangeStart !== rangeEnd) {
          ranges.push(`${rangeStart} - ${rangeEnd}`);
        } else {
          ranges.push(rangeStart);
        }
      }
      // Start new range
      rangeStart = current;
      rangeEnd = current;
    }
  }

  // Finish last range
  if (rangeStart !== null) {
    if (rangeEnd !== null && rangeStart !== rangeEnd) {
      ranges.push(`${rangeStart} - ${rangeEnd}`);
    } else {
      ranges.push(rangeStart);
    }
  }

  return ranges.join(', ');
};

// Convert GeoJSON to Polygon format
const convertGeoJSONToPolygons = (geoJson: any, baseId: string): Polygon[] => {
  const polygons: Polygon[] = [];
  
  if (!geoJson || geoJson.type !== 'FeatureCollection' || !Array.isArray(geoJson.features)) {
    return polygons;
  }
  
  geoJson.features.forEach((feature: any, featureIndex: number) => {
    if (!feature.geometry || feature.geometry.type !== 'Polygon') {
      return;
    }
    
    // GeoJSON Polygon coordinates: array of rings, first ring is exterior
    const coordinates = feature.geometry.coordinates;
    if (!Array.isArray(coordinates) || coordinates.length === 0) {
      return;
    }
    
    // Use the first ring (exterior ring) for the polygon
    const exteriorRing = coordinates[0];
    if (!Array.isArray(exteriorRing) || exteriorRing.length < 3) {
      return;
    }
    
    // Convert [lon, lat] to {lat, lon, id}
    let points: PolygonPoint[] = exteriorRing.map((coord: number[], pointIndex: number) => {
      const [lon, lat] = coord;
      return {
        lat,
        lon,
        id: `${baseId}_${featureIndex}_${pointIndex}`,
      };
    });
    
    // Remove duplicate first/last point if present (GeoJSON polygons are closed)
    if (points.length > 1) {
      const first = points[0];
      const last = points[points.length - 1];
      if (first.lat === last.lat && first.lon === last.lon && points.length > 3) {
        points = points.slice(0, -1);
      }
      
      // Remove any other duplicate consecutive points
      const uniquePoints: PolygonPoint[] = [points[0]];
      for (let i = 1; i < points.length; i++) {
        const current = points[i];
        const prev = uniquePoints[uniquePoints.length - 1];
        const dist = Math.sqrt(
          Math.pow(current.lat - prev.lat, 2) +
          Math.pow(current.lon - prev.lon, 2)
        );
        if (dist > 1e-9) {
          uniquePoints.push(current);
        }
      }
      points = uniquePoints;
    }
    
    // Generate unique polygon ID
    const polygonId = feature.properties?.name || feature.properties?.id || `${baseId}_${featureIndex}`;
    
    polygons.push({
      id: polygonId,
      points,
      isClosed: true, // Imported polygons are always closed
    });
  });
  
  return polygons;
};

// Convert internal Polygon format to GeoJSON FeatureCollection
const convertPolygonsToGeoJSON = (polygons: Polygon[]): any => {
  const features = polygons
    .filter((polygon) => polygon.isClosed && polygon.points.length >= 3)
    .map((polygon) => {
      // GeoJSON polygons need closed rings (first point == last point)
      const coordinates = polygon.points.map((p) => [p.lon, p.lat]);
      // Close the ring
      if (coordinates.length > 0) {
        coordinates.push([polygon.points[0].lon, polygon.points[0].lat]);
      }

      return {
        type: 'Feature',
        properties: {
          name: polygon.id,
        },
        geometry: {
          type: 'Polygon',
          coordinates: [coordinates],
        },
      };
    });

  return {
    type: 'FeatureCollection',
    features,
  };
};

// Offset polygon by distance in meters using Clipper2-ts
// Positive offset = shrink inward, negative offset = expand outward
// Based on: https://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/_Body.htm
const offsetPolygon = (polygon: PolygonPoint[], offsetMeters: number): PolygonPoint[] => {
  if (polygon.length < 3 || offsetMeters === 0) {
    return polygon;
  }

  // Remove duplicate first/last point if polygon is already closed
  let workingPoints = [...polygon];
  if (workingPoints.length > 0) {
    const first = workingPoints[0];
    const last = workingPoints[workingPoints.length - 1];
    // Check if first and last points are the same (closed polygon)
    if (first.lat === last.lat && first.lon === last.lon && workingPoints.length > 3) {
      workingPoints = workingPoints.slice(0, -1); // Remove duplicate last point
    }
  }
  
  if (workingPoints.length < 3) {
    return polygon; // Return original if not enough points after deduplication
  }

  // Calculate average latitude for coordinate conversion
  const avgLat = workingPoints.reduce((sum, p) => sum + p.lat, 0) / workingPoints.length;
  
  // Convert lat/lon to local meter-based coordinate system for better precision
  // Use the centroid as the origin
  const centroidLat = avgLat;
  const centroidLon = workingPoints.reduce((sum, p) => sum + p.lon, 0) / workingPoints.length;
  
  // Conversion factors
  const metersPerDegreeLat = 111000;
  const metersPerDegreeLon = 111000 * Math.cos((avgLat * Math.PI) / 180);
  
  // Scale factor to convert meters to a reasonable coordinate system
  // Use 1:1 mapping (1 unit = 1 meter) for best precision
  const scale = 1.0;
  
  try {
    // Convert polygon points to local meter-based coordinates
    // PathD: array of {x, y} where x and y are in meters from centroid
    const pathD: PathD = workingPoints.map((p) => ({
      x: (p.lon - centroidLon) * metersPerDegreeLon * scale,
      y: (p.lat - centroidLat) * metersPerDegreeLat * scale,
    }));
    
    // Create PathsD containing our path
    const pathsD: PathsD = [pathD];
    
    // Clipper2's inflatePathsD: positive delta = expand, negative = shrink
    // Our convention: positive = shrink inward, so we negate
    // Delta is in the same units as the coordinates (meters * scale)
    const delta = -offsetMeters * scale;
    
    // Apply offset using Clipper2's inflatePathsD
    // According to docs: positive delta expands outer contours, negative contracts them
    const offsetResult = inflatePathsD(
      pathsD,
      delta,
      JoinType.Miter,
      EndType.Polygon,
      10.0,  // miterLimit
      undefined, // precision (use default)
      0.1    // arcTolerance (in meters, relative to coordinate system)
    );
    
    // Check if we got a result
    if (!offsetResult || offsetResult.length === 0 || offsetResult[0].length === 0) {
      // Offset failed (e.g., too large offset), return original
      return polygon;
    }
    
    // Get the first path from result
    const resultPath = offsetResult[0];
    
    // Convert back from local meter-based coordinates to lat/lon
    const offsetPoints: PolygonPoint[] = resultPath.map((pt, index) => ({
      lat: centroidLat + (pt.y / scale) / metersPerDegreeLat,
      lon: centroidLon + (pt.x / scale) / metersPerDegreeLon,
      id: workingPoints[index]?.id || `offset_${index}`,
    }));
    
    // Remove duplicate consecutive points
    if (offsetPoints.length > 1) {
      const cleanedPoints: PolygonPoint[] = [offsetPoints[0]];
      
      for (let i = 1; i < offsetPoints.length; i++) {
        const current = offsetPoints[i];
        const prev = cleanedPoints[cleanedPoints.length - 1];
        const dist = Math.sqrt(
          Math.pow(current.lat - prev.lat, 2) +
          Math.pow(current.lon - prev.lon, 2)
        );
        
        // Only add if point is different from previous
        if (dist > 1e-9) {
          cleanedPoints.push(current);
        }
      }
      
      // Check if last point is same as first (closed polygon)
      if (cleanedPoints.length > 1) {
        const firstPoint = cleanedPoints[0];
        const lastPoint = cleanedPoints[cleanedPoints.length - 1];
        const dist = Math.sqrt(
          Math.pow(lastPoint.lat - firstPoint.lat, 2) +
          Math.pow(lastPoint.lon - firstPoint.lon, 2)
        );
        if (dist < 1e-6) {
          // Remove duplicate last point
          cleanedPoints.pop();
        }
      }
      
      return cleanedPoints.length >= 3 ? cleanedPoints : polygon;
    }
    
    return offsetPoints.length >= 3 ? offsetPoints : polygon;
  } catch (error) {
    // If offset fails, return original polygon
    console.warn('Polygon offset failed:', error);
    return polygon;
  }
};

// Get polygons to use for filtering (offset if offsetDistance !== 0, otherwise original)
const getPolygonsForFiltering = (polygons: Polygon[], offsetDistance: number): PolygonPoint[][] => {
  const closedPolygons = polygons.filter((p) => p.isClosed);
  
  if (offsetDistance === 0) {
    return closedPolygons.map((p) => p.points);
  }
  
  return closedPolygons.map((polygon) => {
    try {
      return offsetPolygon(polygon.points, offsetDistance);
    } catch {
      return polygon.points; // Fallback to original if offset fails
    }
  });
};

// Check if any point from rosbag overlaps with any closed polygon
const checkRosbagOverlap = async (
  rosbagName: string,
  closedPolygons: Polygon[],
  offsetDistance: number = 0
): Promise<boolean> => {
  if (closedPolygons.length === 0) {
    return true; // No polygons means all overlap
  }

  try {
    const response = await fetch(`/api/positions/rosbags/${encodeURIComponent(rosbagName)}`);
    if (!response.ok) {
      return false; // If we can't fetch data, assume no overlap
    }
    const data = await response.json();
    const rosbagPoints: RosbagPoint[] = Array.isArray(data.points) ? data.points : [];

    // Get polygons to use (offset or original)
    const polygonsToCheck = getPolygonsForFiltering(closedPolygons, offsetDistance);

    // Check if any point from rosbag is inside any polygon
    for (const point of rosbagPoints) {
      for (const polygonPoints of polygonsToCheck) {
        if (pointInPolygon({ lat: point.lat, lon: point.lon }, polygonPoints)) {
          return true; // Found overlap
        }
      }
    }

    return false; // No overlap found
  } catch {
    return false;
  }
};

// Get MCAP IDs that have points inside closed polygons
const getMcapsInsidePolygons = async (
  rosbagName: string,
  closedPolygons: Polygon[],
  offsetDistance: number = 0
): Promise<Set<string>> => {
  const mcapIds = new Set<string>();

  if (closedPolygons.length === 0) {
    return mcapIds; // No polygons means no MCAPs inside
  }

  try {
    const response = await fetch(`/api/positions/rosbags/${encodeURIComponent(rosbagName)}/mcaps`);
    if (!response.ok) {
      return mcapIds; // If we can't fetch data, return empty set
    }
    const data = await response.json();
    const mcapPoints: McapLocationPoint[] = Array.isArray(data.points) ? data.points : [];

    // Get polygons to use (offset or original)
    const polygonsToCheck = getPolygonsForFiltering(closedPolygons, offsetDistance);

    // Check each MCAP point
    for (const point of mcapPoints) {
      // Check if point is inside any polygon
      let isInside = false;
      for (const polygonPoints of polygonsToCheck) {
        if (pointInPolygon({ lat: point.lat, lon: point.lon }, polygonPoints)) {
          isInside = true;
          break;
        }
      }

      // If point is inside, add all MCAP IDs from this point
      if (isInside && point.mcaps) {
        for (const mcapId of Object.keys(point.mcaps)) {
          mcapIds.add(mcapId);
        }
      }
    }

    return mcapIds;
  } catch {
    return mcapIds;
  }
};

const PositionalOverview: React.FC = () => {
  const navigate = useNavigate();
  const [rosbags, setRosbags] = useState<RosbagMeta[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [loadingRosbags, setLoadingRosbags] = useState<boolean>(true);
  const [loadingPoints, setLoadingPoints] = useState<boolean>(false);
  const [points, setPoints] = useState<RosbagPoint[]>([]);
  const [allPoints, setAllPoints] = useState<RosbagPoint[]>([]);
  const [loadingAllPoints, setLoadingAllPoints] = useState<boolean>(false);
  const [allPointsLoaded, setAllPointsLoaded] = useState<boolean>(false);
  const [showAllRosbags, setShowAllRosbags] = useState<boolean>(false);
  const [showMcaps, setShowMcaps] = useState<boolean>(false);
  const [mcapLocationPoints, setMcapLocationPoints] = useState<McapLocationPoint[]>([]);
  const [availableMcaps, setAvailableMcaps] = useState<McapInfo[]>([]);
  const [mcapRanges, setMcapRanges] = useState<McapRangeMeta[]>([]);
  const [selectedMcapIndex, setSelectedMcapIndex] = useState<number>(0);
  const [expandedLocation, setExpandedLocation] = useState<{ lat: number; lon: number } | null>(null);
  const [loadingMcaps, setLoadingMcaps] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [useSatellite, setUseSatellite] = useState<boolean>(false);
  const [polygons, setPolygons] = useState<Polygon[]>([]);
  const [activePolygonId, setActivePolygonId] = useState<string | null>(null);
  const [selectedPolygonId, setSelectedPolygonId] = useState<string | null>(null);
  const [selectedPointId, setSelectedPointId] = useState<{ polygonId: string; pointId: string } | null>(null);
  const [edgeHoverInfo, setEdgeHoverInfo] = useState<{
    polygonId: string;
    segmentIndex: number; // Insert after this index
    lat: number;
    lon: number;
  } | null>(null);
  const [rosbagOverlapStatus, setRosbagOverlapStatus] = useState<Map<string, boolean>>(new Map());
  const [mcapOverlapIds, setMcapOverlapIds] = useState<Set<string> | null>(null);
  const [checkingOverlap, setCheckingOverlap] = useState<boolean>(false);
  const [polygonFiles, setPolygonFiles] = useState<string[]>([]);
  const [selectedPolygonFile, setSelectedPolygonFile] = useState<string>('');
  const [loadingPolygonFiles, setLoadingPolygonFiles] = useState<boolean>(false);
  const [importingPolygon, setImportingPolygon] = useState<boolean>(false);
  const [isRestoringPolygons, setIsRestoringPolygons] = useState<boolean>(false);
  const [exportingList, setExportingList] = useState<boolean>(false);
  const [applyingToSearch, setApplyingToSearch] = useState<boolean>(false);
  const [rosbagBoundaries, setRosbagBoundaries] = useState<Record<string, RosbagBoundary> | null>(null);
  const [offsetDistance, setOffsetDistance] = useState<number>(0);
  const [showPolygonPopper, setShowPolygonPopper] = useState<boolean>(false);
  const [showPolygonSaveField, setShowPolygonSaveField] = useState<boolean>(false);
  const [newPolygonName, setNewPolygonName] = useState<string>('');
  const [savingPolygon, setSavingPolygon] = useState<boolean>(false);
  const [deletingPolygon, setDeletingPolygon] = useState<string | null>(null);
  const polygonButtonRef = useRef<HTMLButtonElement | null>(null);
  const prevPolygonCountRef = useRef<number>(0);

  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const baseLayersRef = useRef<{ map: L.TileLayer | null; satellite: L.TileLayer | null }>({ map: null, satellite: null });
  const heatLayerRef = useRef<L.Layer | null>(null);
  const allHeatLayerRef = useRef<L.Layer | null>(null);
  const mcapBorderLayersRef = useRef<Map<string, { borders: L.Polygon[]; labels: L.Marker[]; markers: L.Marker[] }>>(new Map());
  const mcapHeatLayerRef = useRef<L.Layer | null>(null);
  const hoverTooltipRef = useRef<L.Popup | null>(null);
  const polygonLayersRef = useRef<Map<string, { markers: L.Marker[]; polyline: L.Polyline | null; polygon: L.Polygon | null }>>(new Map());
  const offsetPolygonLayersRef = useRef<Map<string, L.Polyline>>(new Map());
  const edgeGhostMarkerRef = useRef<L.Marker | null>(null);
  const isRestoringRef = useRef(false);
  const prevPointsRef = useRef<RosbagPoint[]>([]);

  const handleBack = () => {
    if (window.history.length > 1) {
      navigate(-1);
    } else {
      navigate('/search');
    }
  };

  useEffect(() => {
    let cancelled = false;
    const fetchRosbags = async () => {
      setLoadingRosbags(true);
      setError(null);
      try {
        const response = await fetch('/api/positions/rosbags');
        if (!response.ok) {
          throw new Error(`Failed to load rosbag list (${response.status})`);
        }
        const data = await response.json();
        if (!cancelled) {
          const names: string[] = Array.isArray(data.rosbags) ? data.rosbags : [];
          const meta = names
            .map((name) => ({
              name,
              timestamp: parseRosbagTimestamp(name),
            }))
            .sort((a, b) => {
              if (a.timestamp && b.timestamp) {
                return a.timestamp - b.timestamp;
              }
              if (a.timestamp) return -1;
              if (b.timestamp) return 1;
              return a.name.localeCompare(b.name);
            });

          setRosbags(meta);
          // Pre-select: 1) positional filter match, 2) last MAP selection (sessionStorage), 3) last rosbag
          let idx = meta.length ? meta.length - 1 : 0;
          try {
            if (meta.length > 0) {
              // Try positional filter first (from Apply to Search)
              const filterRaw = sessionStorage.getItem('__BagSeekPositionalFilter');
              if (filterRaw) {
                const filterNames: string[] = JSON.parse(filterRaw);
                if (Array.isArray(filterNames) && filterNames.length > 0) {
                  const filterNorm = new Set(filterNames.map((n) => extractRosbagName(String(n).trim())));
                  const match = meta.findIndex((m) => filterNorm.has(extractRosbagName(m.name)));
                  if (match >= 0) idx = match;
                }
              }
              // Fallback: last selected rosbag in MAP (for when opening MAP directly)
              if (idx === meta.length - 1) {
                const lastMap = sessionStorage.getItem('__BagSeekMapSelectedRosbag');
                if (lastMap) {
                  const lastNorm = extractRosbagName(lastMap.trim());
                  const match = meta.findIndex((m) => extractRosbagName(m.name) === lastNorm);
                  if (match >= 0) idx = match;
                }
              }
            }
          } catch {}
          setSelectedIndex(idx);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : 'Failed to load rosbag list');
          setRosbags([]);
          setSelectedIndex(0);
        }
      } finally {
        if (!cancelled) {
          setLoadingRosbags(false);
        }
      }
    };

    fetchRosbags();

    return () => {
      cancelled = true;
    };
  }, []);

  // Fetch rosbag boundaries (concave hulls) for fast overlap pre-filtering
  useEffect(() => {
    let cancelled = false;
    fetch('/api/positions/boundaries')
      .then((res) => (res.ok ? res.json() : { boundaries: {} }))
      .then((data) => {
        if (!cancelled) {
          setRosbagBoundaries(data.boundaries ?? {});
        }
      })
      .catch(() => { if (!cancelled) setRosbagBoundaries({}); });
    return () => { cancelled = true; };
  }, []);


  // Restore polygons from sessionStorage on mount
  useEffect(() => {
    try {
      const cachedPolygons = sessionStorage.getItem('__BagSeekPositionalPolygons');
      if (cachedPolygons) {
        const parsed = JSON.parse(cachedPolygons);
        if (Array.isArray(parsed) && parsed.length > 0) {
          isRestoringRef.current = true;
          setIsRestoringPolygons(true);
          setPolygons(parsed);
          prevPolygonCountRef.current = parsed.length;
          // If there's an active (non-closed) polygon, set it as active
          const activePolygon = parsed.find((p: Polygon) => !p.isClosed);
          if (activePolygon) {
            setActivePolygonId(activePolygon.id);
          }
          // Reset flag after a short delay to allow state update
          setTimeout(() => {
            isRestoringRef.current = false;
            setIsRestoringPolygons(false);
          }, 0);
        }
      }
    } catch (e) {
      console.error('Failed to restore polygons from cache:', e);
    }
  }, []);

  // Fetch available polygon files
  useEffect(() => {
    let cancelled = false;
    const fetchPolygonFiles = async () => {
      setLoadingPolygonFiles(true);
      try {
        const response = await fetch('/api/polygons/list');
        if (!response.ok) {
          throw new Error(`Failed to load polygon files (${response.status})`);
        }
        const data = await response.json();
        if (!cancelled && Array.isArray(data.files)) {
          setPolygonFiles(data.files);
        }
      } catch (fetchError) {
        console.error('Failed to load polygon files:', fetchError);
        if (!cancelled) {
          setPolygonFiles([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingPolygonFiles(false);
        }
      }
    };

    fetchPolygonFiles();

    return () => {
      cancelled = true;
    };
  }, []);

  // Handle polygon import
  const handleImportPolygon = useCallback(async (filename: string) => {
    if (!filename || importingPolygon) {
      return;
    }

    setImportingPolygon(true);
    setError(null);

    try {
      const response = await fetch(`/api/polygons/${encodeURIComponent(filename)}`);
      if (!response.ok) {
        throw new Error(`Failed to load polygon file (${response.status})`);
      }

      const geoJson = await response.json();
      const baseId = filename.replace('.json', '');
      const importedPolygons = convertGeoJSONToPolygons(geoJson, baseId);

      if (importedPolygons.length === 0) {
        throw new Error('No valid polygons found in file');
      }

      // Replace existing polygons with imported ones
      setPolygons(importedPolygons);
      setActivePolygonId(null);
      setSelectedPolygonId(null);
      
      // Update sessionStorage
      try {
        sessionStorage.setItem('__BagSeekPositionalPolygons', JSON.stringify(importedPolygons));
      } catch (e) {
        console.error('Failed to save polygons to cache:', e);
      }

      // Keep the selected file name in the dropdown
      setSelectedPolygonFile(filename);
    } catch (fetchError) {
      const errorMessage = fetchError instanceof Error ? fetchError.message : 'Failed to import polygon';
      setError(errorMessage);
      console.error('Failed to import polygon:', fetchError);
    } finally {
      setImportingPolygon(false);
    }
  }, [importingPolygon]);

  // Handle save polygon to file
  const handleSavePolygon = useCallback(async (name: string) => {
    if (!name.trim() || savingPolygon) {
      return;
    }

    const closedPolygons = polygons.filter((p) => p.isClosed);
    if (closedPolygons.length === 0) {
      setError('No closed polygons to save');
      return;
    }

    setSavingPolygon(true);
    setError(null);

    try {
      const geojson = convertPolygonsToGeoJSON(closedPolygons);
      const response = await fetch('/api/polygons', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), geojson }),
      });

      if (!response.ok) {
        let errorMessage = `Failed to save polygon (${response.status})`;
        try {
          const errorData = await response.json();
          if (errorData.error) {
            errorMessage = errorData.error;
          }
        } catch {
          // Response wasn't JSON, use default error message
        }
        throw new Error(errorMessage);
      }

      // Refresh polygon files list
      const listResponse = await fetch('/api/polygons/list');
      if (listResponse.ok) {
        const listData = await listResponse.json();
        setPolygonFiles(listData.files || []);
      }

      // Reset UI state
      setNewPolygonName('');
      setShowPolygonSaveField(false);
    } catch (saveError) {
      const errorMessage = saveError instanceof Error ? saveError.message : 'Failed to save polygon';
      setError(errorMessage);
      console.error('Failed to save polygon:', saveError);
    } finally {
      setSavingPolygon(false);
    }
  }, [polygons, savingPolygon]);

  // Handle delete polygon file
  const handleDeletePolygonFile = useCallback(async (filename: string) => {
    if (deletingPolygon) {
      return;
    }

    setDeletingPolygon(filename);
    setError(null);

    try {
      const response = await fetch(`/api/polygons/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        let errorMessage = `Failed to delete polygon (${response.status})`;
        try {
          const errorData = await response.json();
          if (errorData.error) {
            errorMessage = errorData.error;
          }
        } catch {
          // Response wasn't JSON, use default error message
        }
        throw new Error(errorMessage);
      }

      // Refresh polygon files list
      const listResponse = await fetch('/api/polygons/list');
      if (listResponse.ok) {
        const listData = await listResponse.json();
        setPolygonFiles(listData.files || []);
      }

      // If the deleted file was selected, clear the selection
      if (selectedPolygonFile === filename) {
        setSelectedPolygonFile('');
      }
    } catch (deleteError) {
      const errorMessage = deleteError instanceof Error ? deleteError.message : 'Failed to delete polygon';
      setError(errorMessage);
      console.error('Failed to delete polygon:', deleteError);
    } finally {
      setDeletingPolygon(null);
    }
  }, [deletingPolygon, selectedPolygonFile]);

  // Handle export list
  const handleExportList = useCallback(async () => {
    if (exportingList || rosbags.length === 0) {
      return;
    }

    const closedPolygons = polygons.filter((p) => p.isClosed);
    if (closedPolygons.length === 0) {
      setError('No closed polygons to export');
      return;
    }

    setExportingList(true);
    setError(null);

    try {
      const lines: string[] = [];

      // Process each rosbag
      for (const rosbag of rosbags) {
        const mcapIds = await getMcapsInsidePolygons(rosbag.name, closedPolygons, offsetDistance);
        
        if (mcapIds.size > 0) {
          const formatted = formatMcapRanges(Array.from(mcapIds));
          lines.push(`${rosbag.name}: ${formatted}`);
        }
      }

      if (lines.length === 0) {
        setError('No MCAPs found inside polygons');
        setExportingList(false);
        return;
      }

      // Create and download file
      const content = lines.join('\n\n');
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'list.txt';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (exportError) {
      const errorMessage = exportError instanceof Error ? exportError.message : 'Failed to export list';
      setError(errorMessage);
      console.error('Failed to export list:', exportError);
    } finally {
      setExportingList(false);
    }
  }, [exportingList, rosbags, polygons, offsetDistance]);

  useEffect(() => {
    if (!rosbags.length) {
      setPoints([]);
      return;
    }

    let cancelled = false;
    const fetchPoints = async () => {
      setLoadingPoints(true);
      setError(null);
      try {
        const selected = rosbags[selectedIndex];
        if (!selected) {
          setPoints([]);
          return;
        }

        const response = await fetch(`/api/positions/rosbags/${encodeURIComponent(selected.name)}`);
        if (!response.ok) {
          if (response.status === 404) {
            throw new Error(`No positional data found for ${selected.name}`);
          }
          throw new Error(`Failed to load positional data (${response.status})`);
        }
        const data = await response.json();
        if (!cancelled) {
          const newPoints: RosbagPoint[] = Array.isArray(data.points) ? data.points : [];
          setPoints(newPoints);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : 'Failed to load positional data');
          setPoints([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingPoints(false);
        }
      }
    };

    fetchPoints();

    return () => {
      cancelled = true;
    };
  }, [rosbags, selectedIndex]);

  // Persist selected rosbag so Export (and future visits) can use it
  useEffect(() => {
    const selected = rosbags[selectedIndex];
    if (selected?.name) {
      try {
        sessionStorage.setItem('__BagSeekMapSelectedRosbag', selected.name);
      } catch {}
    }
  }, [rosbags, selectedIndex]);

  useEffect(() => {
    if (!showMcaps || !rosbags.length) {
      setMcapLocationPoints([]);
      setAvailableMcaps([]);
      setMcapRanges([]);
      setSelectedMcapIndex(0);
      return;
    }

    let cancelled = false;
    const fetchMcapData = async () => {
      setLoadingMcaps(true);
      setError(null);
      try {
        const selected = rosbags[selectedIndex];
        if (!selected) {
          setMcapLocationPoints([]);
          setAvailableMcaps([]);
          setMcapRanges([]);
          return;
        }

        // Fetch mcap location points
        const pointsResponse = await fetch(`/api/positions/rosbags/${encodeURIComponent(selected.name)}/mcaps`);
        if (!pointsResponse.ok) {
          if (pointsResponse.status === 404) {
            throw new Error(`No mcap positional data found for ${selected.name}`);
          }
          throw new Error(`Failed to load mcap positional data (${pointsResponse.status})`);
        }
        const pointsData = await pointsResponse.json();
        
        // Fetch mcap list
        const listResponse = await fetch(`/api/positions/rosbags/${encodeURIComponent(selected.name)}/mcap-list`);
        if (!listResponse.ok) {
          if (listResponse.status === 404) {
            throw new Error(`No mcap list found for ${selected.name}`);
          }
          throw new Error(`Failed to load mcap list (${listResponse.status})`);
        }
        const listData = await listResponse.json();

        // Fetch timestamp summary for MCAP time-of-day marks
        let ranges: McapRangeMeta[] = [];
        try {
          const tsResponse = await fetch(`/api/get-timestamp-summary?rosbag=${encodeURIComponent(selected.name)}`);
          if (tsResponse.ok) {
            const tsData = await tsResponse.json();
            ranges = (tsData.mcapRanges || []).map((r: any) => ({
              mcapIdentifier: r.mcapIdentifier,
              count: r.count,
              startIndex: r.startIndex,
              firstTimestampNs: r.firstTimestampNs,
              lastTimestampNs: r.lastTimestampNs,
            }));
          }
        } catch { /* non-critical */ }

        if (!cancelled) {
          const newPoints: McapLocationPoint[] = Array.isArray(pointsData.points) ? pointsData.points : [];
          const newMcaps: McapInfo[] = Array.isArray(listData.mcaps) ? listData.mcaps : [];
          setMcapLocationPoints(newPoints);
          setAvailableMcaps(newMcaps);
          setMcapRanges(ranges);
          setSelectedMcapIndex(0);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : 'Failed to load mcap data');
          setMcapLocationPoints([]);
          setAvailableMcaps([]);
          setMcapRanges([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingMcaps(false);
        }
      }
    };

    fetchMcapData();

    return () => {
      cancelled = true;
    };
  }, [showMcaps, rosbags, selectedIndex]);

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) {
      return;
    }

    const initialView: L.LatLngExpression = [51.0, 12.0];
    const map = L.map(mapContainerRef.current, {
      preferCanvas: true,
      zoomControl: true,
      center: initialView,
      zoom: 13,
    });

    const mapLayer = L.tileLayer(TILE_LAYER_URL, {
      attribution: TILE_LAYER_ATTRIBUTION,
    });

    const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles © Esri',
      maxZoom: 19,
    });

    baseLayersRef.current = { map: mapLayer, satellite: satelliteLayer };
 
    mapLayer.addTo(map);

    mapRef.current = map;

    // Handle resizing issues when container becomes visible
    setTimeout(() => {
      map.invalidateSize();
    }, 0);

    // Left-click handler for deselecting points/polygons
    const handleMapClick = (e: L.LeafletMouseEvent) => {
      const target = e.originalEvent.target as HTMLElement;

      // Don't deselect if clicking on a polygon layer (marker, path, etc.)
      if (target && (target.tagName === 'path' || target.tagName === 'polyline' || target.closest('.leaflet-interactive') || target.closest('.custom-polygon-marker'))) {
        return;
      }

      // Deselect point and polygon when clicking on empty map area
      setSelectedPointId(null);
      setSelectedPolygonId(null);
    };
    map.on('click', handleMapClick);

    // Right-click handler for creating polygon points
    const handleContextMenu = (e: L.LeafletMouseEvent) => {
      e.originalEvent.preventDefault();

      const target = e.originalEvent.target as HTMLElement;

      // Don't add points if right-clicking on a polygon marker
      if (target && target.closest('.custom-polygon-marker')) {
        return;
      }

      // Add point to polygon
      const newPoint: PolygonPoint = {
        lat: e.latlng.lat,
        lon: e.latlng.lng,
        id: `${Date.now()}-${Math.random()}`,
      };

      setPolygons((prev) => {
        // Find active polygon (not closed)
        const activePolygon = prev.find((p) => !p.isClosed);

        if (activePolygon) {
          // Add point to active polygon
          const updatedPolygons = prev.map((p) =>
            p.id === activePolygon.id
              ? { ...p, points: [...p.points, newPoint] }
              : p
          );
          setActivePolygonId(activePolygon.id);
          return updatedPolygons;
        } else {
          // Create new polygon
          const newPolygonId = `polygon-${Date.now()}-${Math.random()}`;
          setActivePolygonId(newPolygonId);
          setSelectedPolygonId(null); // Deselect any selected polygon when starting new one
          setSelectedPointId(null); // Deselect any selected point when starting new one
          return [
            ...prev,
            {
              id: newPolygonId,
              points: [newPoint],
              isClosed: false,
            },
          ];
        }
      });
    };
    map.on('contextmenu', handleContextMenu);

    return () => {
      map.off('click', handleMapClick);
      map.off('contextmenu', handleContextMenu);
      // Clean up polygon layers
      polygonLayersRef.current.forEach((layers) => {
        if (layers.polyline) {
          map.removeLayer(layers.polyline);
        }
        if (layers.polygon) {
          map.removeLayer(layers.polygon);
        }
        layers.markers.forEach((marker) => {
          map.removeLayer(marker);
        });
      });
      polygonLayersRef.current.clear();
      // Clean up offset polygon layers
      offsetPolygonLayersRef.current.forEach((polyline) => {
        map.removeLayer(polyline);
      });
      offsetPolygonLayersRef.current.clear();
      map.remove();
      mapRef.current = null;
      if (heatLayerRef.current) {
        map.removeLayer(heatLayerRef.current);
        heatLayerRef.current = null;
      }
      if (allHeatLayerRef.current) {
        map.removeLayer(allHeatLayerRef.current);
        allHeatLayerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    const baseLayers = baseLayersRef.current;
    if (!map || !baseLayers.map || !baseLayers.satellite) {
      return;
    }

    if (useSatellite) {
      if (map.hasLayer(baseLayers.map)) {
        map.removeLayer(baseLayers.map);
      }
      if (!map.hasLayer(baseLayers.satellite)) {
        baseLayers.satellite.addTo(map);
      }
    } else {
      if (map.hasLayer(baseLayers.satellite)) {
        map.removeLayer(baseLayers.satellite);
      }
      if (!map.hasLayer(baseLayers.map)) {
        baseLayers.map.addTo(map);
      }
    }
  }, [useSatellite]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }

    if (allHeatLayerRef.current) {
      map.removeLayer(allHeatLayerRef.current);
      allHeatLayerRef.current = null;
    }

    if (heatLayerRef.current) {
      map.removeLayer(heatLayerRef.current);
      heatLayerRef.current = null;
    }

    if (showAllRosbags && allPoints.length) {
      const allMaxCount = Math.max(...allPoints.map((point) => point.count || 0));
      const allHeatmapData = allPoints.map((point) => {
        const intensity = allMaxCount ? Math.max(point.count / allMaxCount, 0.05) : 0.1;
        return [point.lat, point.lon, intensity] as [number, number, number];
      });

      const allLayerFactory = (L as typeof L & {
        heatLayer?: (latlngs: [number, number, number?][], options?: Record<string, unknown>) => L.Layer;
      }).heatLayer;

      if (allLayerFactory) {
        const currentZoom = map.getZoom();
        const zoomFactor = calculateHeatmapZoomFactor(currentZoom);
        const allLayer = allLayerFactory(allHeatmapData, {
          minOpacity: 0.2,
          maxZoom: 18,
          radius: HEATMAP_BASE_RADIUS * zoomFactor,
          blur: HEATMAP_BASE_BLUR * zoomFactor,
          gradient: {
            0.2: '#444127',
            0.4: '#857e42',
            0.6: '#d6ca62',
            1: '#fff8b0',
          },
        });
        allLayer.addTo(map);
        allHeatLayerRef.current = allLayer;
        
        // Set pointer-events: none on heatmap canvas so clicks pass through to polygons
        setTimeout(() => {
          const mapContainer = map.getContainer();
          const heatmapCanvases = mapContainer.querySelectorAll('canvas.leaflet-heatmap-layer');
          heatmapCanvases.forEach((canvas) => {
            (canvas as HTMLElement).style.pointerEvents = 'none';
          });
        }, 0);
      }
    }

    if (!points.length) {
      return;
    }

    const maxCount = Math.max(...points.map((point) => point.count || 0));
    const heatmapData = points.map((point) => {
      const intensity = maxCount ? Math.max(point.count / maxCount, 0.05) : 0.1;
      return [point.lat, point.lon, intensity] as [number, number, number];
    });

    const heatLayerFactory = (L as typeof L & {
      heatLayer?: (latlngs: [number, number, number?][], options?: Record<string, unknown>) => L.Layer;
    }).heatLayer;

    if (heatLayerFactory) {
      const currentZoom = map.getZoom();
      const zoomFactor = calculateHeatmapZoomFactor(currentZoom);
      // Keep current rosbag at full opacity so it overlays on top of "all rosbags" layer
      const heatLayer = heatLayerFactory(heatmapData, {
        minOpacity: 0.2,
        maxZoom: 18,
        radius: HEATMAP_BASE_RADIUS * zoomFactor,
        blur: HEATMAP_BASE_BLUR * zoomFactor,
        gradient: {
          0.2: 'blue',
          0.4: 'lime',
          0.6: 'orange',
          1: 'red',
        },
      });
      heatLayer.addTo(map);
      heatLayerRef.current = heatLayer;
      
      // Set pointer-events: none on heatmap canvas so clicks pass through to polygons
      setTimeout(() => {
        const mapContainer = map.getContainer();
        const heatmapCanvases = mapContainer.querySelectorAll('canvas.leaflet-heatmap-layer');
        heatmapCanvases.forEach((canvas) => {
          (canvas as HTMLElement).style.pointerEvents = 'none';
        });
      }, 0);
    }

    // Only reset zoom when points actually change, not when showAllRosbags toggles
    const pointsChanged = prevPointsRef.current.length === 0 || 
      points.length !== prevPointsRef.current.length || 
      points.some((p, i) => !prevPointsRef.current[i] || 
        p.lat !== prevPointsRef.current[i].lat || 
        p.lon !== prevPointsRef.current[i].lon);
    
    if (pointsChanged && points.length > 0) {
      const latLngs: L.LatLngTuple[] = points.map((point) => [point.lat, point.lon]);

      if (latLngs.length === 1) {
        map.setView(latLngs[0], 15);
      } else {
        map.fitBounds(L.latLngBounds(latLngs), { padding: [32, 32] });
      }
      
      prevPointsRef.current = [...points];
    }

    map.invalidateSize();
    
  }, [points, showAllRosbags, allPoints]);

  // Generate consistent color for each mcap_id
  const getMcapColor = (mcapId: string): string => {
    let hash = 0;
    for (let i = 0; i < mcapId.length; i++) {
      hash = mcapId.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue}, 70%, 50%)`;
  };

  // Calculate convex hull for a set of points (simplified version)
  const calculateConvexHull = (points: { lat: number; lon: number }[]): L.LatLngTuple[] => {
    if (points.length < 3) {
      // For less than 3 points, return a simple circle or just the points
      if (points.length === 1) {
        const p = points[0];
        // Create a small circle around the point
        const radius = 0.001; // ~100m
        const circlePoints: L.LatLngTuple[] = [];
        for (let i = 0; i < 16; i++) {
          const angle = (i / 16) * 2 * Math.PI;
          circlePoints.push([
            p.lat + radius * Math.cos(angle),
            p.lon + radius * Math.sin(angle) / Math.cos(p.lat * Math.PI / 180)
          ]);
        }
        return circlePoints;
      }
      return points.map(p => [p.lat, p.lon] as L.LatLngTuple);
    }

    // Graham scan algorithm for convex hull
    const sorted = [...points].sort((a, b) => {
      if (a.lat !== b.lat) return a.lat - b.lat;
      return a.lon - b.lon;
    });

    const cross = (o: { lat: number; lon: number }, a: { lat: number; lon: number }, b: { lat: number; lon: number }) => {
      return (a.lat - o.lat) * (b.lon - o.lon) - (a.lon - o.lon) * (b.lat - o.lat);
    };

    const lower: { lat: number; lon: number }[] = [];
    for (const point of sorted) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
        lower.pop();
      }
      lower.push(point);
    }

    const upper: { lat: number; lon: number }[] = [];
    for (let i = sorted.length - 1; i >= 0; i--) {
      const point = sorted[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
        upper.pop();
      }
      upper.push(point);
    }

    upper.pop();
    lower.pop();

    const hull = [...lower, ...upper];
    return hull.map(p => [p.lat, p.lon] as L.LatLngTuple);
  };

  // Calculate centroid of points
  const calculateCentroid = (points: { lat: number; lon: number }[]): { lat: number; lon: number } => {
    if (points.length === 0) return { lat: 0, lon: 0 };
    const sum = points.reduce((acc, p) => ({ lat: acc.lat + p.lat, lon: acc.lon + p.lon }), { lat: 0, lon: 0 });
    return { lat: sum.lat / points.length, lon: sum.lon / points.length };
  };

  // Effect to handle mcap bordered visualization
  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }

    // Remove existing mcap border layers
    mcapBorderLayersRef.current.forEach((layers) => {
      layers.borders.forEach(border => map.removeLayer(border));
      layers.labels.forEach(label => map.removeLayer(label));
      layers.markers.forEach(marker => map.removeLayer(marker));
    });
    mcapBorderLayersRef.current.clear();

    // Only show borders when showMcaps is enabled and we have data
    if (!showMcaps || !mcapLocationPoints.length || !availableMcaps.length || selectedMcapIndex >= availableMcaps.length) {
      return;
    }

    const selectedMcap = availableMcaps[selectedMcapIndex];
    if (!selectedMcap) {
      return;
    }

    const mcapId = selectedMcap.id;
    const color = getMcapColor(mcapId);

    // Filter points that belong to the selected mcap
    const mcapPoints: { lat: number; lon: number; count: number; hasMultiple: boolean }[] = [];

    mcapLocationPoints.forEach((point) => {
      if (point.mcaps[mcapId]) {
        const mcapCount = Object.keys(point.mcaps).length;
        mcapPoints.push({
          lat: point.lat,
          lon: point.lon,
          count: point.mcaps[mcapId],
          hasMultiple: mcapCount > 1
        });
      }
    });

    if (mcapPoints.length === 0) {
      return;
    }

    // Remove existing mcap heat layer if present
    if (mcapHeatLayerRef.current) {
      map.removeLayer(mcapHeatLayerRef.current);
      mcapHeatLayerRef.current = null;
    }

    // Calculate centroid for label
    const centroid = calculateCentroid(mcapPoints);

    // Create inverted heatmap for mcap points
    // Calculate global max across ALL mcaps for absolute coloring
    const globalMaxCount = Math.max(
      ...mcapLocationPoints.flatMap(point => Object.values(point.mcaps)),
      0
    );
    const mcapHeatmapData = mcapPoints.map((point) => {
      const intensity = globalMaxCount ? Math.max(point.count / globalMaxCount, 0.05) : 0.1;
      return [point.lat, point.lon, intensity] as [number, number, number];
    });

    const heatLayerFactory = (L as typeof L & {
      heatLayer?: (latlngs: [number, number, number?][], options?: Record<string, unknown>) => L.Layer;
    }).heatLayer;

    if (heatLayerFactory) {
      // Inverted color gradient (red -> orange -> yellow -> cyan -> blue)
      const currentZoom = map.getZoom();
      const zoomFactor = calculateHeatmapZoomFactor(currentZoom);
      const mcapHeatLayer = heatLayerFactory(mcapHeatmapData, {
        minOpacity: 0.3,
        maxZoom: 18,
        radius: HEATMAP_BASE_RADIUS * zoomFactor,
        blur: HEATMAP_BASE_BLUR * zoomFactor,
        gradient: {
          0.2: 'cyan',
          0.4: 'yellow',
          0.6: 'red',
          1: 'purple',
        },
      });
      mcapHeatLayer.addTo(map);
      mcapHeatLayerRef.current = mcapHeatLayer;
      
      // Set pointer-events: none on heatmap canvas so clicks pass through
      setTimeout(() => {
        const mapContainer = map.getContainer();
        const heatmapCanvases = mapContainer.querySelectorAll('canvas.leaflet-heatmap-layer');
        heatmapCanvases.forEach((canvas) => {
          (canvas as HTMLElement).style.pointerEvents = 'none';
        });
      }, 0);
    }

    // Create label marker at centroid
    const labelIcon = L.divIcon({
      className: 'mcap-label',
      html: `<div style="
        background-color: rgba(0, 0, 0, 0.8);
        color: ${escapeHtml(color)};
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
        border: 2px solid ${escapeHtml(color)};
        white-space: nowrap;
        text-align: center;
      ">${escapeHtml(mcapId)}</div>`,
      iconSize: [100, 30],
      iconAnchor: [50, 15],
    });

    const labelMarker = L.marker([centroid.lat, centroid.lon], { icon: labelIcon });
    labelMarker.addTo(map);

    // Store layers for cleanup
    mcapBorderLayersRef.current.set(mcapId, {
      borders: [],
      labels: [labelMarker],
      markers: []
    });

    return () => {
      if (mcapHeatLayerRef.current) {
        map.removeLayer(mcapHeatLayerRef.current);
        mcapHeatLayerRef.current = null;
      }
      mcapBorderLayersRef.current.forEach((layers) => {
        layers.borders.forEach(border => map.removeLayer(border));
        layers.labels.forEach(label => map.removeLayer(label));
        layers.markers.forEach(marker => map.removeLayer(marker));
      });
      mcapBorderLayersRef.current.clear();
    };
  }, [showMcaps, mcapLocationPoints, availableMcaps, selectedMcapIndex]);

  // Effect to handle hover tooltip for mcap locations
  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }

    // Mouse move handler to show mcap info on hover
    const handleMouseMove = (e: L.LeafletMouseEvent) => {
      if (!showMcaps || !mcapLocationPoints.length) {
        // Remove tooltip if mcap mode is off
        if (hoverTooltipRef.current) {
          map.closePopup(hoverTooltipRef.current);
          hoverTooltipRef.current = null;
        }
        return;
      }

      // Calculate distance to find nearest point (in degrees, roughly)
      // For small distances, we can use simple Euclidean distance
      let nearestPoint: McapLocationPoint | null = null;
      let minDistance = Infinity;
      const threshold = 0.0001; // ~10m at equator (lower threshold for closer proximity)

      for (const point of mcapLocationPoints) {
        const dx = point.lat - e.latlng.lat;
        const dy = point.lon - e.latlng.lng;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < minDistance && distance < threshold) {
          minDistance = distance;
          nearestPoint = point;
        }
      }

      // Remove existing tooltip
      if (hoverTooltipRef.current) {
        map.closePopup(hoverTooltipRef.current);
        hoverTooltipRef.current = null;
      }

      // Show tooltip if we found a nearby point
      if (nearestPoint) {
        const mcapEntries = Object.entries(nearestPoint.mcaps).sort((a, b) => (b[1] as number) - (a[1] as number));
        // Escape each entry to prevent XSS before joining with <br>
        const mcapList = mcapEntries.map(([id, count]) => `${escapeHtml(id)} (${escapeHtml(count)})`).join('<br>');

        const popup = L.popup({
          closeButton: false,
          className: 'mcap-hover-popup',
          offset: [0, -10],
        })
          .setLatLng(e.latlng)
          .setContent(`
            <div style="
              background-color: rgba(0, 0, 0, 0.85);
              color: white;
              padding: 8px 12px;
              border-radius: 4px;
              font-size: 12px;
              max-width: 200px;
            ">
              <strong>MCAPs at location:</strong><br>
              ${mcapList}
            </div>
          `)
          .openOn(map);

        hoverTooltipRef.current = popup;
      }
    };

    map.on('mousemove', handleMouseMove);

    return () => {
      map.off('mousemove', handleMouseMove);
      if (hoverTooltipRef.current) {
        map.closePopup(hoverTooltipRef.current);
        hoverTooltipRef.current = null;
      }
    };
  }, [showMcaps, mcapLocationPoints]);

  // Effect to handle polygon drawing
  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }

    // Remove all existing polygon layers
    polygonLayersRef.current.forEach((layers, polygonId) => {
      if (layers.polyline) {
        map.removeLayer(layers.polyline);
      }
      if (layers.polygon) {
        map.removeLayer(layers.polygon);
      }
      layers.markers.forEach((marker) => {
        map.removeLayer(marker);
      });
    });
    polygonLayersRef.current.clear();

    // Render each polygon
    polygons.forEach((polygon) => {
      if (!polygon.points.length) {
        return;
      }

      const markers: L.Marker[] = [];
      const isActive = polygon.id === activePolygonId && !polygon.isClosed;

      // Clean points to remove duplicates before rendering
      let cleanedPoints = [...polygon.points];
      if (cleanedPoints.length > 1) {
        // Remove duplicate first/last point if polygon is closed
        const first = cleanedPoints[0];
        const last = cleanedPoints[cleanedPoints.length - 1];
        if (polygon.isClosed && first.lat === last.lat && first.lon === last.lon && cleanedPoints.length > 3) {
          cleanedPoints = cleanedPoints.slice(0, -1);
        }
        
        // Remove any other duplicate consecutive points
        const uniquePoints: PolygonPoint[] = [cleanedPoints[0]];
        for (let i = 1; i < cleanedPoints.length; i++) {
          const current = cleanedPoints[i];
          const prev = uniquePoints[uniquePoints.length - 1];
          const dist = Math.sqrt(
            Math.pow(current.lat - prev.lat, 2) +
            Math.pow(current.lon - prev.lon, 2)
          );
          if (dist > 1e-9) {
            uniquePoints.push(current);
          }
        }
        cleanedPoints = uniquePoints;
      }
      
      // Create markers for each point
      cleanedPoints.forEach((point, index) => {
        const isFirstPoint = index === 0;
        const isLastPoint = index === cleanedPoints.length - 1;
        const isPointSelected = selectedPointId?.polygonId === polygon.id && selectedPointId?.pointId === point.id;

        // Create custom icon - highlight selected points
        const icon = L.divIcon({
          className: 'custom-polygon-marker',
          html: `<div style="
            width: ${isPointSelected ? '20px' : '16px'};
            height: ${isPointSelected ? '20px' : '16px'};
            border-radius: 50%;
            background-color: ${isPointSelected ? '#e91e63' : isFirstPoint ? '#4caf50' : isLastPoint && isActive ? '#ff9800' : '#2196f3'};
            border: ${isPointSelected ? '3px solid #fff' : '2px solid white'};
            box-shadow: ${isPointSelected ? '0 0 8px rgba(233,30,99,0.6), 0 2px 4px rgba(0,0,0,0.3)' : '0 2px 4px rgba(0,0,0,0.3)'};
            cursor: ${polygon.isClosed ? 'pointer' : 'move'};
            transition: all 0.15s ease;
          "></div>`,
          iconSize: [isPointSelected ? 20 : 16, isPointSelected ? 20 : 16],
          iconAnchor: [isPointSelected ? 10 : 8, isPointSelected ? 10 : 8],
        });

        const marker = L.marker([point.lat, point.lon], {
          icon,
          draggable: true,
        });

        // Add click handler based on polygon state
        if (isActive) {
          marker.on('click', (e) => {
            L.DomEvent.stopPropagation(e);

            if (isFirstPoint && polygon.points.length >= 3) {
              // Click on first point with >= 3 points closes the polygon
              setPolygons((prev) =>
                prev.map((p) => {
                  if (p.id !== polygon.id) return p;

                  // Clean duplicates when closing
                  let cleanedPoints = [...p.points];
                  if (cleanedPoints.length > 1) {
                    // Remove duplicate first/last point
                    const first = cleanedPoints[0];
                    const last = cleanedPoints[cleanedPoints.length - 1];
                    if (first.lat === last.lat && first.lon === last.lon && cleanedPoints.length > 3) {
                      cleanedPoints = cleanedPoints.slice(0, -1);
                    }

                    // Remove any other duplicate consecutive points
                    const uniquePoints: PolygonPoint[] = [cleanedPoints[0]];
                    for (let i = 1; i < cleanedPoints.length; i++) {
                      const current = cleanedPoints[i];
                      const prev = uniquePoints[uniquePoints.length - 1];
                      const dist = Math.sqrt(
                        Math.pow(current.lat - prev.lat, 2) +
                        Math.pow(current.lon - prev.lon, 2)
                      );
                      if (dist > 1e-9) {
                        uniquePoints.push(current);
                      }
                    }
                    cleanedPoints = uniquePoints;
                  }

                  return { ...p, points: cleanedPoints, isClosed: true };
                })
              );
              setActivePolygonId(null);
              setSelectedPointId(null);
            } else if (!isFirstPoint) {
              // Click on any other point selects it for deletion
              setSelectedPointId({ polygonId: polygon.id, pointId: point.id });
            }
          });
        } else if (polygon.isClosed) {
          // For closed polygons, click selects the point
          marker.on('click', (e) => {
            L.DomEvent.stopPropagation(e);
            setSelectedPointId({ polygonId: polygon.id, pointId: point.id });
            setSelectedPolygonId(null); // Deselect polygon when selecting a point
          });
        }

        // Add dragend handler to update polygon point position
        marker.on('dragend', () => {
          const newLatLng = marker.getLatLng();
          setPolygons((prev) =>
            prev.map((p) => {
              if (p.id !== polygon.id) return p;
              
              // Find the corresponding point in the original points array
              // Use the point's ID or position to match
              const cleanedPoint = cleanedPoints[index];
              const originalIndex = p.points.findIndex(
                (pt) => 
                  (pt.id === cleanedPoint.id) ||
                  (Math.abs(pt.lat - cleanedPoint.lat) < 1e-9 && Math.abs(pt.lon - cleanedPoint.lon) < 1e-9)
              );
              
              if (originalIndex === -1) return p;
              
              // Update the point and also clean duplicates
              let updatedPoints = p.points.map((pt, idx) =>
                idx === originalIndex
                  ? { ...pt, lat: newLatLng.lat, lon: newLatLng.lng }
                  : pt
              );
              
              // Clean duplicates after update
              if (updatedPoints.length > 1) {
                const first = updatedPoints[0];
                const last = updatedPoints[updatedPoints.length - 1];
                if (p.isClosed && first.lat === last.lat && first.lon === last.lon && updatedPoints.length > 3) {
                  updatedPoints = updatedPoints.slice(0, -1);
                }
                
                // Remove consecutive duplicates
                const uniquePoints: PolygonPoint[] = [updatedPoints[0]];
                for (let i = 1; i < updatedPoints.length; i++) {
                  const current = updatedPoints[i];
                  const prev = uniquePoints[uniquePoints.length - 1];
                  const dist = Math.sqrt(
                    Math.pow(current.lat - prev.lat, 2) +
                    Math.pow(current.lon - prev.lon, 2)
                  );
                  if (dist > 1e-9) {
                    uniquePoints.push(current);
                  }
                }
                updatedPoints = uniquePoints;
              }
              
              return { ...p, points: updatedPoints };
            })
          );
        });

        marker.addTo(map);
        markers.push(marker);
      });

      // Draw polyline connecting points (use cleaned points from above)
      const latLngs = cleanedPoints.map((p) => [p.lat, p.lon] as L.LatLngTuple);
      const isSelected = polygon.id === selectedPolygonId;
      let polyline: L.Polyline | null = null;
      let polygonFill: L.Polygon | null = null;
      
      if (latLngs.length > 1) {
        polyline = L.polyline(latLngs, {
          color: isSelected ? '#ff9800' : '#2196f3',
          weight: isSelected ? 3 : 2,
          opacity: 0.8,
          dashArray: polygon.isClosed ? undefined : '5, 5',
        });

        // Add click handler to select polygon
        polyline.off('click'); // Remove any existing handlers first
        polyline.on('click', (e) => {
          L.DomEvent.stopPropagation(e);
          setSelectedPolygonId(polygon.id);
        });

        // Add mousemove handler for edge hover detection
        polyline.on('mousemove', (e) => {
          const mouseLatLng = e.latlng;
          let closestDist = Infinity;
          let closestPoint: { lat: number; lon: number; segmentIndex: number } | null = null;

          // Check each segment
          for (let i = 0; i < cleanedPoints.length - 1; i++) {
            const p1 = cleanedPoints[i];
            const p2 = cleanedPoints[i + 1];
            const closest = closestPointOnSegment(
              mouseLatLng.lat, mouseLatLng.lng,
              p1.lat, p1.lon,
              p2.lat, p2.lon
            );
            const dist = Math.sqrt(
              Math.pow(closest.x - mouseLatLng.lat, 2) +
              Math.pow(closest.y - mouseLatLng.lng, 2)
            );
            if (dist < closestDist) {
              closestDist = dist;
              closestPoint = { lat: closest.x, lon: closest.y, segmentIndex: i };
            }
          }

          // For closed polygons, also check the closing segment
          if (polygon.isClosed && cleanedPoints.length >= 3) {
            const p1 = cleanedPoints[cleanedPoints.length - 1];
            const p2 = cleanedPoints[0];
            const closest = closestPointOnSegment(
              mouseLatLng.lat, mouseLatLng.lng,
              p1.lat, p1.lon,
              p2.lat, p2.lon
            );
            const dist = Math.sqrt(
              Math.pow(closest.x - mouseLatLng.lat, 2) +
              Math.pow(closest.y - mouseLatLng.lng, 2)
            );
            if (dist < closestDist) {
              closestDist = dist;
              closestPoint = { lat: closest.x, lon: closest.y, segmentIndex: cleanedPoints.length - 1 };
            }
          }

          if (closestPoint) {
            setEdgeHoverInfo({
              polygonId: polygon.id,
              segmentIndex: closestPoint.segmentIndex,
              lat: closestPoint.lat,
              lon: closestPoint.lon,
            });
          }
        });

        polyline.on('mouseout', () => {
          setEdgeHoverInfo(null);
        });

        // Add contextmenu handler for inserting points on edge
        polyline.on('contextmenu', (e) => {
          L.DomEvent.stopPropagation(e);
          e.originalEvent.preventDefault();

          const mouseLatLng = e.latlng;
          let closestDist = Infinity;
          let closestPoint: { lat: number; lon: number; segmentIndex: number } | null = null;

          // Find the closest point on any segment
          for (let i = 0; i < cleanedPoints.length - 1; i++) {
            const p1 = cleanedPoints[i];
            const p2 = cleanedPoints[i + 1];
            const closest = closestPointOnSegment(
              mouseLatLng.lat, mouseLatLng.lng,
              p1.lat, p1.lon,
              p2.lat, p2.lon
            );
            const dist = Math.sqrt(
              Math.pow(closest.x - mouseLatLng.lat, 2) +
              Math.pow(closest.y - mouseLatLng.lng, 2)
            );
            if (dist < closestDist) {
              closestDist = dist;
              closestPoint = { lat: closest.x, lon: closest.y, segmentIndex: i };
            }
          }

          // For closed polygons, also check the closing segment
          if (polygon.isClosed && cleanedPoints.length >= 3) {
            const p1 = cleanedPoints[cleanedPoints.length - 1];
            const p2 = cleanedPoints[0];
            const closest = closestPointOnSegment(
              mouseLatLng.lat, mouseLatLng.lng,
              p1.lat, p1.lon,
              p2.lat, p2.lon
            );
            const dist = Math.sqrt(
              Math.pow(closest.x - mouseLatLng.lat, 2) +
              Math.pow(closest.y - mouseLatLng.lng, 2)
            );
            if (dist < closestDist) {
              closestDist = dist;
              closestPoint = { lat: closest.x, lon: closest.y, segmentIndex: cleanedPoints.length - 1 };
            }
          }

          if (closestPoint) {
            // Insert new point after segmentIndex
            const newPoint: PolygonPoint = {
              lat: closestPoint.lat,
              lon: closestPoint.lon,
              id: `${Date.now()}-${Math.random()}`,
            };

            setPolygons((prev) =>
              prev.map((p) => {
                if (p.id !== polygon.id) return p;

                // Find the original index in p.points that corresponds to cleanedPoints[segmentIndex]
                const cleanedPoint = cleanedPoints[closestPoint!.segmentIndex];
                const originalIndex = p.points.findIndex(
                  (pt) =>
                    pt.id === cleanedPoint.id ||
                    (Math.abs(pt.lat - cleanedPoint.lat) < 1e-9 && Math.abs(pt.lon - cleanedPoint.lon) < 1e-9)
                );

                if (originalIndex === -1) return p;

                // Insert after the found index
                const newPoints = [...p.points];
                newPoints.splice(originalIndex + 1, 0, newPoint);
                return { ...p, points: newPoints };
              })
            );

            setEdgeHoverInfo(null);
          }
        });

        polyline.addTo(map);
      }

      // If polygon is closed, draw filled polygon
      if (polygon.isClosed && latLngs.length >= 3) {
        polygonFill = L.polygon(latLngs, {
          color: isSelected ? '#ff9800' : '#2196f3',
          weight: isSelected ? 3 : 2,
          opacity: 0.8,
          fillColor: isSelected ? '#ff9800' : '#2196f3',
          fillOpacity: isSelected ? 0.3 : 0.2,
        });

        // Add click handler to select polygon
        polygonFill.off('click'); // Remove any existing handlers first
        polygonFill.on('click', (e) => {
          L.DomEvent.stopPropagation(e);
          setSelectedPolygonId(polygon.id);
        });

        // Helper to find closest edge point for closed polygon
        const findClosestEdgePoint = (mouseLatLng: L.LatLng) => {
          let closestDist = Infinity;
          let closestPoint: { lat: number; lon: number; segmentIndex: number } | null = null;

          // Check each segment including the closing segment
          for (let i = 0; i < cleanedPoints.length; i++) {
            const p1 = cleanedPoints[i];
            const p2 = cleanedPoints[(i + 1) % cleanedPoints.length]; // Wrap around for closing segment
            const closest = closestPointOnSegment(
              mouseLatLng.lat, mouseLatLng.lng,
              p1.lat, p1.lon,
              p2.lat, p2.lon
            );
            const dist = Math.sqrt(
              Math.pow(closest.x - mouseLatLng.lat, 2) +
              Math.pow(closest.y - mouseLatLng.lng, 2)
            );
            if (dist < closestDist) {
              closestDist = dist;
              closestPoint = { lat: closest.x, lon: closest.y, segmentIndex: i };
            }
          }

          return { closestPoint, closestDist };
        };

        // Add mousemove handler for edge hover detection on filled polygon
        polygonFill.on('mousemove', (e) => {
          const { closestPoint, closestDist } = findClosestEdgePoint(e.latlng);

          // Only show ghost marker if close to an edge (threshold in lat/lng units)
          // Roughly 0.0001 degrees is about 11 meters
          const edgeThreshold = 0.0015; // ~165 meters - wider detection area
          if (closestPoint && closestDist < edgeThreshold) {
            setEdgeHoverInfo({
              polygonId: polygon.id,
              segmentIndex: closestPoint.segmentIndex,
              lat: closestPoint.lat,
              lon: closestPoint.lon,
            });
          } else {
            setEdgeHoverInfo(null);
          }
        });

        polygonFill.on('mouseout', () => {
          setEdgeHoverInfo(null);
        });

        // Add contextmenu handler for inserting points on edge
        polygonFill.on('contextmenu', (e) => {
          const { closestPoint, closestDist } = findClosestEdgePoint(e.latlng);

          // Only insert if close to an edge
          const edgeThreshold = 0.0015; // ~165 meters
          if (closestPoint && closestDist < edgeThreshold) {
            L.DomEvent.stopPropagation(e);
            e.originalEvent.preventDefault();

            // Insert new point after segmentIndex
            const newPoint: PolygonPoint = {
              lat: closestPoint.lat,
              lon: closestPoint.lon,
              id: `${Date.now()}-${Math.random()}`,
            };

            setPolygons((prev) =>
              prev.map((p) => {
                if (p.id !== polygon.id) return p;

                // Find the original index in p.points that corresponds to cleanedPoints[segmentIndex]
                const cleanedPoint = cleanedPoints[closestPoint!.segmentIndex];
                const originalIndex = p.points.findIndex(
                  (pt) =>
                    pt.id === cleanedPoint.id ||
                    (Math.abs(pt.lat - cleanedPoint.lat) < 1e-9 && Math.abs(pt.lon - cleanedPoint.lon) < 1e-9)
                );

                if (originalIndex === -1) return p;

                // Insert after the found index
                const newPoints = [...p.points];
                newPoints.splice(originalIndex + 1, 0, newPoint);
                return { ...p, points: newPoints };
              })
            );

            setEdgeHoverInfo(null);
          }
        });

        polygonFill.addTo(map);
      }

      polygonLayersRef.current.set(polygon.id, {
        markers,
        polyline,
        polygon: polygonFill,
      });
    });
  }, [polygons, activePolygonId, selectedPolygonId, selectedPointId]);

  // Effect to render edge hover ghost marker
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Remove existing ghost marker
    if (edgeGhostMarkerRef.current) {
      map.removeLayer(edgeGhostMarkerRef.current);
      edgeGhostMarkerRef.current = null;
    }

    // Add new ghost marker if we have edge hover info
    if (edgeHoverInfo) {
      const ghostIcon = L.divIcon({
        className: 'edge-ghost-marker',
        html: `<div style="
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background-color: rgba(156, 39, 176, 0.7);
          border: 2px dashed white;
          box-shadow: 0 0 6px rgba(156, 39, 176, 0.5);
          cursor: pointer;
        "></div>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      });

      edgeGhostMarkerRef.current = L.marker([edgeHoverInfo.lat, edgeHoverInfo.lon], {
        icon: ghostIcon,
        interactive: false, // Don't intercept clicks - let them pass to polyline
      }).addTo(map);
    }

    return () => {
      if (edgeGhostMarkerRef.current && map) {
        map.removeLayer(edgeGhostMarkerRef.current);
        edgeGhostMarkerRef.current = null;
      }
    };
  }, [edgeHoverInfo]);

  // Effect to handle offset polygon rendering
  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }

    // Remove all existing offset polygon layers
    offsetPolygonLayersRef.current.forEach((polyline) => {
      map.removeLayer(polyline);
    });
    offsetPolygonLayersRef.current.clear();

    // Only render offset polygons if offsetDistance is not zero
    if (offsetDistance !== 0) {
      const closedPolygons = polygons.filter((p) => p.isClosed);
      
      // Process each polygon
      closedPolygons.forEach((polygon) => {
        if (polygon.points.length < 3) {
          return;
        }

        try {
          const offsetPoints = offsetPolygon(polygon.points, offsetDistance);
          
          if (offsetPoints.length >= 2) {
            const latLngs = offsetPoints.map((p) => [p.lat, p.lon] as L.LatLngTuple);
            // Close the polygon by adding the first point at the end (only if not already closed)
            const firstPoint = latLngs[0];
            const lastPoint = latLngs[latLngs.length - 1];
            const isAlreadyClosed = firstPoint[0] === lastPoint[0] && firstPoint[1] === lastPoint[1];
            if (!isAlreadyClosed) {
              latLngs.push([firstPoint[0], firstPoint[1]]);
            }
            
            // Create dark blue polyline (lines only, no fill, no points)
            const offsetPolyline = L.polyline(latLngs, {
              color: '#000080', // Dark blue
              weight: 2,
              opacity: 0.8,
              fill: false,
            });
            
            offsetPolyline.addTo(map);
            offsetPolygonLayersRef.current.set(polygon.id, offsetPolyline);
          }
        } catch (error) {
          console.error(`Error creating offset polygon for ${polygon.id}:`, error);
        }
      });
    }

    return () => {
      offsetPolygonLayersRef.current.forEach((polyline) => {
        map.removeLayer(polyline);
      });
      offsetPolygonLayersRef.current.clear();
    };
  }, [polygons, offsetDistance]);

  // Keyboard handler for deleting polygon points or entire polygons
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle backspace, delete, or entf if we're not in an input field
      const activeElement = document.activeElement;
      const isInputField = activeElement?.tagName === 'INPUT' || activeElement?.tagName === 'TEXTAREA' || (activeElement as HTMLElement)?.isContentEditable;

      if ((e.key === 'Backspace' || e.key === 'Delete' || e.key === 'Entf') && !isInputField) {
        e.preventDefault();
        e.stopPropagation();

        // Priority 1: If an individual point is selected, delete just that point
        if (selectedPointId) {
          setPolygons((prev) => {
            const targetPolygon = prev.find((p) => p.id === selectedPointId.polygonId);
            if (!targetPolygon) return prev;

            // Find and remove the selected point
            const updatedPoints = targetPolygon.points.filter((p) => p.id !== selectedPointId.pointId);

            // For closed polygons: need at least 3 points
            // For unclosed polygons: can have any number of points (even 0)
            if (targetPolygon.isClosed && updatedPoints.length < 3) {
              setSelectedPointId(null);
              setActivePolygonId(null);
              return prev.filter((p) => p.id !== selectedPointId.polygonId);
            }

            if (updatedPoints.length === 0) {
              setSelectedPointId(null);
              setActivePolygonId(null);
              return prev.filter((p) => p.id !== selectedPointId.polygonId);
            }

            // Otherwise, update the polygon with the remaining points (polygon automatically reconnects)
            return prev.map((p) =>
              p.id === selectedPointId.polygonId
                ? { ...p, points: updatedPoints }
                : p
            );
          });
          setSelectedPointId(null);
          return;
        }

        // Priority 2: If there's an active polygon being drawn, delete the last point
        if (activePolygonId) {
          setPolygons((prev) => {
            const activePolygon = prev.find((p) => p.id === activePolygonId);
            if (activePolygon && !activePolygon.isClosed && activePolygon.points.length > 0) {
              if (activePolygon.points.length === 1) {
                // Last point - remove the entire polygon
                setActivePolygonId(null);
                return prev.filter((p) => p.id !== activePolygonId);
              } else {
                // Remove the last point
                return prev.map((p) =>
                  p.id === activePolygonId
                    ? { ...p, points: p.points.slice(0, -1) }
                    : p
                );
              }
            }
            return prev;
          });
          return;
        }

        // Priority 3: If a closed polygon is selected, delete it entirely
        if (selectedPolygonId) {
          setPolygons((prev) => prev.filter((p) => p.id !== selectedPolygonId));
          setSelectedPolygonId(null);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedPolygonId, activePolygonId, selectedPointId]);

  // Sync polygons to sessionStorage whenever they change
  useEffect(() => {
    // Skip sync if we're currently restoring from cache
    if (isRestoringRef.current) {
      return;
    }
    
    if (polygons.length > 0) {
      try {
        sessionStorage.setItem('__BagSeekPositionalPolygons', JSON.stringify(polygons));
      } catch (e) {
        console.error('Failed to cache polygons:', e);
      }
    } else {
      // Only clear cache if polygons are explicitly empty (not on initial mount)
      // We check if there was a previous value to avoid clearing on first render
      try {
        const cached = sessionStorage.getItem('__BagSeekPositionalPolygons');
        if (cached) {
          sessionStorage.removeItem('__BagSeekPositionalPolygons');
        }
      } catch (e) {
        console.error('Failed to clear polygon cache:', e);
      }
    }
  }, [polygons]);

  // Check overlap when polygons change
  useEffect(() => {
    const closedPolygons = polygons.filter((p) => p.isClosed);
    const currentPolygonCount = polygons.length;
    const polygonDeleted = currentPolygonCount < prevPolygonCountRef.current;
    prevPolygonCountRef.current = currentPolygonCount;
    
    if (closedPolygons.length === 0) {
      // Clear overlap status if no closed polygons
      setRosbagOverlapStatus(new Map());
      setCheckingOverlap(false);
      // Only clear positional filter if a polygon was deleted
      if (polygonDeleted) {
        try {
          sessionStorage.removeItem('__BagSeekPositionalFilter');
          window.dispatchEvent(new Event('__BagSeekPositionalFilterChanged'));
        } catch (e) {
          console.error('Failed to clear positional filter:', e);
        }
      }
      return;
    }

    if (!rosbags.length) {
      return;
    }

    // Wait for boundaries to load before checking overlaps
    if (rosbagBoundaries === null) {
      return;
    }

    let cancelled = false;
    setCheckingOverlap(true);

    // Check overlap for all rosbags, using boundaries for fast pre-filtering
    const checkAllOverlaps = async () => {
      const overlapMap = new Map<string, boolean>();
      const hasBoundaries = Object.keys(rosbagBoundaries).length > 0;

      for (const rosbag of rosbags) {
        if (cancelled) break;

        // Pre-filter: if we have a boundary for this rosbag, check hull overlap first
        if (hasBoundaries) {
          const boundary = rosbagBoundaries[rosbag.name];
          if (boundary) {
            if (!boundaryOverlapsPolygons(boundary, closedPolygons, offsetDistance)) {
              overlapMap.set(rosbag.name, false);
              continue; // Hull doesn't overlap — skip expensive grid check
            }
          }
        }

        // Detailed check: fetch grid cells and test point-in-polygon
        const overlaps = await checkRosbagOverlap(rosbag.name, closedPolygons, offsetDistance);
        overlapMap.set(rosbag.name, overlaps);
      }

      if (!cancelled) {
        setRosbagOverlapStatus(overlapMap);
        setCheckingOverlap(false);

        // Only automatically update sessionStorage if a polygon was deleted
        if (polygonDeleted) {
          const overlappingRosbags: string[] = [];
          rosbags.forEach((rosbag) => {
            const overlaps = overlapMap.get(rosbag.name) ?? false;
            if (overlaps) {
              overlappingRosbags.push(rosbag.name);
            }
          });

          try {
            sessionStorage.setItem('__BagSeekPositionalFilter', JSON.stringify(overlappingRosbags));
            window.dispatchEvent(new Event('__BagSeekPositionalFilterChanged'));
          } catch (e) {
            console.error('Failed to update positional filter:', e);
          }
        }
      }
    };

    checkAllOverlaps();

    return () => {
      cancelled = true;
    };
  }, [polygons, rosbags, offsetDistance, rosbagBoundaries]);

  // MCAP overlap: which MCAPs are inside polygons (for MCAP slider background when showMcaps)
  useEffect(() => {
    const closedPolygons = polygons.filter((p) => p.isClosed);
    const sel = rosbags[selectedIndex];

    if (closedPolygons.length === 0 || !showMcaps || availableMcaps.length === 0 || !sel) {
      setMcapOverlapIds(null);
      return;
    }

    let cancelled = false;
    getMcapsInsidePolygons(sel.name, closedPolygons, offsetDistance).then((ids) => {
      if (!cancelled) setMcapOverlapIds(ids);
    });
    return () => { cancelled = true; };
  }, [polygons, rosbags, selectedIndex, availableMcaps, showMcaps, offsetDistance]);

  // Persist highlighted MCAP IDs to Export (MAP live selection - highlighted MCAPs in slider)
  useEffect(() => {
    const sel = rosbags[selectedIndex];
    const closedPolygons = polygons.filter((p) => p.isClosed);
    if (!sel || closedPolygons.length === 0) {
      try {
        sessionStorage.removeItem('__BagSeekMapMcapFilter');
      } catch {}
      return;
    }
    if (mcapOverlapIds === null || mcapOverlapIds.size === 0) {
      try {
        sessionStorage.removeItem('__BagSeekMapMcapFilter');
      } catch {}
      return;
    }
    try {
      sessionStorage.setItem('__BagSeekMapMcapFilter', JSON.stringify({ [sel.name]: Array.from(mcapOverlapIds) }));
      window.dispatchEvent(new Event('__BagSeekPositionalFilterChanged'));
    } catch {}
  }, [polygons, rosbags, selectedIndex, mcapOverlapIds]);

  const fetchAllPoints = useCallback(async () => {
    if (loadingAllPoints || allPointsLoaded) {
      return;
    }

    setLoadingAllPoints(true);
    setError(null);

    try {
      const response = await fetch('/api/positions/all');
      if (!response.ok) {
        throw new Error(`Failed to load aggregated positional data (${response.status})`);
      }
      const data = await response.json();
      const aggregatedPoints: RosbagPoint[] = Array.isArray(data.points) ? data.points : [];
      setAllPoints(aggregatedPoints);
      setAllPointsLoaded(true);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : 'Failed to load aggregated positional data');
      setShowAllRosbags(false);
    } finally {
      setLoadingAllPoints(false);
    }
  }, [loadingAllPoints, allPointsLoaded]);

  const selectedRosbag = rosbags[selectedIndex] ?? null;

  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#121212',
        color: '#ffffff',
        fontFamily: 'inherit',
        overflow: 'hidden',
      }}
    >
      <Box
        component="main"
        sx={{
          flex: 1,
          minHeight: 0,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <Box
          ref={mapContainerRef}
          tabIndex={0}
          sx={{
            position: 'absolute',
            inset: 0,
            border: '1px solid #2a2a2a',
            backgroundColor: '#0f0f0f',
            outline: 'none',
          }}
        />
        
        {checkingOverlap && (
          <Box
            sx={{
              position: 'absolute',
              top: 16,
              left: '50%',
              zIndex: 1300,
              px: 3,
              py: 1.5,
              borderRadius: 2,
              backgroundColor: 'rgba(18, 18, 18, 0.95)',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              backdropFilter: 'blur(8px)',
              transform: 'translateX(-50%)',
              animation: 'slideDown 0.3s ease-out',
              '@keyframes slideDown': {
                '0%': {
                  transform: 'translateX(-50%) translateY(-100%)',
                  opacity: 0,
                },
                '100%': {
                  transform: 'translateX(-50%) translateY(0)',
                  opacity: 1,
                },
              },
            }}
          >
            <Typography variant="body2" sx={{ color: '#fff', fontWeight: 500 }}>
              Computing overlaps...
            </Typography>
          </Box>
        )}

        <Box
          sx={{
            position: 'absolute',
            top: 16,
            right: 16,
            zIndex: 1200,
          }}
        >
          <Button
            variant="contained"
            color="primary"
            onClick={handleBack}
            sx={{
              borderRadius: 999,
              boxShadow: '0 12px 24px rgba(0, 0, 0, 0.35)',
            }}
          >
            Back
          </Button>
        </Box>

        <Box
          sx={{
            position: 'absolute',
            top: 16,
            left: 16,
            zIndex: 1200,
            display: 'flex',
            flexDirection: 'column',
            gap: 1,
            maxWidth: 'min(340px, 80vw)',
          }}
        >
          {loadingRosbags && (
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                px: 2,
                py: 1,
                borderRadius: 999,
                backgroundColor: 'rgba(18, 18, 18, 0.92)',
                boxShadow: '0 6px 16px rgba(0, 0, 0, 0.35)',
                color: '#fff',
              }}
            >
              <CircularProgress size={16} color="inherit" />
              <Typography variant="body2">Loading rosbags…</Typography>
            </Box>
          )}
          {!loadingRosbags && loadingPoints && selectedRosbag && (
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                px: 2,
                py: 1,
                borderRadius: 999,
                backgroundColor: 'rgba(18, 18, 18, 0.92)',
                boxShadow: '0 6px 16px rgba(0, 0, 0, 0.35)',
                color: '#fff',
              }}
            >
              <CircularProgress size={16} color="inherit" />
              <Typography variant="body2">Loading positional data…</Typography>
            </Box>
          )}
          {error && (
            <Alert severity="error" sx={{ backgroundColor: '#1f1f1f', color: '#ffb4a2' }}>
              {error}
            </Alert>
          )}
          
          {!loadingRosbags && !rosbags.length && !error && (
            <Alert severity="info" sx={{ backgroundColor: '#1f1f1f', color: '#bbdefb' }}>
              No rosbags available. Generate the positional lookup table to see data here.
            </Alert>
          )}
        </Box>

        {showMcaps && availableMcaps.length > 0 && (
          <Box
            sx={{
              position: 'absolute',
              left: '50%',
              bottom: 90,
              transform: 'translateX(-50%)',
              display: 'flex',
              alignItems: 'center',
              gap: 3,
              zIndex: 1000,
              pointerEvents: 'none', // Allow clicks to pass through to map
            }}
          >
            <Box
              sx={{
                width: 'min(620px, 70vw)',
                backgroundColor: 'rgba(18, 18, 18, 0.75)',
                borderRadius: '14px',
                boxShadow: '0 10px 30px rgba(0, 0, 0, 0.35)',
                px: 4,
                pt: 1.5,
                pb: 1.5,
                backdropFilter: 'blur(4px)',
                display: 'flex',
                alignItems: 'center',
                position: 'relative',
                pointerEvents: 'auto', // Enable pointer events for this container
              }}
            >
              {/* Background indicator for MCAPs outside polygons (like rosbag slider) */}
              {availableMcaps.length > 0 && (
                <Box
                  sx={{
                    position: 'absolute',
                    left: '32px',
                    right: '32px',
                    height: '4px',
                    display: 'flex',
                    gap: 0,
                    pointerEvents: 'none',
                    zIndex: 0,
                  }}
                >
                  {availableMcaps.map((mcap, index) => {
                    const overlaps = mcapOverlapIds === null || mcapOverlapIds.has(mcap.id);
                    const segmentWidth = 100 / availableMcaps.length;
                    return (
                      <Box
                        key={mcap.id}
                        sx={{
                          width: `${segmentWidth}%`,
                          height: '100%',
                          backgroundColor: (theme) =>
                            overlaps
                              ? `${theme.palette.primary.main}40`
                              : 'rgba(30, 30, 30, 0.4)',
                          borderRadius: index === 0 ? '2px 0 0 2px' : index === availableMcaps.length - 1 ? '0 2px 2px 0' : '0',
                        }}
                      />
                    );
                  })}
                </Box>
              )}
              <Slider
                value={Math.min(selectedMcapIndex, Math.max(availableMcaps.length - 1, 0))}
                onChange={(_, value) => {
                  if (typeof value === 'number') {
                    setSelectedMcapIndex(value);
                  }
                }}
                onClick={(e) => e.stopPropagation()}
                onMouseDown={(e) => e.stopPropagation()}
                min={0}
                max={Math.max(availableMcaps.length - 1, 0)}
                step={1}
                marks={[]}
                track={false}
                disabled={loadingMcaps || !availableMcaps.length}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => availableMcaps[value]?.id ?? ''}
                color="primary"
                sx={{
                  width: '100%',
                  position: 'relative',
                  zIndex: 1,
                  pointerEvents: 'auto',
                  '& .MuiSlider-thumb': {
                    width: 12,
                    height: 12,
                    boxShadow: (theme) => `0 0 0 3px ${theme.palette.primary.main}40`,
                    '&:hover, &.Mui-focusVisible': {
                      boxShadow: (theme) => `0 0 0 5px ${theme.palette.primary.main}55`,
                    },
                  },
                  '& .MuiSlider-rail': {
                    opacity: 0.2,
                  },
                  '& .MuiSlider-valueLabel': {
                    background: (theme) => theme.palette.background.paper,
                    borderRadius: 5,
                    fontSize: '8pt',
                    fontWeight: 600,
                    color: (theme) => theme.palette.text.primary,
                  },
                }}
              />
              {/* MCAP ID marks above + time-based TOD marks below */}
              {availableMcaps.length > 1 && (() => {
                const maxIdx = availableMcaps.length - 1;
                const rangeMap = new Map(mcapRanges.map(r => [String(r.mcapIdentifier), r]));

                // ID marks: small tick for ALL, text label every Nth for even spacing
                const maxIdLabels = 18;
                let idStep = 1;
                if (availableMcaps.length > maxIdLabels) {
                  idStep = 2;
                  while (Math.ceil(availableMcaps.length / idStep) > maxIdLabels) idStep++;
                }
                const idLabelIndices = new Set<number>();
                for (let i = 0; i <= maxIdx; i += idStep) idLabelIndices.add(i);

                // Build time→index mapping for TOD marks
                const mcapTimes: { index: number; ms: number }[] = [];
                availableMcaps.forEach((mcap, i) => {
                  const range = rangeMap.get(String(mcap.id));
                  if (range?.firstTimestampNs) {
                    const ms = Number(BigInt(range.firstTimestampNs) / BigInt(1_000_000));
                    mcapTimes.push({ index: i, ms });
                  }
                });

                // Generate time-based ticks with adaptive label density
                let timeTicks: { ms: number; frac: number; isLabeled: boolean }[] = [];
                if (mcapTimes.length >= 2) {
                  const startMs = mcapTimes[0].ms;
                  const endMs = mcapTimes[mcapTimes.length - 1].ms;
                  const durationSec = (endMs - startMs) / 1000;

                  // Base tick interval: finest granularity for small tick marks
                  let baseTickSec = 60;
                  if (durationSec < 180) baseTickSec = 30;
                  if (durationSec < 90) baseTickSec = 20;
                  if (durationSec < 60) baseTickSec = 10;

                  const startSec = startMs / 1000;
                  const endSec = endMs / 1000;
                  const firstTick = Math.ceil(startSec / baseTickSec) * baseTickSec;

                  // Interpolate time to slider fraction
                  const timeToFrac = (ms: number): number => {
                    if (ms <= mcapTimes[0].ms) return 0;
                    if (ms >= mcapTimes[mcapTimes.length - 1].ms) return 1;
                    for (let j = 0; j < mcapTimes.length - 1; j++) {
                      if (ms >= mcapTimes[j].ms && ms <= mcapTimes[j + 1].ms) {
                        const localFrac = (ms - mcapTimes[j].ms) / (mcapTimes[j + 1].ms - mcapTimes[j].ms);
                        const pos = mcapTimes[j].index + localFrac * (mcapTimes[j + 1].index - mcapTimes[j].index);
                        return pos / maxIdx;
                      }
                    }
                    return 0;
                  };

                  // Generate all base ticks
                  const allTicks: { sec: number; frac: number }[] = [];
                  for (let t = firstTick; t <= endSec; t += baseTickSec) {
                    allTicks.push({ sec: t, frac: timeToFrac(t * 1000) });
                  }

                  // Choose label interval: target similar count to ID labels
                  const targetLabels = idLabelIndices.size;
                  const niceLabelIntervals = [10, 20, 30, 60, 120, 180, 300, 600, 900, 1800, 3600];
                  let labelIntervalSec = 60;
                  for (const interval of niceLabelIntervals) {
                    if (interval < baseTickSec) continue;
                    const labelCount = allTicks.filter(t => t.sec % interval === 0).length;
                    if (labelCount <= targetLabels + 2) {
                      labelIntervalSec = interval;
                      break;
                    }
                  }

                  timeTicks = allTicks.map(tick => ({
                    ms: tick.sec * 1000,
                    frac: tick.frac,
                    isLabeled: tick.sec % labelIntervalSec === 0,
                  }));
                }

                return (
                  <>
                    {/* ID marks above rail: small tick for all, text for evenly-spaced subset */}
                    <Box sx={{ position: 'absolute', left: '32px', right: '32px', top: 0, bottom: 'calc(50% + 4px)', pointerEvents: 'none' }}>
                      {availableMcaps.map((mcap, i) => {
                        const frac = maxIdx > 0 ? i / maxIdx : 0;
                        const isLabeled = idLabelIndices.has(i);
                        return (
                          <div key={`id-${i}`} style={{ position: 'absolute', left: `${frac * 100}%`, bottom: 0, transform: 'translateX(-50%)', display: 'flex', flexDirection: 'column', alignItems: 'center', pointerEvents: 'none' }}>
                            {isLabeled && (
                              <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.5)', whiteSpace: 'nowrap', marginBottom: 1 }}>
                                {mcap.id}
                              </span>
                            )}
                            <div style={{ width: 1, height: isLabeled ? 5 : 3, backgroundColor: isLabeled ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.15)' }} />
                          </div>
                        );
                      })}
                    </Box>
                    {/* Time-based TOD marks below rail: small ticks for all, text for adaptive subset */}
                    <Box sx={{ position: 'absolute', left: '32px', right: '32px', top: 'calc(50% + 4px)', bottom: 0, pointerEvents: 'none' }}>
                      {timeTicks.map((tick, i) => {
                        const d = new Date(tick.ms);
                        const hh = d.getHours().toString().padStart(2, '0');
                        const mm = d.getMinutes().toString().padStart(2, '0');
                        const ss = d.getSeconds().toString().padStart(2, '0');
                        const label = tick.isLabeled
                          ? (d.getSeconds() === 0 ? `${hh}:${mm}` : `${hh}:${mm}:${ss}`)
                          : null;
                        return (
                          <div key={`tod-${i}`} style={{ position: 'absolute', left: `${tick.frac * 100}%`, top: 0, transform: 'translateX(-50%)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, pointerEvents: 'none' }}>
                            <div style={{ width: 1, height: tick.isLabeled ? 5 : 3, backgroundColor: tick.isLabeled ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.15)' }} />
                            {label && (
                              <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.35)', whiteSpace: 'nowrap', lineHeight: 1 }}>
                                {label}
                              </span>
                            )}
                          </div>
                        );
                      })}
                    </Box>
                  </>
                );
              })()}
            </Box>
          </Box>
        )}

        {rosbags.length > 1 && (
          <Box
            sx={{
              position: 'absolute',
              left: '50%',
              bottom: 32,
              transform: 'translateX(-50%)',
              display: 'flex',
              alignItems: 'center',
              gap: 3,
              zIndex: 1000,
              pointerEvents: 'none', // Allow clicks to pass through to map
            }}
          >
            <Box
              sx={{
                width: 'min(820px, 70vw)',
                backgroundColor: 'rgba(18, 18, 18, 0.75)',
                borderRadius: '14px',
                boxShadow: '0 10px 30px rgba(0, 0, 0, 0.35)',
                px: 4,
                pt: 0,
                pb: 1.5,
                backdropFilter: 'blur(4px)',
                display: 'flex',
                alignItems: 'center',
                position: 'relative',
                pointerEvents: 'auto', // Enable pointer events for this container
              }}
            >
              {/* Background indicator for non-overlapping rosbags */}
              {rosbags.length > 0 && (
                <Box
                  sx={{
                    position: 'absolute',
                    left: '32px',
                    right: '32px',
                    height: '4px',
                    display: 'flex',
                    gap: 0,
                    pointerEvents: 'none',
                    zIndex: 0,
                  }}
                >
                  {rosbags.map((rosbag, index) => {
                    const overlaps = rosbagOverlapStatus.get(rosbag.name) ?? true;
                    const segmentWidth = 100 / rosbags.length;
                    return (
                      <Box
                        key={rosbag.name}
                        sx={{
                          width: `${segmentWidth}%`,
                          height: '100%',
                          backgroundColor: (theme) =>
                            overlaps
                              ? `${theme.palette.primary.main}40`
                              : 'rgba(30, 30, 30, 0.4)',
                          borderRadius: index === 0 ? '2px 0 0 2px' : index === rosbags.length - 1 ? '0 2px 2px 0' : '0',
                        }}
                      />
                    );
                  })}
                </Box>
              )}
              <Slider
                value={Math.min(selectedIndex, Math.max(rosbags.length - 1, 0))}
                onChange={(_, value) => {
                  if (typeof value === 'number') {
                    setSelectedIndex(value);
                  }
                }}
                onClick={(e) => e.stopPropagation()}
                onMouseDown={(e) => e.stopPropagation()}
                min={0}
                max={Math.max(rosbags.length - 1, 0)}
                step={1}
                marks={[]}
                track={false}
                disabled={loadingRosbags || !rosbags.length}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) =>
                  formatTimestampLabel(rosbags[value]?.timestamp ?? null, rosbags[value]?.name ?? '')
                }
                color="primary"
                sx={{
                  width: '100%',
                  position: 'relative',
                  zIndex: 1,
                  pointerEvents: 'auto',
                  '& .MuiSlider-thumb': {
                    width: 12,
                    height: 12,
                    boxShadow: (theme) => `0 0 0 3px ${theme.palette.primary.main}40`,
                    '&:hover, &.Mui-focusVisible': {
                      boxShadow: (theme) => `0 0 0 5px ${theme.palette.primary.main}55`,
                    },
                  },
                  '& .MuiSlider-rail': {
                    opacity: 0.2,
                  },
                  '& .MuiSlider-valueLabel': {
                    background: (theme) => theme.palette.background.paper,
                    borderRadius: 5,
                    fontSize: '8pt',
                    fontWeight: 600,
                    color: (theme) => theme.palette.text.primary,
                  },
                }}
              />
              {/* Rosbag date marks below slider: small ticks for all, labels for evenly-spaced subset */}
              {rosbags.length > 1 && (() => {
                const maxIdx = rosbags.length - 1;
                const maxLabels = 12;
                let step = 1;
                if (rosbags.length > maxLabels) {
                  step = 2;
                  while (Math.ceil(rosbags.length / step) > maxLabels) step++;
                }
                const labelIndices = new Set<number>();
                for (let i = 0; i <= maxIdx; i += step) labelIndices.add(i);
                const fmt = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' });
                return (
                  <Box sx={{ position: 'absolute', left: '32px', right: '32px', pointerEvents: 'none' }}>
                    {rosbags.map((rosbag, i) => {
                      const frac = maxIdx > 0 ? i / maxIdx : 0;
                      const isLabeled = labelIndices.has(i);
                      const label = isLabeled && rosbag.timestamp ? fmt.format(new Date(rosbag.timestamp)) : '';
                      return (
                        <Box key={rosbag.name} sx={{ position: 'absolute', top: 'calc(50% + 5px)', left: `${frac * 100}%`, transform: 'translateX(-50%)', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                          <Box sx={{ width: '1px', height: isLabeled ? 5 : 3, bgcolor: isLabeled ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.15)' }} />
                          {label && (
                            <Typography sx={{ fontSize: 8, color: 'rgba(255,255,255,0.4)', whiteSpace: 'nowrap', lineHeight: 1.2 }}>
                              {label}
                            </Typography>
                          )}
                        </Box>
                      );
                    })}
                  </Box>
                );
              })()}
            </Box>
          </Box>
        )}

        <Box
          sx={{
            position: 'absolute',
            left: 32,
            bottom: 32,
            zIndex: 1100,
            backgroundColor: 'rgba(18, 18, 18, 0.8)',
            borderRadius: 2,
            padding: '16px',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.35)',
            backdropFilter: 'blur(6px)',
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
            width: '430px',
            color: '#fff',
          }}
        >
          {/* First Part: Selected Rosbag Section */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              Selected Rosbag
            </Typography>
            {selectedRosbag ? (
              <>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '8pt', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                    {selectedRosbag.name}
                  </Typography>
                  {selectedRosbag.timestamp ? (
                    <Typography variant="body2" sx={{ opacity: 0.7, fontSize: '8pt', whiteSpace: 'nowrap', marginLeft: 'auto' }}>
                      {new Intl.DateTimeFormat(undefined, { dateStyle: 'short', timeStyle: 'short' }).format(new Date(selectedRosbag.timestamp))}
                    </Typography>
                  ) : (
                    <Typography variant="body2" sx={{ opacity: 0.5, fontStyle: 'italic', whiteSpace: 'nowrap', marginLeft: 'auto' }}>
                      Timestamp unavailable
                    </Typography>
                  )}
                </Box>
              </>
            ) : (
              <>
                <Typography variant="body2" sx={{ opacity: 0.5, fontStyle: 'italic' }}>
                  No rosbag selected
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.5, fontStyle: 'italic' }}>
                  Timestamp unavailable
                </Typography>
              </>
            )}
          </Box>

          {/* Separator */}
          <Box sx={{ borderTop: '1px solid rgba(255, 255, 255, 0.1)' }} />

          {/* Display Options Section */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              Options
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, width: '100%' }}>
              {/* Show all visited locations toggle - half width */}
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 1,
                  px: 1.5,
                  py: 0.75,
                  borderRadius: 1,
                  backgroundColor: 'rgba(30, 30, 30, 0.6)',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  flex: 1,
                }}
              >
                <Typography variant="body2" sx={{ fontSize: '8pt', fontWeight: 500, whiteSpace: 'nowrap' }}>
                  Show all visited locations
                </Typography>
                <Switch
                  checked={showAllRosbags}
                  onChange={(event) => {
                    const checked = event.target.checked;
                    setShowAllRosbags(checked);
                    if (checked) {
                      fetchAllPoints();
                    }
                  }}
                  color="primary"
                  disabled={loadingAllPoints}
                  size="small"
                />
              </Box>
              {/* Show mcaps toggle - half width */}
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 1,
                  px: 1.5,
                  py: 0.75,
                  borderRadius: 1,
                  backgroundColor: 'rgba(30, 30, 30, 0.6)',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  flex: 1,
                }}
              >
                <Typography variant="body2" sx={{ fontSize: '8pt', fontWeight: 500, whiteSpace: 'nowrap' }}>
                  Show individual mcap files
                </Typography>
                <Switch
                  checked={showMcaps}
                  onChange={(event) => {
                    setShowMcaps(event.target.checked);
                  }}
                  color="primary"
                  disabled={loadingMcaps}
                  size="small"
                />
              </Box>
            </Box>
          </Box>

          {/* Separator */}
          <Box sx={{ borderTop: '1px solid rgba(255, 255, 255, 0.1)' }} />

          {/* Second Part: Polygons Section */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              Area Filter
            </Typography>
            
            {/* Count */}
            {/*}
            <Box>
              <Typography variant="body2" sx={{ opacity: 0.8 }}>
                count: {polygons.length}
              </Typography>
            </Box>
            */}

            {/* Filter: Import Polygons, Offset, Apply to Search, Clear All */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {/* Polygon Popper Menu */}
              <Popper
                open={showPolygonPopper}
                anchorEl={polygonButtonRef.current}
                placement="top-start"
                sx={{ zIndex: 10000, width: '300px' }}
              >
                <Paper sx={{ padding: '8px', background: '#202020', borderRadius: '8px', maxHeight: '400px', overflowY: 'auto' }}>
                  {/* RESET button */}
                  <Button
                    onClick={() => {
                      setPolygons([]);
                      setActivePolygonId(null);
                      setSelectedPolygonId(null);
                      setSelectedPolygonFile('');
                      try {
                        sessionStorage.removeItem('__BagSeekPositionalFilter');
                        sessionStorage.removeItem('__BagSeekPositionalPolygons');
                      } catch (e) {
                        console.error('Failed to clear positional filter:', e);
                      }
                      setShowPolygonPopper(false);
                    }}
                    variant="contained"
                    color="primary"
                    fullWidth
                    sx={{
                      fontSize: '0.8rem',
                      marginBottom: '8px',
                      textTransform: 'none',
                    }}
                  >
                    RESET
                  </Button>

                  {/* Polygon files list */}
                  {loadingPolygonFiles ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', padding: '8px' }}>
                      <CircularProgress size={20} />
                    </Box>
                  ) : polygonFiles.length === 0 ? (
                    <Typography sx={{ fontSize: '0.8rem', color: '#888', textAlign: 'center', padding: '8px' }}>
                      No saved polygons
                    </Typography>
                  ) : (
                    polygonFiles.map((filename) => {
                      const isProtected = filename.startsWith('Lehr- und Forsch');
                      return (
                        <MenuItem
                          key={filename}
                          sx={{
                            fontSize: '0.8rem',
                            padding: '4px 8px',
                            display: 'flex',
                            alignItems: 'center',
                            color: selectedPolygonFile === filename ? '#2196f3' : 'white',
                            backgroundColor: selectedPolygonFile === filename ? 'rgba(33, 150, 243, 0.1)' : 'transparent',
                            '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.1)' },
                          }}
                          onClick={() => {
                            setSelectedPolygonFile(filename);
                            handleImportPolygon(filename);
                            setShowPolygonPopper(false);
                          }}
                          disabled={importingPolygon || deletingPolygon === filename}
                        >
                          <Box sx={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {filename.replace('.json', '')}
                          </Box>
                          <IconButton
                            size="small"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeletePolygonFile(filename);
                            }}
                            disabled={isProtected || deletingPolygon === filename}
                            sx={{ color: '#888', '&:hover': { color: isProtected ? '#888' : '#ff5555' }, marginLeft: '4px' }}
                          >
                            {deletingPolygon === filename ? (
                              <CircularProgress size={16} />
                            ) : (
                              <DeleteIcon fontSize="small" />
                            )}
                          </IconButton>
                        </MenuItem>
                      );
                    })
                  )}

                  {/* Save section */}
                  {showPolygonSaveField ? (
                    <TextField
                      autoFocus
                      fullWidth
                      size="small"
                      variant="outlined"
                      placeholder="Enter name..."
                      value={newPolygonName}
                      onChange={(e) => setNewPolygonName(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleSavePolygon(newPolygonName)}
                      onBlur={() => {
                        if (!newPolygonName.trim()) {
                          setShowPolygonSaveField(false);
                        }
                      }}
                      disabled={savingPolygon}
                      InputProps={{
                        endAdornment: savingPolygon ? (
                          <CircularProgress size={16} />
                        ) : (
                          <IconButton
                            size="small"
                            onClick={() => handleSavePolygon(newPolygonName)}
                            disabled={!newPolygonName.trim()}
                          >
                            <AddIcon fontSize="small" />
                          </IconButton>
                        ),
                      }}
                      sx={{
                        backgroundColor: '#303030',
                        borderRadius: '4px',
                        marginTop: '4px',
                        input: { color: 'white', fontSize: '0.8rem' },
                        '& .MuiOutlinedInput-root': {
                          '& fieldset': { borderColor: '#555' },
                          '&:hover fieldset': { borderColor: '#777' },
                          '&.Mui-focused fieldset': { borderColor: '#aaa' },
                        },
                      }}
                    />
                  ) : (
                    <MenuItem
                      onClick={() => setShowPolygonSaveField(true)}
                      disabled={polygons.filter((p) => p.isClosed).length === 0}
                      sx={{ fontSize: '0.8rem', padding: '4px 8px', display: 'flex', justifyContent: 'center' }}
                    >
                      <AddIcon fontSize="small" />
                    </MenuItem>
                  )}
                </Paper>
              </Popper>

              {/* First row: Button and TextField */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                <Button
                  ref={polygonButtonRef}
                  onClick={() => setShowPolygonPopper(!showPolygonPopper)}
                  variant="outlined"
                  size="small"
                  disabled={loadingPolygonFiles}
                  sx={{
                    fontSize: '8pt',
                    flex: 1,
                    backgroundColor: 'rgba(30, 30, 30, 0.8)',
                    color: '#ffffff',
                    borderColor: 'rgba(255, 255, 255, 0.23)',
                    textTransform: 'none',
                    justifyContent: 'flex-start',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    '&:hover': {
                      borderColor: 'rgba(255, 255, 255, 0.4)',
                      backgroundColor: 'rgba(40, 40, 40, 0.8)',
                    },
                  }}
                >
                  {selectedPolygonFile ? selectedPolygonFile.replace('.json', '') : 'Load/Save Polygons'}
                </Button>
                <TextField
                  type="number"
                  value={offsetDistance}
                  onChange={(e) => {
                    const value = parseFloat(e.target.value) || 0;
                    setOffsetDistance(value);
                  }}
                  size="small"
                  disabled={polygons.length === 0}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Typography sx={{ fontSize: '8pt', color: '#cccccc' }}>
                          Offset
                        </Typography>
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <InputAdornment position="end">
                        <Typography sx={{ fontSize: '8pt', color: '#cccccc' }}>
                          m
                        </Typography>
                      </InputAdornment>
                    ),
                  }}
                  inputProps={{
                    step: 1,
                    min: -1000,
                    max: 1000,
                    style: { color: '#ffffff', fontSize: '8pt' },
                  }}
                  sx={{
                    flex: 1,
                    '& .MuiOutlinedInput-root': {
                      backgroundColor: 'rgba(30, 30, 30, 0.8)',
                      borderRadius: 1,
                      '& fieldset': {
                        borderColor: 'rgba(255, 255, 255, 0.23)',
                      },
                      '&:hover fieldset': {
                        borderColor: 'rgba(255, 255, 255, 0.4)',
                      },
                      '&.Mui-focused fieldset': {
                        borderColor: '#2196f3',
                      },
                    },
                  }}
                />
              </Box>
              {/* Second row: Buttons with equal spacing */}
              <Box sx={{ display: 'flex', gap: 1, width: '100%' }}>
                <Button
                  size="small"
                  variant="contained"
                  color="primary"
                  disabled={polygons.filter((p) => p.isClosed).length === 0 || applyingToSearch || importingPolygon || isRestoringPolygons}
                onClick={async () => {
                  setApplyingToSearch(true);
                  try {
                  // Get rosbags that overlap with closed polygons (using offset if applicable)
                  const closedPolygons = polygons.filter((p) => p.isClosed);
                  const overlappingRosbags: string[] = [];
                  const mcapFilterMap: Record<string, string[]> = {};

                  console.log('\t\t[MCAP-DEBUG] === Apply to Search clicked ===');
                  console.log('\t\t[MCAP-DEBUG] Total rosbags to check:', rosbags.length);
                  console.log('\t\t[MCAP-DEBUG] Closed polygons:', closedPolygons.length);

                  // Re-check overlaps and collect per-rosbag MCAP IDs
                  // Use boundaries for fast pre-filtering
                  const hasBoundaries = rosbagBoundaries !== null && Object.keys(rosbagBoundaries).length > 0;
                  for (const rosbag of rosbags) {
                    // Pre-filter with concave hull boundary
                    if (hasBoundaries) {
                      const boundary = rosbagBoundaries[rosbag.name];
                      if (boundary && !boundaryOverlapsPolygons(boundary, closedPolygons, offsetDistance)) {
                        continue; // Hull doesn't overlap — skip
                      }
                    }

                    const overlaps = await checkRosbagOverlap(rosbag.name, closedPolygons, offsetDistance);
                    if (overlaps) {
                      overlappingRosbags.push(rosbag.name);
                      const mcapIds = await getMcapsInsidePolygons(rosbag.name, closedPolygons, offsetDistance);
                      if (mcapIds.size > 0) {
                        mcapFilterMap[rosbag.name] = Array.from(mcapIds);
                        console.log(`\t\t[MCAP-DEBUG] Rosbag "${rosbag.name}": ${mcapIds.size} MCAPs inside polygons -> [${Array.from(mcapIds).join(', ')}]`);
                      } else {
                        console.log(`\t\t[MCAP-DEBUG] Rosbag "${rosbag.name}": overlaps but 0 MCAPs inside`);
                      }
                    }
                  }

                  console.log(`\t\t[MCAP-DEBUG] overlappingRosbags (${overlappingRosbags.length}):`, overlappingRosbags);
                  console.log(`\t\t[MCAP-DEBUG] mcapFilterMap keys (${Object.keys(mcapFilterMap).length}):`, Object.keys(mcapFilterMap));

                  // Store filtered rosbags, MCAP IDs, and navigation flag in sessionStorage.
                  // Critical data first — polygon backup last (largest payload, non-essential for search).
                  try {
                    sessionStorage.setItem('__BagSeekPositionalFilter', JSON.stringify(overlappingRosbags));
                    if (Object.keys(mcapFilterMap).length > 0) {
                      sessionStorage.setItem('__BagSeekPositionalMcapFilter', JSON.stringify(mcapFilterMap));
                      console.log('\t\t[MCAP-DEBUG] Stored mcapFilterMap in sessionStorage');
                    } else {
                      sessionStorage.removeItem('__BagSeekPositionalMcapFilter');
                      console.log('\t\t[MCAP-DEBUG] No mcapFilterMap to store (empty)');
                    }
                    sessionStorage.setItem('__BagSeekApplyToSearchJustNavigated', '1');
                    window.dispatchEvent(new Event('__BagSeekPositionalFilterChanged'));
                  } catch (e) {
                    console.error('Failed to store positional filter:', e);
                  }
                  // Polygon backup stored separately — not needed for search, only for restoring MAP view
                  try {
                    sessionStorage.setItem('__BagSeekPositionalPolygons', JSON.stringify(polygons));
                  } catch (e) {
                    console.warn('Could not persist polygon data (storage quota may be full):', e);
                  }

                  console.log('\t\t[MCAP-DEBUG] Navigating to /search');
                  // Navigate to GlobalSearch
                  navigate('/search');
                  } finally {
                    setApplyingToSearch(false);
                  }
                }}
                sx={{ fontSize: '8pt', py: 0.25, px: 1, flex: 1, borderRadius: 1 }}
              >
                {applyingToSearch ? 'Computing...' : isRestoringPolygons ? 'Restoring...' : 'Apply to Search'}
              </Button>
                <Button
                  size="small"
                  variant="outlined"
                  disabled={polygons.length === 0}
                  onClick={() => {
                    setPolygons([]);
                    setActivePolygonId(null);
                    setSelectedPolygonId(null);
                    setSelectedPolygonFile(''); // Reset the selected file so it can be reselected
                    // Clear positional filter, MCAP filter, and polygons from sessionStorage
                    try {
                      sessionStorage.removeItem('__BagSeekPositionalFilter');
                      sessionStorage.removeItem('__BagSeekPositionalPolygons');
                      sessionStorage.removeItem('__BagSeekPositionalMcapFilter');
                      sessionStorage.removeItem('__BagSeekMapMcapFilter');
                    } catch (e) {
                      console.error('Failed to clear positional filter:', e);
                    }
                  }}
                  sx={{ fontSize: '8pt', py: 0.25, px: 1, flex: 1, borderRadius: 1 }}
                >
                  Clear All
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => {
                    handleExportList();
                  }}
                  sx={{ fontSize: '8pt', py: 0.25, px: 1, flex: 1, borderRadius: 1 }}
                >
                  Export List
                </Button>

              </Box>
            </Box>
          </Box>
        </Box>

        <Tooltip title={useSatellite ? 'Switch to map view' : 'Switch to satellite view'} placement="left">
          <IconButton
            onClick={() => setUseSatellite((prev) => !prev)}
            color="primary"
            sx={{
              position: 'absolute',
              bottom: 32,
              right: 32,
              zIndex: 1100,
              backgroundColor: 'rgba(18, 18, 18, 0.8)',
              borderRadius: 2,
              boxShadow: '0 10px 24px rgba(0, 0, 0, 0.4)',
              width: 52,
              height: 52,
              '&:hover': {
                backgroundColor: 'rgba(30, 30, 30, 0.95)',
              },
            }}
            size="large"
          >
            {useSatellite ? <MapIcon fontSize="inherit" /> : <SatelliteAltIcon fontSize="inherit" />}
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  );
};

export default PositionalOverview;
