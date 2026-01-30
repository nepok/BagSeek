import { Alert, Box, Button, CircularProgress, IconButton, InputAdornment, MenuItem, Select, Slider, TextField, Tooltip, Typography } from '@mui/material';
import MapIcon from '@mui/icons-material/Map';
import SatelliteAltIcon from '@mui/icons-material/SatelliteAlt';
import Switch from '@mui/material/Switch';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.heat';
import './PositionalOverview.css';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { inflatePathsD, PathsD, PathD, JoinType, EndType } from 'clipper2-ts';

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
  const [checkingOverlap, setCheckingOverlap] = useState<boolean>(false);
  const [polygonFiles, setPolygonFiles] = useState<string[]>([]);
  const [selectedPolygonFile, setSelectedPolygonFile] = useState<string>('');
  const [loadingPolygonFiles, setLoadingPolygonFiles] = useState<boolean>(false);
  const [importingPolygon, setImportingPolygon] = useState<boolean>(false);
  const [exportingList, setExportingList] = useState<boolean>(false);
  const [offsetDistance, setOffsetDistance] = useState<number>(0);
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
          setSelectedIndex(meta.length ? meta.length - 1 : 0);
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

  // Restore polygons from sessionStorage on mount
  useEffect(() => {
    try {
      const cachedPolygons = sessionStorage.getItem('__BagSeekPositionalPolygons');
      if (cachedPolygons) {
        const parsed = JSON.parse(cachedPolygons);
        if (Array.isArray(parsed) && parsed.length > 0) {
          isRestoringRef.current = true;
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

  useEffect(() => {
    if (!showMcaps || !rosbags.length) {
      setMcapLocationPoints([]);
      setAvailableMcaps([]);
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

        if (!cancelled) {
          const newPoints: McapLocationPoint[] = Array.isArray(pointsData.points) ? pointsData.points : [];
          const newMcaps: McapInfo[] = Array.isArray(listData.mcaps) ? listData.mcaps : [];
          setMcapLocationPoints(newPoints);
          setAvailableMcaps(newMcaps);
          setSelectedMcapIndex(0);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : 'Failed to load mcap data');
          setMcapLocationPoints([]);
          setAvailableMcaps([]);
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

    let cancelled = false;
    setCheckingOverlap(true);
    
    // Check overlap for all rosbags
    const checkAllOverlaps = async () => {
      const overlapMap = new Map<string, boolean>();
      
      // Check each rosbag
      for (const rosbag of rosbags) {
        if (cancelled) break;
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
            // Dispatch custom event for same-tab updates (storage event only fires in other tabs)
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
  }, [polygons, rosbags, offsetDistance]);

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
              bottom: 74,
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
                backgroundColor: 'rgba(18, 18, 18, 0.9)',
                borderRadius: 999,
                boxShadow: '0 10px 30px rgba(0, 0, 0, 0.35)',
                px: 4,
                py: 0.3,
                backdropFilter: 'blur(8px)',
                display: 'flex',
                alignItems: 'center',
                position: 'relative',
                pointerEvents: 'auto', // Enable pointer events for this container
              }}
            >
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
                backgroundColor: 'rgba(18, 18, 18, 0.9)',
                borderRadius: 999,
                boxShadow: '0 10px 30px rgba(0, 0, 0, 0.35)',
                px: 4,
                py: 0.3,
                backdropFilter: 'blur(8px)',
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
                    left: '16px',
                    right: '16px',
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
              {/* Show all toggle - half width */}
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
                  Show all
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
                  Show mcaps
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
              Area Restrictions
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
              {/* First row: Select and TextField */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                <Select
                  value={selectedPolygonFile}
                  onChange={(e) => {
                    const filename = e.target.value as string;
                    setSelectedPolygonFile(filename);
                    if (filename) {
                      handleImportPolygon(filename);
                    } else {
                      // Clear polygons when "None" is selected (same as Clear All)
                      setPolygons([]);
                      setActivePolygonId(null);
                      setSelectedPolygonId(null);
                      // Clear positional filter and polygons from sessionStorage
                      try {
                        sessionStorage.removeItem('__BagSeekPositionalFilter');
                        sessionStorage.removeItem('__BagSeekPositionalPolygons');
                      } catch (e) {
                        console.error('Failed to clear positional filter:', e);
                      }
                    }
                  }}
                  displayEmpty
                  size="small"
                  disabled={loadingPolygonFiles || importingPolygon || polygonFiles.length === 0}
                  sx={{
                    fontSize: '8pt',
                    flex: 1,
                    backgroundColor: 'rgba(30, 30, 30, 0.8)',
                    color: '#ffffff',
                    overflow: 'hidden',
                    borderRadius: 1,
                    '& .MuiSelect-select': {
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    },
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(255, 255, 255, 0.23)',
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(255, 255, 255, 0.4)',
                    },
                    '& .MuiSvgIcon-root': {
                      color: '#ffffff',
                    },
                  }}
                >
                  <MenuItem value="">
                    None
                  </MenuItem>
                  {polygonFiles.map((filename) => (
                    <MenuItem key={filename} value={filename}>
                      {filename.replace('.json', '')}
                    </MenuItem>
                  ))}
                </Select>
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
                  disabled={polygons.filter((p) => p.isClosed).length === 0}
                onClick={async () => {
                  // Get rosbags that overlap with closed polygons (using offset if applicable)
                  const closedPolygons = polygons.filter((p) => p.isClosed);
                  const overlappingRosbags: string[] = [];
                  
                  // Re-check overlaps with current offset distance
                  for (const rosbag of rosbags) {
                    const overlaps = await checkRosbagOverlap(rosbag.name, closedPolygons, offsetDistance);
                    if (overlaps) {
                      overlappingRosbags.push(rosbag.name);
                    }
                  }
                  
                  // Store filtered rosbags and polygons in sessionStorage
                  try {
                    sessionStorage.setItem('__BagSeekPositionalFilter', JSON.stringify(overlappingRosbags));
                    sessionStorage.setItem('__BagSeekPositionalPolygons', JSON.stringify(polygons));
                    // Dispatch custom event for same-tab updates
                    window.dispatchEvent(new Event('__BagSeekPositionalFilterChanged'));
                  } catch (e) {
                    console.error('Failed to store positional filter:', e);
                  }
                  
                  // Navigate to GlobalSearch
                  navigate('/search');
                }}
                sx={{ fontSize: '8pt', py: 0.25, px: 1, flex: 1, borderRadius: 1 }}
              >
                Apply to Search
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
                    // Clear positional filter and polygons from sessionStorage
                    try {
                      sessionStorage.removeItem('__BagSeekPositionalFilter');
                      sessionStorage.removeItem('__BagSeekPositionalPolygons');
                    } catch (e) {
                      console.error('Failed to clear positional filter:', e);
                    }
                  }}
                  sx={{ fontSize: '8pt', py: 0.25, px: 1, flex: 1, borderRadius: 1 }}
                >
                  Clear All
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
