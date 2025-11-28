import { Alert, Box, Button, CircularProgress, IconButton, Slider, Tooltip, Typography } from '@mui/material';
import MapIcon from '@mui/icons-material/Map';
import SatelliteAltIcon from '@mui/icons-material/SatelliteAlt';
import Switch from '@mui/material/Switch';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet.heat';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

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

const TILE_LAYER_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const TILE_LAYER_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

const ROSBAG_TIMESTAMP_REGEX = /^rosbag2_(\d{4})_(\d{2})_(\d{2})-(\d{2})_(\d{2})_(\d{2})(?:_short)?$/;

const parseRosbagTimestamp = (name: string): number | null => {
  const match = ROSBAG_TIMESTAMP_REGEX.exec(name.trim());
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
      yi > point.lat !== yj > point.lat &&
      point.lon < ((xj - xi) * (point.lat - yi)) / (yj - yi) + xi;
    if (intersect) {
      inside = !inside;
    }
  }

  return inside;
};

// Check if any point from rosbag overlaps with any closed polygon
const checkRosbagOverlap = async (
  rosbagName: string,
  closedPolygons: Polygon[]
): Promise<boolean> => {
  if (closedPolygons.length === 0) {
    return true; // No polygons means all overlap
  }

  try {
    const response = await fetch(`/api/gps/rosbags/${encodeURIComponent(rosbagName)}`);
    if (!response.ok) {
      return false; // If we can't fetch data, assume no overlap
    }
    const data = await response.json();
    const rosbagPoints: RosbagPoint[] = Array.isArray(data.points) ? data.points : [];

    // Check if any point from rosbag is inside any closed polygon
    for (const point of rosbagPoints) {
      for (const polygon of closedPolygons) {
        if (polygon.isClosed && pointInPolygon({ lat: point.lat, lon: point.lon }, polygon.points)) {
          return true; // Found overlap
        }
      }
    }

    return false; // No overlap found
  } catch {
    return false;
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
  const [error, setError] = useState<string | null>(null);
  const [useSatellite, setUseSatellite] = useState<boolean>(false);
  const [polygons, setPolygons] = useState<Polygon[]>([]);
  const [activePolygonId, setActivePolygonId] = useState<string | null>(null);
  const [selectedPolygonId, setSelectedPolygonId] = useState<string | null>(null);
  const [rosbagOverlapStatus, setRosbagOverlapStatus] = useState<Map<string, boolean>>(new Map());
  const [checkingOverlap, setCheckingOverlap] = useState<boolean>(false);
  const prevPolygonCountRef = useRef<number>(0);

  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const baseLayersRef = useRef<{ map: L.TileLayer | null; satellite: L.TileLayer | null }>({ map: null, satellite: null });
  const heatLayerRef = useRef<L.Layer | null>(null);
  const allHeatLayerRef = useRef<L.Layer | null>(null);
  const polygonLayersRef = useRef<Map<string, { markers: L.Marker[]; polyline: L.Polyline | null; polygon: L.Polygon | null }>>(new Map());
  const isRestoringRef = useRef(false);

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
        const response = await fetch('/api/gps/rosbags');
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

        const response = await fetch(`/api/gps/rosbags/${encodeURIComponent(selected.name)}`);
        if (!response.ok) {
          if (response.status === 404) {
            throw new Error(`No GPS data found for ${selected.name}`);
          }
          throw new Error(`Failed to load GPS data (${response.status})`);
        }
        const data = await response.json();
        if (!cancelled) {
          const newPoints: RosbagPoint[] = Array.isArray(data.points) ? data.points : [];
          setPoints(newPoints);
        }
      } catch (fetchError) {
        if (!cancelled) {
          setError(fetchError instanceof Error ? fetchError.message : 'Failed to load GPS data');
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

    // Right-click handler for creating polygon points
    const handleContextMenu = (e: L.LeafletMouseEvent) => {
      e.originalEvent.preventDefault();
      
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
    
    // Click handler to deselect polygon when clicking on map (but not on polygons)
    const handleMapClick = (e: L.LeafletMouseEvent) => {
      const target = e.originalEvent.target as HTMLElement;
      
      // Don't deselect if clicking on a polygon layer
      if (target && (target.tagName === 'path' || target.tagName === 'polyline' || target.closest('.leaflet-interactive'))) {
        return;
      }
      
      setSelectedPolygonId(null);
    };
    map.on('click', handleMapClick);

    return () => {
      map.off('contextmenu', handleContextMenu);
      map.off('click', handleMapClick);
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
        const allLayer = allLayerFactory(allHeatmapData, {
          minOpacity: 0.08,
          maxZoom: 18,
          radius: 20,
          blur: 25,
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
      const heatLayer = heatLayerFactory(heatmapData, {
        minOpacity: 0.2,
        maxZoom: 18,
        radius: 15,
        blur: 15,
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

    const latLngs: L.LatLngTuple[] = points.map((point) => [point.lat, point.lon]);

    if (latLngs.length === 1) {
      map.setView(latLngs[0], 15);
    } else {
      map.fitBounds(L.latLngBounds(latLngs), { padding: [32, 32] });
    }

    map.invalidateSize();
    
  }, [points, showAllRosbags, allPoints]);

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

      // Create markers for each point
      polygon.points.forEach((point, index) => {
        const isFirstPoint = index === 0;
        const isLastPoint = index === polygon.points.length - 1;
        
        // Create custom icon
        const icon = L.divIcon({
          className: 'custom-polygon-marker',
          html: `<div style="
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background-color: ${isFirstPoint ? '#4caf50' : isLastPoint && isActive ? '#ff9800' : '#2196f3'};
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            cursor: move;
          "></div>`,
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        });

        const marker = L.marker([point.lat, point.lon], { 
          icon,
          draggable: true,
        });
        
        // Add click handler to close polygon if clicking on any existing point
        if (isActive) {
          marker.on('click', () => {
            if (polygon.points.length >= 3) {
              setPolygons((prev) =>
                prev.map((p) =>
                  p.id === polygon.id ? { ...p, isClosed: true } : p
                )
              );
              setActivePolygonId(null);
            }
          });
        }

        // Add dragend handler to update polygon point position
        marker.on('dragend', () => {
          const newLatLng = marker.getLatLng();
          setPolygons((prev) =>
            prev.map((p) =>
              p.id === polygon.id
                ? {
                    ...p,
                    points: p.points.map((pt, idx) =>
                      idx === index
                        ? { ...pt, lat: newLatLng.lat, lon: newLatLng.lng }
                        : pt
                    ),
                  }
                : p
            )
          );
        });

        marker.addTo(map);
        markers.push(marker);
      });

      // Draw polyline connecting points
      const latLngs = polygon.points.map((p) => [p.lat, p.lon] as L.LatLngTuple);
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
        
        polygonFill.addTo(map);
      }

      polygonLayersRef.current.set(polygon.id, {
        markers,
        polyline,
        polygon: polygonFill,
      });
    });
  }, [polygons, activePolygonId, selectedPolygonId]);

  // Keyboard handler for deleting selected polygon
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle backspace, delete, or entf if a polygon is selected and we're not in an input field
      const activeElement = document.activeElement;
      const isInputField = activeElement?.tagName === 'INPUT' || activeElement?.tagName === 'TEXTAREA' || (activeElement as HTMLElement)?.isContentEditable;
      
      if ((e.key === 'Backspace' || e.key === 'Delete' || e.key === 'Entf') && selectedPolygonId && !isInputField) {
        e.preventDefault();
        e.stopPropagation();
        setPolygons((prev) => prev.filter((p) => p.id !== selectedPolygonId));
        if (activePolygonId === selectedPolygonId) {
          setActivePolygonId(null);
        }
        setSelectedPolygonId(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedPolygonId, activePolygonId]);

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
        const overlaps = await checkRosbagOverlap(rosbag.name, closedPolygons);
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
  }, [polygons, rosbags]);

  const fetchAllPoints = useCallback(async () => {
    if (loadingAllPoints || allPointsLoaded) {
      return;
    }

    setLoadingAllPoints(true);
    setError(null);

    try {
      const response = await fetch('/api/gps/all');
      if (!response.ok) {
        throw new Error(`Failed to load aggregated GPS data (${response.status})`);
      }
      const data = await response.json();
      const aggregatedPoints: RosbagPoint[] = Array.isArray(data.points) ? data.points : [];
      setAllPoints(aggregatedPoints);
      setAllPointsLoaded(true);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : 'Failed to load aggregated GPS data');
      setShowAllRosbags(false);
    } finally {
      setLoadingAllPoints(false);
    }
  }, [loadingAllPoints, allPointsLoaded]);

  const selectedRosbag = rosbags[selectedIndex] ?? null;

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#121212',
        color: '#ffffff',
        fontFamily: 'inherit',
      }}
    >
      <Box
        component="main"
        sx={{
          flex: 1,
          minHeight: 0,
          position: 'relative',
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
              <Typography variant="body2">Loading GPS data…</Typography>
            </Box>
          )}
          {error && (
            <Alert severity="error" sx={{ backgroundColor: '#1f1f1f', color: '#ffb4a2' }}>
              {error}
            </Alert>
          )}
          
          {!loadingRosbags && !rosbags.length && !error && (
            <Alert severity="info" sx={{ backgroundColor: '#1f1f1f', color: '#bbdefb' }}>
              No rosbags available. Generate the GPS lookup table to see data here.
            </Alert>
          )}
        </Box>

        {rosbags.length > 1 && (
          <Box
            sx={{
              position: 'absolute',
              left: '50%',
              bottom: 72,
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
                py: 1.5,
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
                    width: 18,
                    height: 18,
                    boxShadow: (theme) => `0 0 0 8px ${theme.palette.primary.main}40`,
                    '&:hover, &.Mui-focusVisible': {
                      boxShadow: (theme) => `0 0 0 12px ${theme.palette.primary.main}55`,
                    },
                  },
                  '& .MuiSlider-rail': {
                    opacity: 0.2,
                  },
                  '& .MuiSlider-valueLabel': {
                    background: (theme) => theme.palette.background.paper,
                    borderRadius: 5,
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    color: (theme) => theme.palette.text.primary,
                  },
                }}
              />
            </Box>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                padding: '10px 18px',
                borderRadius: 999,
                backgroundColor: 'rgba(18, 18, 18, 0.92)',
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.35)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                backdropFilter: 'blur(8px)',
                pointerEvents: 'auto', // Enable pointer events for this container
              }}
            >
              <Typography variant="body2" sx={{ fontSize: '0.85rem', fontWeight: 500 }}>
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
          </Box>
        )}

        {selectedRosbag && (
          <Box
            sx={{
              position: 'absolute',
              left: 32,
              bottom: 88,
              backgroundColor: 'rgba(18, 18, 18, 0.92)',
              borderRadius: 5,
              padding: '12px 16px',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.35)',
              backdropFilter: 'blur(6px)',
              maxWidth: '70%',
              zIndex: 1100,
              color: '#fff',
            }}
          >
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              Selected Rosbag
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
              {selectedRosbag.name}
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.7 }}>
              {formatTimestampLabel(selectedRosbag.timestamp, 'Timestamp unavailable')}
            </Typography>
            {showAllRosbags && (
              <Typography variant="body2" sx={{ opacity: 0.7 }}>
                Overlay: All rosbags (yellow)
              </Typography>
            )}
            {showAllRosbags && loadingAllPoints && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                <CircularProgress size={14} color="inherit" />
                <Typography variant="caption">Loading overlay…</Typography>
              </Box>
            )}
            {!loadingAllPoints && !showAllRosbags && allPointsLoaded && (
              <Typography variant="caption" sx={{ opacity: 0.6 }}>
                Overlay data cached
              </Typography>
            )}
            {polygons.length > 0 && (
              <Box sx={{ mt: 1, pt: 1, borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
                <Typography variant="caption" sx={{ opacity: 0.8 }}>
                  Polygons: {polygons.length}
                  {polygons.filter((p) => !p.isClosed).length > 0 && (
                    <span> ({polygons.filter((p) => !p.isClosed).length} active)</span>
                  )}
                </Typography>
                <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5, flexWrap: 'wrap' }}>
                  {polygons.filter((p) => p.isClosed).length > 0 && (
                    <Button
                      size="small"
                      variant="contained"
                      color="primary"
                      onClick={() => {
                        // Get rosbags that overlap with closed polygons
                        const closedPolygons = polygons.filter((p) => p.isClosed);
                        const overlappingRosbags: string[] = [];
                        
                        rosbags.forEach((rosbag) => {
                          const overlaps = rosbagOverlapStatus.get(rosbag.name) ?? false;
                          if (overlaps) {
                            overlappingRosbags.push(rosbag.name);
                          }
                        });
                        
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
                      sx={{ fontSize: '0.7rem', py: 0.25, px: 1 }}
                    >
                      Apply to Search
                    </Button>
                  )}
                  {polygons.length > 0 && (
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => {
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
                      }}
                      sx={{ fontSize: '0.7rem', py: 0.25, px: 1 }}
                    >
                      Clear All
                    </Button>
                  )}
                </Box>
              </Box>
            )}
          </Box>
        )}

        <Tooltip title={useSatellite ? 'Switch to map view' : 'Switch to satellite view'} placement="left">
          <IconButton
            onClick={() => setUseSatellite((prev) => !prev)}
            color="primary"
            sx={{
              position: 'absolute',
              bottom: 84,
              right: 32,
              zIndex: 1100,
              backgroundColor: 'rgba(18, 18, 18, 0.92)',
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
