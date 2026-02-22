import React, { useState, useEffect, useRef } from "react";
import "./NodeContent.css"; // Import the CSS file
import { Box, Typography, CircularProgress } from "@mui/material";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import L from "leaflet";
import "leaflet/dist/leaflet.css"; // Ensure Leaflet CSS is loaded
import 'leaflet.heat';

interface NodeContentProps {
  nodeTopic: string | null; 
  nodeTopicType: string | null; // Type of the topic, e.g., "sensor_msgs/Image"
  selectedRosbag: string | null;
  mappedTimestamp: number | null; // Optional timestamp for image reference
  mcapIdentifier: string | null;
}

const NodeContent: React.FC<NodeContentProps> = ({ nodeTopic, nodeTopicType, selectedRosbag, mappedTimestamp, mcapIdentifier }) => {

  console.log('NodeContent props:', { nodeTopic, nodeTopicType, selectedRosbag, mappedTimestamp, mcapIdentifier });

  const [text, setText] = useState<string | null>(null);   // fetched textual message (e.g. string payloads)
  const [imageUrl, setImageUrl] = useState<string | null>(null); // image data URL or blob URL
  const [pointCloud, setPointCloud] = useState<{ positions: number[]; colors: number[] } | null>(null); // 3D point cloud data from sensor with optional colors
  const [realTimestamp, setRealTimestamp] = useState<string | null>(null); // human-readable timestamp string
  const [position, setPosition] = useState<{ latitude: number; longitude: number; altitude: number} | null>(null); // GPS position (lat, lon, alt)
  const [gpsPath, setGpsPath] = useState<[number, number][]>([]); // array of previous GPS coordinates for path drawing
  const [imuData, setImuData] = useState<{
    orientation: { x: number; y: number; z: number; w: number };
    angular_velocity: { x: number; y: number; z: number };
    linear_acceleration: { x: number; y: number; z: number };
  } | null>(null); // orientation, angular velocity, and acceleration from IMU
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [positionIsZero, setPositionIsZero] = useState<boolean>(false);

  const [heatmapPoints, setHeatmapPoints] = useState<{ lat: number; lon: number; count: number }[]>([]);

  const mapRef = useRef<L.Map | null>(null); // reference to the Leaflet map instance
  const mapContainerRef = useRef<HTMLDivElement | null>(null); // reference to the div container holding the map
  const heatLayerRef = useRef<L.Layer | null>(null); // reference to the leaflet.heat layer
  
  // Fetch data from API for selected topic/timestamp
  const fetchData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/content-mcap?rosbag=${selectedRosbag}&topic=${encodeURIComponent(nodeTopic!)}&mcap_identifier=${mcapIdentifier}&timestamp=${mappedTimestamp}`);
      const data = await response.json();

      if (data.error) {
        console.error("API error:", data.error);
        // Reset all states in case of error
        setText(null);
        setImageUrl(null);
        setPointCloud(null);
        setPosition(null);
        setImuData(null);
        setRealTimestamp(null);
        return;
      }

      // Clear all previous data
      setText(null);
      setImageUrl(null);
      setPointCloud(null);
      setPosition(null);
      setImuData(null);
      setRealTimestamp(null);
      setPositionIsZero(false);

      // Handle different response types
      if (data.timestamp) {
        // Convert timestamp to human-readable format
        const date = new Date(Number(data.timestamp) / 1000000); // Convert nanoseconds to milliseconds
        setRealTimestamp(date.toLocaleString());
      }

      // Handle image response
      if (data.type === 'image' && data.image) {
        const format = data.format || 'jpeg';
        setImageUrl(`data:image/${format};base64,${data.image}`);
      }
      // Handle point cloud response
      else if (data.type === 'pointCloud' && data.pointCloud) {
        setPointCloud({
          positions: data.pointCloud.positions || [],
          colors: data.pointCloud.colors || []
        });
      }
      // Handle GPS position response
      else if (data.type === 'position' && data.position) {
        const { latitude, longitude, altitude } = data.position;
        if (latitude === 0 && longitude === 0 && (altitude === 0 || altitude == null)) {
          setPositionIsZero(true);
        } else {
          setPosition({ latitude, longitude, altitude: altitude || 0 });
        }
      }
      // Handle IMU response
      else if (data.type === 'imu' && data.imu) {
        setImuData({
          orientation: data.imu.orientation || { x: 0, y: 0, z: 0, w: 1 },
          angular_velocity: data.imu.angular_velocity || { x: 0, y: 0, z: 0 },
          linear_acceleration: data.imu.linear_acceleration || { x: 0, y: 0, z: 0 }
        });
      }
      // Handle text response (fallback for unsupported types)
      else if (data.type === 'text' && data.text) {
        setText(data.text);
      }
      // Handle TF response (transform)
      else if (data.type === 'tf' && data.tf) {
        // For TF messages, we could display as text or visualize
        setText(`Transform:\nTranslation: (${data.tf.translation.x}, ${data.tf.translation.y}, ${data.tf.translation.z})\nRotation: (${data.tf.rotation.x}, ${data.tf.rotation.y}, ${data.tf.rotation.z}, ${data.tf.rotation.w})`);
      }
      // Handle odometry response
      else if (data.type === 'odometry' && data.odometry) {
        const odom = data.odometry;
        setImuData({
          orientation: odom.pose.orientation || { x: 0, y: 0, z: 0, w: 1 },
          angular_velocity: odom.twist.angular || { x: 0, y: 0, z: 0 },
          linear_acceleration: odom.twist.linear || { x: 0, y: 0, z: 0 } // Note: this is actually linear velocity, not acceleration
        });
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      // Reset all states in case of error
      setText(null);
      setImageUrl(null);
      setPointCloud(null);
      setPosition(null);
      setImuData(null);
      setRealTimestamp(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Trigger fetch when topic or timestamp changes
  useEffect(() => {
    if (nodeTopic && nodeTopicType && mappedTimestamp && mcapIdentifier) {
      fetchData();
    }
  }, [nodeTopic, nodeTopicType, mappedTimestamp, mcapIdentifier]);

  useEffect(() => {
    if (position) {
      setGpsPath(prevPath => [...prevPath, [position.latitude, position.longitude]]); // extend GPS path with current position
    }
  }, [position]);

  // Fetch rosbag heatmap data when a GPS topic is selected
  useEffect(() => {
    const isGps = nodeTopicType === "sensor_msgs/msg/NavSatFix" || nodeTopicType === "novatel_oem7_msgs/msg/BESTPOS";
    if (!isGps || !selectedRosbag) { setHeatmapPoints([]); return; }
    fetch(`/api/positions/rosbags/${encodeURIComponent(selectedRosbag)}`)
      .then(r => r.json())
      .then(data => setHeatmapPoints(Array.isArray(data.points) ? data.points : []))
      .catch(() => setHeatmapPoints([]));
  }, [nodeTopicType, selectedRosbag]);

  // Remove existing heat layer and add a fresh one from heatmapPoints
  const applyHeatmap = (map: L.Map) => {
    if (heatLayerRef.current) { map.removeLayer(heatLayerRef.current); heatLayerRef.current = null; }
    if (heatmapPoints.length === 0) return;
    const heatLayerFactory = (L as typeof L & { heatLayer?: (latlngs: [number, number, number?][], options?: Record<string, unknown>) => L.Layer }).heatLayer;
    if (!heatLayerFactory) return;
    const maxCount = Math.max(...heatmapPoints.map(p => p.count || 0));
    const data = heatmapPoints.map(p => [p.lat, p.lon, maxCount ? Math.max(p.count / maxCount, 0.05) : 0.1] as [number, number, number]);
    const layer = heatLayerFactory(data, {
      minOpacity: 0.2, maxZoom: 18, radius: 30, blur: 30,
      gradient: { 0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red' },
    });
    layer.addTo(map);
    heatLayerRef.current = layer;
    setTimeout(() => {
      map.getContainer().querySelectorAll('canvas.leaflet-heatmap-layer')
        .forEach(c => { (c as HTMLElement).style.pointerEvents = 'none'; });
    }, 0);
  };

  // Re-apply heatmap when data arrives (map may already be initialized)
  useEffect(() => {
    if (mapRef.current) applyHeatmap(mapRef.current);
  }, [heatmapPoints]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // Clear the map if switching away from GPS
    if ((nodeTopicType !== "sensor_msgs/msg/NavSatFix" && nodeTopicType !== "novatel_oem7_msgs/msg/BESTPOS") && mapRef.current) {
      mapRef.current.remove(); // remove old map if topic is not GPS
      mapRef.current = null;
      heatLayerRef.current = null;
    }

    // Initialize or update the map if GPS topic is selected
    if (position) {
      // Initialize the map ONLY if it doesn't exist or when switching back to GPS
      if (!mapRef.current && mapContainerRef.current) {
        mapRef.current = L.map(mapContainerRef.current).setView(
          [position.latitude, position.longitude], 16
        ); // create new map instance

        L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxZoom: 19,
          attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        }).addTo(mapRef.current!);
        applyHeatmap(mapRef.current!);
      }

      // Update the view to the new position but preserve current zoom level
      if (mapRef.current) {
        const currentZoom = mapRef.current.getZoom();
        mapRef.current.setView([position.latitude, position.longitude], currentZoom); // pan to new GPS position
        mapRef.current.invalidateSize();

        // Clear existing markers and polylines
        mapRef.current.eachLayer((layer) => {
          if (layer instanceof L.Circle || layer instanceof L.Polyline) {
            mapRef.current?.removeLayer(layer);
          }
        });

        // Add the new marker
        if (mapRef.current) {
          L.circle([position.latitude, position.longitude], {
            radius: 8,
            color: 'blue',
            fillColor: 'blue',
            fillOpacity: 0.8
          }).addTo(mapRef.current); // draw current position marker
        }

        // Draw the path as a polyline
        /*if (gpsPath.length > 1 && mapRef.current) {
          L.polyline(gpsPath, {
            color: 'blue',
            weight: 3,
          }).addTo(mapRef.current);
        }*/
      }
    }
  }, [nodeTopicType, position, gpsPath]);

  // Create circular texture once and reuse it (standard Three.js approach for circular points)
  const circleTextureRef = useRef<THREE.CanvasTexture | null>(null);
  
  if (!circleTextureRef.current) {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const context = canvas.getContext('2d');
    if (context) {
      // Draw a white circle on transparent background
      context.beginPath();
      context.arc(32, 32, 30, 0, Math.PI * 2);
      context.fillStyle = 'white';
      context.fill();
    }
    circleTextureRef.current = new THREE.CanvasTexture(canvas);
    circleTextureRef.current.needsUpdate = true;
  }

  // Render 3D point cloud using Three.js
  const PointCloud: React.FC<{ pointCloud: { positions: number[]; colors: number[] } }> = ({ pointCloud }) => {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(pointCloud.positions);
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    // Check if colors are available
    const hasColors = pointCloud.colors && pointCloud.colors.length > 0;
    
    if (hasColors) {
      // Normalize colors from 0-255 to 0-1 range for Three.js
      const normalizedColors = pointCloud.colors.map(c => c / 255);
      const colors = new Float32Array(normalizedColors);
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    }

    const material = new THREE.PointsMaterial({
      size: 0.03,
      vertexColors: hasColors, // Use vertex colors if available
      color: hasColors ? 0xffffff : 0x888888, // White if using colors, gray otherwise
      map: circleTextureRef.current, // Use circular texture (standard way to render circles in Three.js)
      transparent: true,
      alphaTest: 0.1, // Discard pixels with low alpha
    });

    return <primitive object={new THREE.Points(geometry, material)} />;
  };

  // Rotate Three.js scene for better viewing angle
  const RotateScene = () => {
    const { scene, camera } = useThree();

    // Initial scene tilt
    useEffect(() => {
      scene.rotation.x = -1.8; // -3 für boden
      scene.rotation.y = 0;
      scene.rotation.z = 1.6;
    }, [scene]);

    return null; // No need to render anything here
  };
  // Render IMU visualization with orientation and vector arrows
  const ImuVisualizer: React.FC<{ imu: NonNullable<typeof imuData> }> = ({ imu }) => {
    const groupRef = useRef<THREE.Group>(null);

    useEffect(() => {
      if (groupRef.current && imu.orientation) {
        const { x, y, z, w } = imu.orientation;
        const quaternion = new THREE.Quaternion(x, y, z, w);
        groupRef.current.setRotationFromQuaternion(quaternion);
      }
    }, [imu]);

    return (
      <>
        <group ref={groupRef}>
          <primitive object={new THREE.AxesHelper(1)} />
          <primitive object={new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 1, 0xff0000, 0.2, 0.1)} />
          <primitive object={new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 1, 0x00ff00, 0.2, 0.1)} />
          <primitive object={new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 1, 0x0000ff, 0.2, 0.1)} />

          {(() => {
            const angularVector = new THREE.Vector3(
              imu.angular_velocity.x,
              imu.angular_velocity.y,
              imu.angular_velocity.z
            );
            if (angularVector.length() === 0) return null;
            return (
              <primitive object={new THREE.ArrowHelper(
                angularVector.clone().normalize(),
                new THREE.Vector3(0, 0, 0),
                angularVector.length(),
                0xff69b4, 0.2, 0.1
              )} />
            );
          })()}

          {(() => {
            const accVector = new THREE.Vector3(
              imu.linear_acceleration.x,
              imu.linear_acceleration.y,
              imu.linear_acceleration.z
            );
            const accScale = 0.2;
            if (accVector.length() === 0) return null; // Don't render yellow arrow if zero
            return (
              <primitive object={new THREE.ArrowHelper(
                accVector.clone().normalize(),
                new THREE.Vector3(0, 0, 0),
                accVector.length() * accScale,
                0xffff00, 0.2, 0.1
              )} />
            );
          })()}
        </group>
      </>
    );
  };

  // Reusable TypographyBox component - always shows topic and timestamp
  const TypographyBox: React.FC = () => (
    <div className="typography-box">
      <Typography variant="body2">{nodeTopic}</Typography>
    </div>
  );

  // Render content based on the topicType
  let renderedContent: React.ReactNode = null;
  const isGpsTopic = nodeTopicType === "sensor_msgs/msg/NavSatFix" || nodeTopicType === "novatel_oem7_msgs/msg/BESTPOS";
  if (isGpsTopic) {
    // Always keep the map container in the DOM so mapRef stays attached across timestamp changes.
    // The spinner is overlaid on top instead of replacing the content.
    renderedContent = (
      <div className="gps-container" style={{ position: 'relative' }}>
        <div ref={mapContainerRef} className="map-container"></div>
        {positionIsZero && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
            <p style={{ color: 'white', fontSize: '0.8rem' }}>No positional data found.</p>
          </div>
        )}
      </div>
    );
  } else if (isLoading) {
    renderedContent = (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100%', height: '100%' }}>
        <CircularProgress size={40} sx={{ color: 'white' }} />
      </Box>
    );
  } else if ((nodeTopicType === "sensor_msgs/msg/CompressedImage" || nodeTopicType === "sensor_msgs/msg/Image") && imageUrl) {
    renderedContent = (
      <div className="image-container">
        <img
          src={imageUrl}
          alt={`Image for ${nodeTopic}`}
          loading="lazy"
        />
      </div>
    ); // image rendering block
  } else if (pointCloud) {
    renderedContent = (
      <div className="canvas-container">
        <Canvas camera={{ position: [5, 0, -4], fov: 75 }}>
          <RotateScene />
          <OrbitControls />
          <pointLight position={[10, 10, 10]} />
          <PointCloud pointCloud={pointCloud} />
        </Canvas>
      </div>
    ); // point cloud rendering block
  } else if (text) {
    renderedContent = (
      <Box>
        <Typography variant="body2" sx={{ color: "white", whiteSpace: "pre-wrap" }}>
          {text}
        </Typography>
      </Box>
    ); // text rendering block
  } else if (imuData) {
    renderedContent = (
      <div className="canvas-container">
        <Canvas camera={{ position: [5, 0, -4], fov: 75 }}>
          <ambientLight />
          <ImuVisualizer imu={imuData} />
        </Canvas>
      </div>
    ); // IMU visualization block
  } else {
    renderedContent = (
      <div className="centered-text">
        <p style={{ color: "white", fontSize: "0.8rem" }}>
          {nodeTopic ? "No content found for this topic and timestamp." : "Select a topic"}
        </p>
      </div>
    ); // fallback for empty/no content
  }

  return (
    <div className="node-content">
      {renderedContent}
      <TypographyBox />
    </div>
  );
};

export default NodeContent;