import React, { useState, useEffect, useRef } from "react";
import "./NodeContent.css"; // Import the CSS file
import { Box, Typography } from "@mui/material";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import L from 'leaflet';
import 'leaflet/dist/leaflet.css'; // Ensure Leaflet CSS is loaded

interface NodeContentProps {
  topic: string | null;
  timestamp: number | null; // Timestamp passed for fetching relevant data
  selectedRosbag: string | null;
}

const NodeContent: React.FC<NodeContentProps> = ({ topic, timestamp, selectedRosbag }) => {
  //const [image, setImage] = useState<string | null>(null); // Store the fetched image
  const [text, setText] = useState<string | null>(null);   // Store the fetched text
  const [points, setPoints] = useState<number[] | null>(null); // Store fetched point cloud
  const [realTimestamp, setRealTimestamp] = useState<string | null>(null);
  const [gpsData, setGpsData] = useState<{ latitude: number; longitude: number; altitude: number} | null>(null);
  const [gpsPath, setGpsPath] = useState<[number, number][]>([]);

  const imageUrl =
  topic && timestamp && selectedRosbag
    ? `http://localhost:5000/images/${selectedRosbag}/${topic.replaceAll("/", "__")}-${timestamp}.webp`
    : undefined;

  const isImage = topic && (topic.includes("image") || topic.includes("camera"));

  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  
  // Function to fetch data based on the topic and timestamp
  const fetchData = async () => {
    try {
      const response = await fetch(`/api/ros?timestamp=${timestamp}&topic=${topic}`);
      const data = await response.json();
      // Reset all states
      //setImage(null);
      setText(null);
      setPoints(null);
      setGpsData(null);

      // Dynamically update state based on the keys present in the response
      //if (data.image) {
      //  setImage(data.image);
      //}
      if (data.gpsData) {
        setGpsData(data.gpsData);
      }
      if (data.points) {
        setPoints(data.points);
      }
      if (data.text) {
        setText(data.text);
      }
      if (data.realTimestamp) {
        setRealTimestamp(data.realTimestamp);
      }
    } catch (error) {
      console.error("Error fetching data:", error);
      // Reset all states in case of error
      //setImage(null);
      setText(null);
      setPoints(null);
      setRealTimestamp(null);
    }
  };

  // Trigger fetch when topic or timestamp changes
  useEffect(() => {
    if (topic && timestamp && selectedRosbag) {
      if (!topic.includes("image") && !topic.includes("camera")) { // Only call the API if there is NO image or camera        
        fetchData();
      } else {
        setText(null);
        setPoints(null);
        setGpsData(null);
      }
    }
  }, [topic, timestamp, selectedRosbag]);

  useEffect(() => {
    if (gpsData) {
      setGpsPath(prevPath => [...prevPath, [gpsData.latitude, gpsData.longitude]]);
    }
  }, [gpsData]);

  useEffect(() => {
  
    // Clear the map if switching away from GPS
    if (!topic?.includes("gps") && mapRef.current) {
      mapRef.current.remove(); // Properly remove the map instance
      mapRef.current = null;
    }
  
    // Initialize or update the map if GPS topic is selected
    if (topic?.includes("gps") && gpsData) {
      // Initialize the map ONLY if it doesn't exist or when switching back to GPS
      if (!mapRef.current && mapContainerRef.current) {
        mapRef.current = L.map(mapContainerRef.current).setView(
          [gpsData.latitude, gpsData.longitude], 16
        );
  
        L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxZoom: 19,
          attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        }).addTo(mapRef.current!);
      }
  
      // Always update the view and layers when GPS data changes
      if (mapRef.current) {
        mapRef.current.setView([gpsData.latitude, gpsData.longitude], 16);
        mapRef.current.invalidateSize();
  
        // Clear existing markers and polylines
        mapRef.current.eachLayer((layer) => {
          if (layer instanceof L.Circle || layer instanceof L.Polyline) {
            mapRef.current?.removeLayer(layer);
          }
        });
  
        // Add the new marker
        if (mapRef.current) {
          L.circle([gpsData.latitude, gpsData.longitude], {
            radius: 8, 
            color: 'blue',
            fillColor: 'blue',
            fillOpacity: 0.8
          }).addTo(mapRef.current);
        }
  
        // Draw the path as a polyline
        if (gpsPath.length > 1 && mapRef.current) {
          L.polyline(gpsPath, {
            color: 'blue',
            weight: 3,
          }).addTo(mapRef.current);
        }
      }
    }
  }, [topic, gpsData, gpsPath]);

  // Point cloud rendering component
  const PointCloud: React.FC<{ points: number[] }> = ({ points }) => {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(points);
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      color: 0x888888,
      size: 0.05,
    });

    return <primitive object={new THREE.Points(geometry, material)} />;
  };

  // Rotate the scene 90 degrees on the Y-axis
  const RotateScene = () => {
    const { scene } = useThree();
    useEffect(() => {
      scene.rotation.x = -1.8; // -3 für boden
      scene.rotation.y = -0;
      scene.rotation.z = 1.6;
    }, [scene]);
    return null; // No need to render anything here
  };

  // Render content based on the type of data fetched
  return (
    <div className="node-content">
      {isImage && (
        <div className="image-container">
          <img
            src={imageUrl}
            alt="Select a topic"
            loading="lazy"
          />
          <div className="typography-box">
            <Typography
              variant="body2"
              sx={{
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {topic}
            </Typography>
            <Typography variant="body2">{realTimestamp}</Typography>
          </div>
        </div>
      )}

      {points && (
        <div className="canvas-container">
          <Canvas camera={{ position: [0, 0, 5], fov: 90 }}>
            <RotateScene /> {/* Rotate the whole scene */}
            <OrbitControls /> {/* Set the target to rotate the camera */}
            <pointLight position={[10, 10, 10]} />
            <PointCloud points={points} />
          </Canvas>
          <div className="typography-box">
            <Typography variant="body2">{topic}</Typography>
            <Typography variant="body2">{realTimestamp}</Typography>
          </div>
        </div>
      )}

      {text && (
        <Box>
          <Typography variant="body2" sx={{ color: "white", whiteSpace: "pre-wrap" }}>
            {text}
          </Typography>
          <div className="typography-box">
            <Typography variant="body2">{topic}</Typography>
            <Typography variant="body2">{realTimestamp}</Typography>
          </div>
        </Box>
      )}

    {gpsData && (
      <div className="gps-container">           
        <div ref={mapContainerRef} className="map-container"></div>
      </div>
    )}

      {/* Default case: No data */}
      {!imageUrl && !points && !text && !gpsData && (
        <div className="centered-text">
          <p style={{ color: "white", fontSize: "0.8rem" }}>
            {topic ? "No data availible" : "Select a topic"}
          </p>
        </div>
      )}
    </div>
  );
};

export default NodeContent;