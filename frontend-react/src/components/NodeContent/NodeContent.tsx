import React, { useState, useEffect } from "react";
import "./NodeContent.css"; // Import the CSS file
import { Box, Typography } from "@mui/material";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

interface NodeContentProps {
  topic: string | null;
  timestamp: number | null; // Timestamp passed for fetching relevant data
}

const NodeContent: React.FC<NodeContentProps> = ({ topic, timestamp }) => {
  const [image, setImage] = useState<string | null>(null); // Store the fetched image
  const [text, setText] = useState<string | null>(null);   // Store the fetched text
  const [points, setPoints] = useState<number[] | null>(null); // Store fetched point cloud
  const [realTimestamp, setRealTimestamp] = useState<string | null>(null);

  // Function to fetch data based on the topic and timestamp
  const fetchData = async () => {
    try {
      const response = await fetch(`/api/ros?timestamp=${timestamp}&topic=${topic}`);
      const data = await response.json();

      // Reset all states
      setImage(null);
      setText(null);
      setPoints(null);

      // Dynamically update state based on the keys present in the response
      if (data.image) {
        setImage(data.image);
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
      setImage(null);
      setText(null);
      setPoints(null);
      setRealTimestamp(null);
    }
  };

  // Trigger fetch when topic or timestamp changes
  useEffect(() => {
    if (topic) {
      fetchData();
    } else {
      setImage(null);
      setText(null);
      setPoints(null);
    }
  }, [topic, timestamp]);

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
      {image && (
        <div className="image-container">
          <img
            src={`data:image/webp;base64,${image}`}
            alt="Fetched from ROS"
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

      {/* Default case: No data */}
      {!image && !points && !text && (
        <div className="centered-text">
          <p style={{ color: "white", fontSize: "0.8rem" }}>
            {topic ? "Loading data..." : "Select a topic"}
          </p>
        </div>
      )}
    </div>
  );
};

export default NodeContent;