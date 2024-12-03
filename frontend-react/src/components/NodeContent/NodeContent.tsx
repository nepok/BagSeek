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
    if (topic && timestamp) {
      try {
        const response = await fetch(`/api/ros?timestamp=${timestamp}&topic=${topic}`);
        const data = await response.json();

        if (topic.startsWith("/camera_image")) {
          setImage(data.image || null); // Store image for camera topics
          setText(null);
          setPoints(null);
        } else if (topic === "/points") {
          setPoints(data.points || null); // Store points for point cloud topics
          setImage(null);
          setText(null);
        } else if (data.text) {
          setText(data.text); // Store text for other topics
          setImage(null);
          setPoints(null);
        } else {
          setImage(null);
          setText(null);
          setPoints(null);
        }

        setRealTimestamp(data.realTimestamp || null); // Store real timestamp if provided
      } catch (error) {
        console.error("Error fetching data:", error);
        setImage(null);
        setText(null);
        setPoints(null);
      }
    } else {
      setImage(null);
      setText(null);
      setPoints(null);
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
      scene.rotation.x = -1;
      scene.rotation.y = 0.9;
      scene.rotation.z = 1.6;
    }, [scene]);
    return null; // No need to render anything here
  };

  // Render content based on the topic
  switch (true) {
    case topic?.startsWith("/camera_image"):
      return (
        <div className="node-content">
          <div className="image-container">
            {image ? (
                <img
                  src={`data:image/webp;base64,${image}`}
                  alt="Fetched from ROS"
                  loading="lazy"
                />
            ) : (
              <p className="centered-text" style={{ color: "white", fontSize: "0.8rem" }}>
                Loading image...
              </p>
            )}
          </div>
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
      );

    case topic === "/points":
      return (
        <div className="node-content">
          <div className="canvas-container">
            {points ? (
              <Canvas camera={{ position: [0, 0, 5], fov: 90 }}>
                <RotateScene /> {/* Rotate the whole scene */}
                <OrbitControls /> {/* Set the target to rotate the camera */}
                <pointLight position={[10, 10, 10]} />
                <PointCloud points={points} />
              </Canvas>
            ) : (
              <p className="centered-text" style={{ color: "white", fontSize: "0.8rem" }}>
                Loading point cloud...
              </p>
            )}
          </div>
          <div className="typography-box">
            <Typography variant="body2">{topic}</Typography>
            <Typography variant="body2">{realTimestamp}</Typography>
          </div>
        </div>
      );

    case topic && !topic.startsWith("/camera_image"):
      return (
        <div className="node-content">
          <Box>
            {text ? (
              <Typography variant="body2" sx={{ color: "white", whiteSpace: "pre-wrap" }}>
                {text}
              </Typography>
            ) : (
              <p style={{ color: "white", fontSize: "0.8rem" }}></p>
            )}
          </Box>
          <div className="typography-box">
            <Typography variant="body2">{topic}</Typography>
            <Typography variant="body2">{realTimestamp}</Typography>
          </div>
        </div>
      );

    default:
      return (
        <div className="centered-text">
          <p style={{ color: "white", fontSize: "0.8rem" }}>
            Unknown topic: {topic}
          </p>
        </div>
      );
  }
};

export default NodeContent;