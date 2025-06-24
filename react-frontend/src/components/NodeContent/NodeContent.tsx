import React, { useState, useEffect, useRef } from "react";
import "./NodeContent.css"; // Import the CSS file
import { Box, Typography } from "@mui/material";
import { Canvas, useThree, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { ArrowHelper } from "three";
import L from 'leaflet';
import 'leaflet/dist/leaflet.css'; // Ensure Leaflet CSS is loaded

interface NodeContentProps {
  nodeTopic: string | null;
  nodeTopicType: string | null; // Type of the topic, e.g., "sensor_msgs/Image"
  selectedRosbag: string | null;
  mappedTimestamp: number | null; // Optional timestamp for image reference
}

const NodeContent: React.FC<NodeContentProps> = ({ nodeTopic, nodeTopicType, selectedRosbag, mappedTimestamp }) => {
  //const [image, setImage] = useState<string | null>(null); // Store the fetched image
  const [referenceMap, setReferenceMap] = useState<Record<string, Record<string, string>>>({});
  const [text, setText] = useState<string | null>(null);   // Store the fetched text
  const [pointCloud, setPointCloud] = useState<number[] | null>(null); // Store fetched point cloud
  const [realTimestamp, setRealTimestamp] = useState<string | null>(null);
  const [position, setPosition] = useState<{ latitude: number; longitude: number; altitude: number} | null>(null);
  const [gpsPath, setGpsPath] = useState<[number, number][]>([]);
  const [dataType, setDataType] = useState<string | null>(null); // Store the type of data fetched
  const [imuData, setImuData] = useState<{
    orientation: { x: number; y: number; z: number; w: number };
    angular_velocity: { x: number; y: number; z: number };
    linear_acceleration: { x: number; y: number; z: number };
  } | null>(null);
  
  const imageUrl =
    nodeTopic && mappedTimestamp && selectedRosbag
      ? `http://localhost:5000/images/${selectedRosbag}/${nodeTopic.replaceAll("/", "__")}-${mappedTimestamp}.webp`
      : undefined;

  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  
  // Function to fetch data based on the topic and timestamp
  const fetchData = async () => {
    try {
      const response = await fetch(`/api/content?timestamp=${mappedTimestamp}&topic=${nodeTopic}`);
      const data = await response.json();

      setText(null);
      setPointCloud(null);
      setPosition(null);

      if(data.type) {
        setDataType(data.type);
      }

      switch (data.type) {
        case 'text':
          setText(data.text);
          break;
        case 'pointCloud':
          setPointCloud(data.pointCloud);
          break;
        case 'position':
          setPosition(data.position);
          break;
        case 'imu':
            setImuData(data.imu);
          break;
        case 'tf':
            const { translation, rotation } = data.tf;
            setImuData({
              orientation: rotation,
              angular_velocity: { x: 0, y: 0, z: 0 },
              linear_acceleration: { x: 0, y: 0, z: 0 }
            });
          break;
        default:
          console.warn("Unknown data type:", data.type);
      }
  // IMU Visualizer component
  const ImuVisualizer: React.FC<{ imu: typeof imuData }> = ({ imu }) => {
    const groupRef = useRef<THREE.Group>(null);

    useEffect(() => {
      if (groupRef.current && imu) {
        const { x, y, z, w } = imu.orientation;
        const quaternion = new THREE.Quaternion(x, y, z, w);
        groupRef.current.setRotationFromQuaternion(quaternion);
      }
    }, [imu]);

    return (
      <>
        <group ref={groupRef}>
          <mesh position={[1, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
            <coneGeometry args={[0.05, 0.2, 8]} />
            <meshBasicMaterial color="red" />
          </mesh>
          <mesh position={[0, 1, 0]}>
            <coneGeometry args={[0.05, 0.2, 8]} />
            <meshBasicMaterial color="green" />
          </mesh>
          <mesh position={[0, 0, 1]} rotation={[Math.PI / 2, 0, 0]}>
            <coneGeometry args={[0.05, 0.2, 8]} />
            <meshBasicMaterial color="blue" />
          </mesh>
        </group>
      </>
    );
  };
    } catch (error) {
      console.error("Error fetching data:", error);
      // Reset all states in case of error
      setText(null);
      setPointCloud(null);
      setRealTimestamp(null);
    }
  };

  // Trigger fetch when topic or timestamp changes
  useEffect(() => {
    if (nodeTopic && nodeTopicType && mappedTimestamp && selectedRosbag) {
      // Only call the API if the topicType is not an image
      if (
        nodeTopicType !== "sensor_msgs/msg/CompressedImage" &&
        nodeTopicType !== "sensor_msgs/msg/Image"
      ) {
        fetchData();
      } else {
        setText(null);
        setPointCloud(null);
        setPosition(null);
      }
    }
  }, [nodeTopic, nodeTopicType, mappedTimestamp, selectedRosbag]);

  useEffect(() => {
    if (position) {
      setGpsPath(prevPath => [...prevPath, [position.latitude, position.longitude]]);
    }
  }, [position]);

  useEffect(() => {
    // Clear the map if switching away from GPS
    if ((nodeTopicType !== "sensor_msgs/msg/NavSatFix" && nodeTopicType !== "novatel_oem7_msgs/msg/BESTPOS") && mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }

    // Initialize or update the map if GPS topic is selected
    if (position) {
      // Initialize the map ONLY if it doesn't exist or when switching back to GPS
      if (!mapRef.current && mapContainerRef.current) {
        mapRef.current = L.map(mapContainerRef.current).setView(
          [position.latitude, position.longitude], 16
        );

        L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxZoom: 19,
          attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        }).addTo(mapRef.current!);
      }

      // Update the view to the new position but preserve current zoom level
      if (mapRef.current) {
        const currentZoom = mapRef.current.getZoom();
        mapRef.current.setView([position.latitude, position.longitude], currentZoom);
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
          }).addTo(mapRef.current);
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

  // Point cloud rendering component
  const PointCloud: React.FC<{ pointCloud: number[] }> = ({ pointCloud }) => {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(pointCloud);
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

  // ImuVisualizer component
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

  // Render content based on the topicType
  let renderedContent: React.ReactNode = null;
  if ((nodeTopicType === "sensor_msgs/msg/CompressedImage" || nodeTopicType === "sensor_msgs/msg/Image") && imageUrl) {
    renderedContent = (
      <div className="image-container">
        <img
          src={imageUrl}
          alt={`Image for ${nodeTopic}`}
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
            {nodeTopic}
          </Typography>
          <Typography variant="body2">{realTimestamp}</Typography>
        </div>
      </div>
    );
  } else if (pointCloud) {
    renderedContent = (
      <div className="canvas-container">
        <Canvas camera={{ position: [0, 0, 5], fov: 90 }}>
          <RotateScene />
          <OrbitControls />
          <pointLight position={[10, 10, 10]} />
          <PointCloud pointCloud={pointCloud} />
        </Canvas>
        <div className="typography-box">
          <Typography variant="body2">{nodeTopic}</Typography>
          <Typography variant="body2">{realTimestamp}</Typography>
        </div>
      </div>
    );
  } else if (text) {
    renderedContent = (
      <Box>
        <Typography variant="body2" sx={{ color: "white", whiteSpace: "pre-wrap" }}>
          {text}
        </Typography>
        <div className="typography-box">
          <Typography variant="body2">{nodeTopic}</Typography>
          <Typography variant="body2">{realTimestamp}</Typography>
        </div>
      </Box>
    );
  } else if (position) {
    renderedContent = (
      <div className="gps-container">
        <div ref={mapContainerRef} className="map-container"></div>
      </div>
    );
  } else if (imuData) {
    renderedContent = (
      <div className="canvas-container">
        <Canvas camera={{ position: [0, 0, 5], fov: 70 }}>
          <ambientLight />
          <ImuVisualizer imu={imuData} />
        </Canvas>
      </div>
    );
  } else {
    renderedContent = (
      <div className="centered-text">
        <p style={{ color: "white", fontSize: "0.8rem" }}>
          {nodeTopic ? "No content found for this topic and timestamp." : "Select a topic"}
        </p>
      </div>
    );
  }

  return (
    <div className="node-content">
      {renderedContent}
    </div>
  );
};

export default NodeContent;