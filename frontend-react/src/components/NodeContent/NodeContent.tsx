import React, { useState, useEffect } from "react";
import "./NodeContent.css"; // Import the CSS file
import { Box, Typography } from "@mui/material";

interface NodeContentProps {
  topic: string | null;
  timestamp: number | null; // Timestamp passed for fetching relevant images
}

const NodeContent: React.FC<NodeContentProps> = ({ topic, timestamp }) => {
  const [image, setImage] = useState<string | null>(null); // Store the fetched image
  const [realTimestamp, setRealTimestamp] = useState<string | null>(null);

  // Function to fetch an image based on the topic and timestamp
  const fetchImage = async () => {
    if (topic && timestamp) {
      try {
        const response = await fetch(`/api/ros?timestamp=${timestamp}&topic=${topic}`);
        const data = await response.json();
        if (data.image) {
          setImage(data.image); // Store fetched image
          setRealTimestamp(data.realTimestamp || null); // Store real timestamp if provided
        } else {
          setImage(null); // Reset if no image is available
        }
      } catch (error) {
        console.error("Error fetching image:", error);
        setImage(null);
      }
    } else {
      setImage(null); // Reset image if conditions aren't met
    }
  };

  // Trigger image fetch when topic or timestamp changes
  useEffect(() => {
    if (topic?.startsWith("/camera_image/")) {
      fetchImage();
    } else {
      setImage(null); // Reset image if topic is not relevant
    }
  }, [topic, timestamp]);

  // Render content based on the topic
  switch (true) {
    case topic?.startsWith("/camera_image") && topic !== "/camera_image/Cam_MR":
      return (
        <div> 
          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            sx={{
              position: 'absolute',
              bottom: 10,
              width: '100%',
              maxHeight: '30px',
              backgroundColor: 'rgba(33, 33, 33, 0.8)', // Dark grey with transparency
              padding: '8px',
              borderRadius: '10px', // Rounded corners
              color: 'white', // Text color
              zIndex: 10, // Ensure it stays on top of the image
              overflow: 'hidden', // Ensures content is clipped if it overflows
            }}
          >
            <Typography variant="body2" sx={{ color: 'white', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1}}>
              {topic}
            </Typography>
            <Typography variant="body2" sx={{color: 'white'}}>
              {realTimestamp}
            </Typography>
          </Box>
          {image ? (
            <div className="image-container">
              <img
                src={`data:image/png;base64,${image}`}
                alt="Fetched from ROS"
                loading="lazy"
              />
            </div>
          ) : (
            <p style={{ color: "white", fontSize: "0.8rem" }}>Loading image...</p>
          )}
        </div>
      );

    default:
      return <p style={{ color: "white", fontSize: "0.8rem" }}>Unknown topic: {topic}</p>;
  }
};

export default NodeContent;