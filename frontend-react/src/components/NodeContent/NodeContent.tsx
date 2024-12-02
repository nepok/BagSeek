import React, { useState, useEffect } from "react";
import "./NodeContent.css"; // Import the CSS file
import { Box, Typography } from "@mui/material";

interface NodeContentProps {
  topic: string | null;
  timestamp: number | null; // Timestamp passed for fetching relevant images
}

const NodeContent: React.FC<NodeContentProps> = ({ topic, timestamp }) => {
  const [image, setImage] = useState<string | null>(null); // Store the fetched image
  const [text, setText] = useState<string | null>(null);   // Store the fetched text
  const [realTimestamp, setRealTimestamp] = useState<string | null>(null);

  // Function to fetch image or text based on the topic and timestamp
  const fetchData = async () => {
    if (topic && timestamp) {
      try {
        const response = await fetch(`/api/ros?timestamp=${timestamp}&topic=${topic}`);
        const data = await response.json();

        if (data.image) {
          setImage(data.image); // Store fetched image
          setText(null);         // Reset text if image is fetched
        } else if (data.text) {
          setText(data.text);   // Store fetched text
          setImage(null);        // Reset image if text is fetched
        } else {
          setImage(null);        // Reset image if no image or text is found
          setText(null);         // Reset text
        }

        setRealTimestamp(data.realTimestamp || null); // Store real timestamp if provided
      } catch (error) {
        console.error("Error fetching data:", error);
        setImage(null);
        setText(null);
      }
    } else {
      setImage(null); // Reset image if conditions aren't met
      setText(null);  // Reset text if conditions aren't met
    }
  };

  // Trigger fetch when topic or timestamp changes
  useEffect(() => {
    if (topic) {
      fetchData();
    } else {
      setImage(null);  // Reset if no topic is provided
      setText(null);   // Reset text if no topic is provided
    }
  }, [topic, timestamp]);

  // Render content based on the topic
  switch (true) {
    case topic?.startsWith("/camera_image") && topic !== "/camera_image/Cam_MR":
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
              <p style={{ color: "white", fontSize: "0.8rem" }}>Loading image...</p>
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

    case topic && !topic.startsWith("/camera_image") && topic !== "/camera_image/Cam_MR":
      return (
        <div className="node-content">
          <Box>
            {text ? (
              <Typography variant="body2" sx={{ color:"white", whiteSpace: "pre-wrap" }}>
                {text}
              </Typography>
            ) : (
              <p style={{ color: "white", fontSize: "0.8rem" }}>Loading text...</p>
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