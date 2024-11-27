import React, { useState, useEffect } from "react";
import { Button, Menu, MenuItem, Dialog, DialogActions, DialogContent, Typography, Box } from "@mui/material";
import "./CanvasContainer.css";

interface CanvasContainerProps {
  selectedTimestamp: number | null;
  topics: string[];
}

const CanvasContainer: React.FC<CanvasContainerProps> = ({
  selectedTimestamp,
  topics,
}) => {
  const [showSettings, setShowSettings] = useState(false);
  const [showTopicMenu, setShowTopicMenu] = useState<null | HTMLElement>(null);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [image, setImage] = useState<string | null>(null); // State to store the fetched image
  const [realTimestamp, setRealTimestamp] = useState<string | null>(null);

  const toggleSettings = () => {
    setShowSettings(!showSettings);
    setShowTopicMenu(null); // Reset topic menu visibility
  };
  
  const handleTopicClick = (topic: string) => {
    setSelectedTopic(topic);
    setShowTopicMenu(null);
    setShowSettings(false);
  };

  // Function to fetch image from API based on timestamp
  const fetchImage = async (timestamp: number) => {
    if (timestamp && selectedTopic && selectedTopic.startsWith("/camera_image/")) { // Check if selected topic starts with "/camera_image/"
      try {
        const response = await fetch(`/api/ros?timestamp=${timestamp}&topic=${selectedTopic}`);
        const data = await response.json();
        if (data.image) {
          setImage(data.image); // Set the fetched image
          if (data.realTimestamp) {
            setRealTimestamp(data.realTimestamp); // Set the fetched value, use it as needed
          }
        } else {
          setImage(null);
        }
      } catch (error) {
        console.error("Error fetching image:", error);
      }
    } else {
      setImage(null); // Reset image if topic is not under "/camera_image/"
    }
  };

  // Whenever the selectedTimestamp or selectedTopic changes, fetch the new image
  useEffect(() => {
    if (selectedTimestamp && selectedTopic && selectedTopic.startsWith("/camera_image/")) {
      fetchImage(selectedTimestamp);
    } else {
      setImage(null); // Reset the image if topic is not under "/camera_image/"
    }
  }, [selectedTimestamp, selectedTopic]);

  return (
    <Box className="canvas-container" sx={{ padding: 2, backgroundColor: "background.default" }}>
      {/* Settings button */}
      <Button onClick={toggleSettings} variant="contained" sx={{ marginBottom: 2 }}>
        Settings
      </Button>

      {/* Settings menu */}
      {showSettings && (
        <Box sx={{ display: "flex", flexDirection: "column" }}>
          <Button onClick={() => setShowTopicMenu((prev) => prev ? null : document.body)} sx={{ marginBottom: 1 }}>
            Choose Topic
          </Button>
          <Button onClick={() => setShowSettings(false)} sx={{ marginBottom: 1 }}>
            Create new canvas below
          </Button>
          <Button onClick={() => setShowSettings(false)}>
            Create new canvas on the right
          </Button>
        </Box>
      )}

      {/* Topic selection menu */}
      <Menu
        anchorEl={showTopicMenu}
        open={Boolean(showTopicMenu)}
        onClose={() => setShowTopicMenu(null)}
      >
        {topics.map((topic, index) => (
          <MenuItem key={index} onClick={() => handleTopicClick(topic)}>
            {topic}
          </MenuItem>
        ))}
      </Menu>

      {selectedTopic && (
        <Box sx={{ marginTop: 2 }}>
          <Typography variant="body2">
            <strong>Selected Topic:</strong> {selectedTopic}
          </Typography>
          <Typography variant="body2">
            <strong>Timestamp:</strong> {realTimestamp}
          </Typography>
        </Box>
      )}

      {/* Display the image only if "/camera_image/..." is selected */}
      {selectedTopic?.startsWith("/camera_image") && image && (
        <Box sx={{ marginTop: 2, display: "flex", justifyContent: "center" }}>
          <img src={`data:image/png;base64,${image}`} alt="Fetched from ROS" />
        </Box>
      )}
    </Box>
  );
};

export default CanvasContainer;