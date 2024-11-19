import React, { useState, useEffect } from 'react';
import './CanvasContainer.css';

interface CanvasContainerProps {
  selectedTimestamp: number | null;
  onCreateCanvas: (direction: 'below' | 'right') => void;
  topics: string[];
}

const CanvasContainer: React.FC<CanvasContainerProps> = ({
  selectedTimestamp,
  onCreateCanvas,
  topics,
}) => {
  const [showSettings, setShowSettings] = useState(false);
  const [showTopicMenu, setShowTopicMenu] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [image, setImage] = useState<string | null>(null); // State to store the fetched image
  
  const toggleSettings = () => setShowSettings(!showSettings);
  const toggleTopicMenu = () => setShowTopicMenu(!showTopicMenu);

  const handleTopicClick = (topic: string) => {
    console.log(`Topic selected: ${topic}`); // Debugging log
    setSelectedTopic(topic);
    setShowTopicMenu(false);
    setShowSettings(false);
  };

  // Function to fetch image from API based on timestamp
  const fetchImage = async (timestamp: number) => {
    if (timestamp && selectedTopic === '/camera_image/Cam_BR') { // Only fetch image if '/camera_image/Cam_BR' is selected
      console.log('Fetching image for timestamp:', timestamp); // Debugging log
      try {
        const response = await fetch(`/api/ros?timestamp=${timestamp}`);
        const data = await response.json();
        if (data.image) {
          console.log('Image fetched successfully'); // Debugging log
          setImage(data.image); // Set the fetched image
        } else {
          console.error('No image found for this timestamp');
        }
      } catch (error) {
        console.error('Error fetching image:', error);
      }
    } else {
      setImage(null); // Reset image if topic is not '/camera_image/Cam_BR'
      console.log('No image fetched, invalid topic or timestamp');
    }
  };

  // Whenever the selectedTimestamp or selectedTopic changes, fetch the new image
  useEffect(() => {
    if (selectedTimestamp && selectedTopic === '/camera_image/Cam_BR') {
      fetchImage(selectedTimestamp);
    } else {
      setImage(null); // Reset the image if topic is not '/camera_image/Cam_BR'
    }
  }, [selectedTimestamp, selectedTopic]);

  return (
    <div className="canvas-container">
      {/* Settings button */}
      <div className="canvas-settings-button" onClick={toggleSettings}>
        &#x22EE;
      </div>

      {/* Settings menu */}
      {showSettings && (
        <div className="canvas-settings-bar">
          <button onClick={() => {toggleTopicMenu();}}>Choose Topic</button>
          <button onClick={() => {onCreateCanvas('below');}}>Create new canvas below</button>
          <button onClick={() => {onCreateCanvas('right');}}>Create new canvas on the right</button>
        </div>
      )}

      {/* Topic selection menu */}
      {showTopicMenu && (
        <div className="canvas-settings-bar left">
          <ul style={{ padding: 0 }}>
            {topics.map((topic, index) => (
              <li
                key={index}
                style={{ color: '#fff', padding: '5px 0', cursor: 'pointer' }}
                onClick={() => handleTopicClick(topic)}
              >
                {topic}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Display the selected timestamp */}
      <p style={{ color: '#fff' }}>Currently selected timestamp: {selectedTimestamp}</p>

      {/* Display the selected topic */}
      {selectedTopic && (
        <p style={{ color: '#fff', marginTop: '10px' }}>
          Selected Topic: {selectedTopic}
        </p>
      )}

      {/* Display the image only if '/camera_image/Cam_BR' is selected */}
      {selectedTopic === '/camera_image/Cam_BR' && image && (
        <div className="image-container">
          <img src={`data:image/png;base64,${image}`} alt="Fetched from ROS" />
        </div>
      )}
    </div>
  );
};

export default CanvasContainer;
