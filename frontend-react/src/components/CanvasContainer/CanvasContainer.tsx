import React, { useState } from 'react';
import './CanvasContainer.css';

interface CanvasContainerProps {
  selectedTimestamp: number | null;
  onCreateCanvas: (direction: 'below' | 'right') => void;
  topics: string[]; // Add a topics prop
}

const CanvasContainer: React.FC<CanvasContainerProps> = ({
  selectedTimestamp,
  onCreateCanvas,
  topics,
}) => {
  const [showSettings, setShowSettings] = useState(false);

  const toggleSettings = () => {
    setShowSettings(!showSettings);
  };

  return (
    <div className="canvas-container">
      {/* Three dots button */}
      <div className="canvas-settings-button" onClick={toggleSettings}>
        &#x22EE; {/* Vertical Ellipsis */}
      </div>

      {/* Settings bar */}
      {showSettings && (
        <div className="canvas-settings-bar">
          <button onClick={() => onCreateCanvas('below')}>Create new canvas below</button>
          <button onClick={() => onCreateCanvas('right')}>Create new canvas on the right</button>
        </div>
      )}

      {/* Display the selected timestamp */}
      <p>Currently selected timestamp: {selectedTimestamp}</p>

      {/* Display the list of topics */}
      <div className="topics-list">
        <h3>Available Topics:</h3>
        <ul>
          {topics.map((topic, index) => (
            <li key={index}>{topic}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default CanvasContainer;
