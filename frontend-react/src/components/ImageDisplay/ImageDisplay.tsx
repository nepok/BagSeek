// ImageDisplay.tsx
import React from 'react';

interface ImageDisplayProps {
  image: string | null;
  selectedTimestamp: number | null;
}

const ImageDisplay: React.FC<ImageDisplayProps> = ({ image, selectedTimestamp }) => {
  return (
    <div style={{ flex: 1, padding: '20px' }}>
      {image ? (
        <div>
          <h2>Image for Timestamp: {selectedTimestamp}</h2>
          <img
            src={`data:image/png;base64,${image}`}
            alt={`Image at timestamp ${selectedTimestamp}`}
            style={{ maxWidth: '100%', height: 'auto' }}
          />
        </div>
      ) : (
        <p>Loading image...</p>
      )}
    </div>
  );
};

export default ImageDisplay;
