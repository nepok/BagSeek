import React from 'react';

interface CanvasContainerProps {
  selectedTimestamp: number | null;
}

const CanvasContainer: React.FC<CanvasContainerProps> = ({ selectedTimestamp }) => {
  return (
    <div style={{ width: '100%', height: '100%', backgroundColor: '#f0f0f0' }}>
      <p>Currently selected timestamp: {selectedTimestamp}</p>
      {/* Add more content or canvas-related logic as needed */}
    </div>
  );
};

export default CanvasContainer;
