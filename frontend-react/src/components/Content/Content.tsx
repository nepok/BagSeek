// components/Content/Content.tsx
import React from 'react';
import CanvasContainer from '../CanvasContainer/CanvasContainer';

interface Container {
  id: number;
  width: number;
  height: number;
}

interface ContentProps {
  containers: Container[];
  selectedTimestamp: number | null;
  handleCreateCanvas: (id: number, direction: 'below' | 'right') => void;
  topics: string[];
}

const Content: React.FC<ContentProps> = ({
  containers,
  selectedTimestamp,
  handleCreateCanvas,
  topics,
}) => {
  return (
    <div
      className="app-container"
      style={{ display: 'flex', flexWrap: 'wrap', height: '100%', width: '100%' }}
    >
      {containers.map((container) => (
        <div
          key={container.id}
          className="container"
          style={{
            flexBasis: `${container.width}%`, 
            height: `${container.height}%`,
            border: '1px solid #ccc',
            boxSizing: 'border-box',
          }}
        >
          <CanvasContainer
            selectedTimestamp={selectedTimestamp}
            onCreateCanvas={(dir) => handleCreateCanvas(container.id, dir)}
            topics={topics} // Pass the topics prop to CanvasContainer
          />
        </div>
      ))}
    </div>
  );
};

export default Content;
