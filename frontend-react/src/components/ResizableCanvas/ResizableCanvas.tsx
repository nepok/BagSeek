// ResizableCanvas.tsx
import React from 'react';
import { ResizableBox } from 'react-resizable';
import 'react-resizable/css/styles.css';
import ImageDisplay from '../ImageDisplay/ImageDisplay';

interface ResizableCanvasProps {
  image: string | null;
  selectedTimestamp: number | null;
  position: { top: string; left: string };
  onResize: (newSize: { width: number; height: number }, index: number) => void;
  index: number;
  width: number;
  height: number;
}

// Define a custom type for the resize data
interface ResizeCallbackData {
  size: { width: number; height: number };
  node: HTMLElement;
  handle: HTMLElement;
  delta: { x: number; y: number };
}

const ResizableCanvas: React.FC<ResizableCanvasProps> = ({
  image,
  selectedTimestamp,
  position,
  onResize,
  index,
  width,
  height,
}) => {
  // Explicitly type the parameters of the onResizeStop handler
  const handleResizeStop = (e: React.SyntheticEvent, data: ResizeCallbackData) => {
    onResize({ width: data.size.width, height: data.size.height }, index);
  };

  return (
    <div style={{ position: 'absolute', ...position }}>
      <ResizableBox
        width={width}
        height={height}
        axis="both"
        minConstraints={[100, 100]} // Set min size
        onResizeStop={handleResizeStop} // Use the typed handler
        resizeHandles={['se']} // Handle resize from the bottom-right corner
      >
        <ImageDisplay image={image} selectedTimestamp={selectedTimestamp} />
      </ResizableBox>
    </div>
  );
};

export default ResizableCanvas;
