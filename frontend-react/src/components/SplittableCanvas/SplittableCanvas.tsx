import React, { useState, useRef } from 'react';
import './SplittableCanvas.css';

interface Node {
  id: number;
  direction?: 'horizontal' | 'vertical'; // Direction of split
  left?: Node; // Left child
  right?: Node; // Right child
  size?: number; // Size of the left or top pane (percentage)
}

function SplittableCanvas() {
  const [root, setRoot] = useState<Node>({ id: 1 });
  const resizingNode = useRef<Node | null>(null);

  // Split node into left and right (or top and bottom) panes
  const splitNode = (node: Node, direction: 'horizontal' | 'vertical') => {
    if (!node.left && !node.right) {
      node.direction = direction;
      node.size = 50; // Initial size of left/top pane (percentage)
      node.left = { id: node.id * 2 };
      node.right = { id: node.id * 2 + 1 };
      setRoot({ ...root }); // Trigger re-render
    }
  };

  // Start resizing operation
  const startResizing = (e: React.MouseEvent, node: Node) => {
    resizingNode.current = node;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', stopResizing);
  };

  // Handle mouse movement during resizing
  const handleMouseMove = (e: MouseEvent) => {
    if (!resizingNode.current || !resizingNode.current.direction || resizingNode.current.size == null) return;

    const container = document.querySelector('.splittable-canvas') as HTMLDivElement;
    const rect = container.getBoundingClientRect();
    const mousePosition =
      resizingNode.current.direction === 'horizontal' ? e.clientX - rect.left : e.clientY - rect.top;
    const totalSize =
      resizingNode.current.direction === 'horizontal' ? rect.width : rect.height;

    let newSize = (mousePosition / totalSize) * 100;
    newSize = Math.max(10, Math.min(90, newSize)); // Constrain sizes between 10% and 90%

    resizingNode.current.size = newSize;
    setRoot({ ...root });
  };

  // Stop resizing operation
  const stopResizing = () => {
    resizingNode.current = null;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', stopResizing);
  };

  // Render each node, either splitting it further or rendering the split panes
  const renderNode = (node: Node) => {
    if (!node.left && !node.right) {
      return (
        <div className="canvas-node">
          <button onClick={() => splitNode(node, 'horizontal')}>Split Horizontally</button>
          <button onClick={() => splitNode(node, 'vertical')}>Split Vertically</button>
        </div>
      );
    }

    return (
      <div
        key={node.id}
        className="canvas-container"
        style={{
          display: 'flex',
          flexDirection: node.direction === 'horizontal' ? 'row' : 'column', // Adjust layout direction
          position: 'relative',
          height: '100%', // Ensure full height for each pane
          width: '100%', // Ensure full width for each pane
        }}
      >
        {/* Pane 1 */}
        <div
          className="canvas-pane"
          style={{
            flexBasis: `${node.size}%`,
            display: 'flex',
            height: '100%', // Ensure full height for each pane
            width: '100%',
          }}
        >
          {node.left && renderNode(node.left)}
        </div>

        {/* Resizer */}
        <div
          className="canvas-resizer"
          style={{
            backgroundColor: '#ccc',
            zIndex: 10,
            position: 'relative',
            // Dynamically set the resizer's size based on direction
            width: node.direction === 'horizontal' ? '10px' : '100%', // Vertical resizer for horizontal split
            height: node.direction === 'vertical' ? '10px' : '100%', // Horizontal resizer for vertical split
            cursor: node.direction === 'horizontal' ? 'col-resize' : 'row-resize', // Adjust cursor
          }}
          onMouseDown={(e) => startResizing(e, node)} // Start resizing on mouse down
        />

        {/* Pane 2 */}
        <div
          className="canvas-pane"
          style={{
            flexBasis: `${100 - (node.size || 50)}%`,
            display: 'flex',
            height: '100%',
            width: '100%',
          }}
        >
          {node.right && renderNode(node.right)}
        </div>
      </div>
    );
  };

  return <div className="splittable-canvas">{renderNode(root)}</div>;
}

export default SplittableCanvas;