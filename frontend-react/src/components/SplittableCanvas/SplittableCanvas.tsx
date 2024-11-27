import React, { useState, useRef } from 'react';
import { Box, IconButton, Popper, Paper, MenuItem } from '@mui/material';
import MoreVertIcon from '@mui/icons-material/MoreVert'; // Import the three dots icon
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
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null); // For the settings menu
  const [currentNode, setCurrentNode] = useState<Node | null>(null); // Current node for the menu
  const [topicMenuAnchorEl, setTopicMenuAnchorEl] = useState<null | HTMLElement>(null); // For the topic menu
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

  // Handle the opening of the settings menu
  const handleClickMenu = (event: React.MouseEvent<HTMLElement>, node: Node) => {
    setAnchorEl(event.currentTarget);
    setCurrentNode(node);
  };

  // Handle the closing of the settings menu
  const handleCloseMenu = () => {
    setAnchorEl(null);
    setCurrentNode(null);
  };

  // Handle the split actions from the menu
  const handleSplitAction = (direction: 'horizontal' | 'vertical') => {
    if (currentNode) {
      splitNode(currentNode, direction);
    }
    handleCloseMenu();
  };

  // Open topic selection menu
  const handleChooseTopic = (event: React.MouseEvent<HTMLElement>) => {
    setTopicMenuAnchorEl(event.currentTarget); // Set the anchor of the topic menu
    // Don't close the first menu when opening the topic menu
  };

  // Handle the selection of a topic
  const handleTopicSelection = () => {
    setTopicMenuAnchorEl(null); // Close the topic menu
    handleCloseMenu(); // Close the first menu
  };

  // Render each node, either splitting it further or rendering the split panes
  const renderNode = (node: Node) => {
    if (!node.left && !node.right) {
      return (
        <div className="canvas-node" style={{ position: 'relative' }}>
          {/* Settings button in the top-right corner */}
          <IconButton
            size='small'
            onClick={(e) => handleClickMenu(e, node)}
            sx={{ position: 'absolute', top: 5, right: 5, padding: 0.5 }} // Reduce the size of the button
          >
            <MoreVertIcon fontSize="small" /> {/* Smaller icon */}
          </IconButton>

          {/* Main menu Popper */}
          <Popper
            open={Boolean(anchorEl) && currentNode?.id === node.id}
            anchorEl={anchorEl}
            placement="bottom-start"
            style={{
              zIndex: 1300, // Ensure it is above other elements
            }}
          >
            <Paper>
              <MenuItem onClick={handleChooseTopic}>Choose Topic</MenuItem>
              <MenuItem onClick={() => handleSplitAction('horizontal')}>Split Horizontally</MenuItem>
              <MenuItem onClick={() => handleSplitAction('vertical')}>Split Vertically</MenuItem>
            </Paper>
          </Popper>

          {/* Topic selection Popper */}
          <Popper
            open={Boolean(topicMenuAnchorEl)}
            anchorEl={topicMenuAnchorEl}
            placement="left-start"
            modifiers={[
              {
                name: 'offset',
                options: {
                  offset: [0, 5], // Add 5px of vertical space between the menus (adjust the 5px value as needed)
                },
              },
            ]}
            style={{
              zIndex: 1300, // Ensure it is above other elements
            }}
          >
            <Paper>
              <MenuItem onClick={handleTopicSelection}>Topic 1</MenuItem>
              <MenuItem onClick={handleTopicSelection}>Topic 2</MenuItem>
              <MenuItem onClick={handleTopicSelection}>Topic 3</MenuItem>
            </Paper>
          </Popper>
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
          position: 'relative', // Ensure the node has relative positioning
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
            backgroundColor: '#202020',
            zIndex: 10,
            position: 'relative',
            // Dynamically set the resizer's size based on direction
            width: node.direction === 'horizontal' ? '2px' : '100%', // Vertical resizer for horizontal split
            height: node.direction === 'vertical' ? '2px' : '100%', // Horizontal resizer for vertical split
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

  return (
    <Box className="splittable-canvas" sx={{ backgroundColor: 'background.default', height: '100vh' }}>
      {renderNode(root)}
    </Box>
  );
}

export default SplittableCanvas;