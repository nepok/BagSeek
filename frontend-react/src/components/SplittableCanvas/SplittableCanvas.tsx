import React, { useState, useRef, useEffect } from 'react';
import { Box, IconButton, Popper, Paper, MenuItem, Typography } from '@mui/material';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import './SplittableCanvas.css';
import { ConstructionOutlined } from '@mui/icons-material';
import NodeContent from '../NodeContent/NodeContent';

interface Node {
  id: number;
  direction?: 'horizontal' | 'vertical';
  left?: Node;
  right?: Node;
  size?: number;
}

interface NodeMetadata {
  topic: string | null;
  timestamp: number | null;
}

interface SplittableCanvasProps {
  topics: string[];
  selectedTimestamp: number | null;
}

function SplittableCanvas({ topics, selectedTimestamp }: SplittableCanvasProps) {
  const [root, setRoot] = useState<Node>({ id: 1 });
  const [nodeMetadata, setNodeMetadata] = useState<{ [id: number]: NodeMetadata }>({});
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [currentNode, setCurrentNode] = useState<Node | null>(null);
  const [topicMenuAnchorEl, setTopicMenuAnchorEl] = useState<null | HTMLElement>(null);
  const resizingNode = useRef<Node | null>(null);

  useEffect(() => {
    setNodeMetadata((prev) =>
      Object.fromEntries(
        Object.entries(prev).map(([id, metadata]) => [
          id,
          { ...metadata, timestamp: selectedTimestamp },
        ])
      )
    );
  }, [selectedTimestamp]);

  const splitNode = (node: Node, direction: 'horizontal' | 'vertical') => {
    if (!node.left && !node.right) {
      node.direction = direction;
      node.size = 50;
      node.left = { id: node.id * 2 };
      node.right = { id: node.id * 2 + 1 };
      setRoot({ ...root });
    }
  };

  const startResizing = (e: React.MouseEvent, node: Node) => {
    // Disable text selection
    document.body.style.userSelect = 'none';
  
    resizingNode.current = node;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', stopResizing);
  };
  
  const stopResizing = () => {
    // Re-enable text selection
    document.body.style.userSelect = '';
  
    resizingNode.current = null;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', stopResizing);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!resizingNode.current || !resizingNode.current.direction || resizingNode.current.size == null) return;

    const container = document.querySelector('.splittable-canvas') as HTMLDivElement;
    const rect = container.getBoundingClientRect();
    const mousePosition =
      resizingNode.current.direction === 'horizontal' ? e.clientX - rect.left : e.clientY - rect.top;
    const totalSize =
      resizingNode.current.direction === 'horizontal' ? rect.width : rect.height;

    let newSize = (mousePosition / totalSize) * 100;
    newSize = Math.max(10, Math.min(90, newSize));

    resizingNode.current.size = newSize;
    setRoot({ ...root });
  };

  const handleClickMenu = (event: React.MouseEvent<HTMLElement>, node: Node) => {
    // Close all menus if a different node is clicked
    if (currentNode?.id !== node.id) {
      setAnchorEl(null);          // Close the main menu
      setTopicMenuAnchorEl(null); // Close the topic menu
    }
    setAnchorEl(event.currentTarget); // Open the main menu for the new node
    setCurrentNode(node);             // Set the current node
  };

  const handleCloseMenu = () => {
    setAnchorEl(null);
    setCurrentNode(null);
  };

  const handleSplitAction = (direction: 'horizontal' | 'vertical') => {
    if (currentNode) {
      // Copy the metadata of the current node to the left child on split
      splitNode(currentNode, direction);
      const currentNodeMetadata = nodeMetadata[currentNode.id] || { topic: null, timestamp: null };
      setNodeMetadata((prev) => ({
        ...prev,
        [currentNode.id * 2]: { ...currentNodeMetadata }, // Left child inherits metadata
        [currentNode.id * 2 + 1]: { topic: null, timestamp: null }, // Right child starts fresh
      }));
    }
    handleCloseMenu();
  };

  const handleChooseTopic = (event: React.MouseEvent<HTMLElement>) => {
    setTopicMenuAnchorEl(event.currentTarget);
  };

  const handleTopicSelection = (topic: string) => {
    if (currentNode) {
      setNodeMetadata((prev) => ({
        ...prev,
        [currentNode.id]: { topic, timestamp: selectedTimestamp },
      }));
    }
    setTopicMenuAnchorEl(null);
    handleCloseMenu();
  };

  const renderNode = (node: Node) => {
    const metadata = nodeMetadata[node.id] || { topic: null, timestamp: null };
  
    if (!node.left && !node.right) {
      return (
        <div className="canvas-node" style={{ position: 'relative', display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
          {/* Render NodeContent dynamically */}
          <div>
            {metadata.topic ? (
              <NodeContent topic={metadata.topic} timestamp={metadata.timestamp} />
            ) : (
              <p style={{ color: "white", fontSize: "0.8rem" }}>No topic selected</p>
            )}
          </div>
  
          <IconButton
            size="small"
            onClick={(e) => handleClickMenu(e, node)}
            sx={{ position: 'absolute', top: 5, right: 5, padding: 0.5 }}
          >
            <MoreVertIcon fontSize="small" />
          </IconButton>
  
          {/* Popper menus remain unchanged */}
          <Popper
            open={Boolean(anchorEl) && currentNode?.id === node.id}
            anchorEl={anchorEl}
            placement="bottom-start"
            style={{ zIndex: 1300 }}
          >
            <Paper>
              <MenuItem
                onClick={handleChooseTopic}
                style={{ fontSize: '0.8rem', padding: '4px 8px' }}
              >
                Choose Topic
              </MenuItem>
              <MenuItem
                onClick={() => handleSplitAction('horizontal')}
                style={{ fontSize: '0.8rem', padding: '4px 8px' }}
              >
                Split Horizontally
              </MenuItem>
              <MenuItem
                onClick={() => handleSplitAction('vertical')}
                style={{ fontSize: '0.8rem', padding: '4px 8px' }}
              >
                Split Vertically
              </MenuItem>
            </Paper>
          </Popper>
  
          <Popper
            open={Boolean(topicMenuAnchorEl)}
            anchorEl={topicMenuAnchorEl}
            placement="left-start"
            style={{ zIndex: 1300 }}
            modifiers={[
              {
                name: 'offset',
                options: {
                  offset: [0, 5], // Add 5px of vertical space between the menus
                },
              },
            ]}
          >
            <Paper>
              {topics.map((topic) => (
                <MenuItem
                  key={topic}
                  onClick={() => handleTopicSelection(topic)}
                  style={{ fontSize: '0.8rem', padding: '4px 8px' }}
                >
                  {topic}
                </MenuItem>
              ))}
            </Paper>
          </Popper>
        </div>
      );
    }
  
    // Render for split nodes remains unchanged
    return (
      <div
        key={node.id}
        className="canvas-container"
        style={{
          display: 'flex',
          flexDirection: node.direction === 'horizontal' ? 'row' : 'column',
          position: 'relative',
          height: '100%',
          width: '100%',
        }}
      >
        <div
          className="canvas-pane"
          style={{
            flexBasis: `${node.size}%`,
            display: 'flex',
            height: '100%',
            width: '100%',
          }}
        >
          {node.left && renderNode(node.left)}
        </div>
        <div
          className="canvas-resizer"
          style={{
            backgroundColor: '#202020',
            zIndex: 10,
            position: 'relative',
            width: node.direction === 'horizontal' ? '2px' : '100%',
            height: node.direction === 'vertical' ? '2px' : '100%',
            cursor: node.direction === 'horizontal' ? 'col-resize' : 'row-resize',
          }}
          onMouseDown={(e) => startResizing(e, node)}
        />
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