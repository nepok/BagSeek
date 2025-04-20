import React, { useState, useRef, useEffect } from 'react';
import { Box, IconButton, Popper, Paper, MenuItem } from '@mui/material';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import './SplittableCanvas.css';
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
  selectedRosbag: string | null;
  onCanvasChange: (root: Node, metadata: { [id: number]: NodeMetadata }) => void;
  currentRoot: Node | null;
  currentMetadata: { [id: number]: NodeMetadata };
}

const SplittableCanvas: React.FC<SplittableCanvasProps> = ({ topics, selectedTimestamp, selectedRosbag, onCanvasChange, currentRoot, currentMetadata }) => {
  
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

  useEffect(() => {
    setNodeMetadata((prev) => {
      const updatedMetadata = { ...prev };
      Object.keys(updatedMetadata).forEach((key) => {
        updatedMetadata[parseInt(key)].topic = null;
      });
      return updatedMetadata;
    });
  }, [selectedRosbag]);

  useEffect(() => {
    if (!currentRoot || !currentMetadata) return;
  
    setRoot((prevRoot) => {
      return JSON.stringify(prevRoot) === JSON.stringify(currentRoot) ? prevRoot : currentRoot;
    });
  
    setNodeMetadata((prevMetadata) => {
      return JSON.stringify(prevMetadata) === JSON.stringify(currentMetadata) ? prevMetadata : currentMetadata;
    });
  }, [currentRoot, currentMetadata]);

  //useEffect(() => {
  //  onCanvasChange(root, nodeMetadata);
  //}, [root, nodeMetadata]);

  const updateNodeInTree = (currentNode: Node, updatedNode: Node): Node => {
    if (!currentNode) return updatedNode;
    if (currentNode.id === updatedNode.id) return updatedNode;
    
    return {
      ...currentNode,
      left: currentNode.left ? updateNodeInTree(currentNode.left, updatedNode) : undefined,
      right: currentNode.right ? updateNodeInTree(currentNode.right, updatedNode) : undefined,
    };
  };

  const splitNode = (node: Node, direction: 'horizontal' | 'vertical') => {
    if (!node.left && !node.right) {
      const newLeft = { id: node.id * 2 };
      const newRight = { id: node.id * 2 + 1 };

      const updatedNode = {
        ...node,
        direction,
        size: 50,
        left: newLeft,
        right: newRight
      };

      const updatedRoot = updateNodeInTree(root, updatedNode);
      setRoot(updatedRoot);

      setNodeMetadata((prev) => {
        const newMetadata = { ...prev };

        // Remove old metadata of the split node
        delete newMetadata[node.id];

        // Add new nodes with empty metadata
        newMetadata[newLeft.id] = { topic: null, timestamp: null };
        newMetadata[newRight.id] = { topic: null, timestamp: null };

        return newMetadata;
      });
    }
  };

  const deleteNode = (nodeToDelete: Node): Node | null => {
    const traverseAndDelete = (node: Node | null): Node | undefined => {
      if (!node) return undefined;
      if (node.left && node.left.id === nodeToDelete.id) {
        return node.right || undefined;
      }
      if (node.right && node.right.id === nodeToDelete.id) {
        return node.left || undefined;
      }
      return {
        ...node,
        left: traverseAndDelete(node.left || null),
        right: traverseAndDelete(node.right || null),
      };
    };

    if (root.id === nodeToDelete.id) {
      return null; // If the root is deleted, return undefined
    }

    return traverseAndDelete(root as Node) || root;
  };

  const handleDeletePanel = () => {
    if (currentNode) {
      const newRoot = deleteNode(currentNode);
      if (newRoot) {
        setRoot(newRoot);
      } else {
        setRoot({ id: 1 }); // Reset to a new root if the entire tree is deleted
      }

      setNodeMetadata((prev) => {
        const newMetadata = { ...prev };
        const removeMetadata = (node: Node) => {
          delete newMetadata[node.id];
          if (node.left) removeMetadata(node.left);
          if (node.right) removeMetadata(node.right);
        };
        removeMetadata(currentNode);
        return newMetadata;
      });

      handleCloseMenu();
    }
  };

  const startResizing = (e: React.MouseEvent, node: Node) => {
    document.body.style.userSelect = 'none';
    resizingNode.current = node;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', stopResizing);
  };

  const stopResizing = () => {
    document.body.style.userSelect = '';
    resizingNode.current = null;
    onCanvasChange(root, nodeMetadata);

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
    console.log(root);
    setRoot({ ...root });
  };

  const handleClickMenu = (event: React.MouseEvent<HTMLElement>, node: Node) => {
    if (currentNode?.id === node.id) {
      // If the same node is clicked again, close the menu
      handleCloseMenu();
    } else {
      setAnchorEl(event.currentTarget);
      setCurrentNode(node);
    }
  };

  const handleCloseMenu = () => {
    setAnchorEl(null);
    setCurrentNode(null);
    setTopicMenuAnchorEl(null); // Close the topic menu as well
  };

  const handleSplitAction = (direction: 'horizontal' | 'vertical') => {
    if (currentNode) {
      splitNode(currentNode, direction);
      const currentNodeMetadata = nodeMetadata[currentNode.id] || { topic: null, timestamp: null };
      setNodeMetadata((prev) => ({
        ...prev,
        [currentNode.id * 2]: { ...currentNodeMetadata },
        [currentNode.id * 2 + 1]: { topic: null, timestamp: null },
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
        <div className="canvas-node" style={{ position: 'relative', display: 'flex', flexDirection: 'column', height: '100%', width: '100%' }}>
          <NodeContent 
            topic={metadata.topic} 
            timestamp={metadata.timestamp}
            selectedRosbag={selectedRosbag}
          />
          <IconButton
            size="small"
            onClick={(e) => handleClickMenu(e, node)}
            sx={{ position: 'absolute', top: 5, right: 5, padding: 0.5 }}
          >
            <MoreVertIcon fontSize="small" />
          </IconButton>

          <Popper
            open={Boolean(anchorEl) && currentNode?.id === node.id}
            anchorEl={anchorEl}
            placement="bottom-start"
            style={{ zIndex: 1300 }}
          >
            <Paper>
              <MenuItem
                onClick={handleChooseTopic}
                style={{
                  fontSize: "0.8rem",
                  padding: "6px 12px",
                  display: "flex",
                  alignItems: "center",
                  gap: "5px",
                }}
              >
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  width="20"
                  height="20" 
                  viewBox="75 -880 960 960" 
                  fill="#e8eaed"
                  style={{ display: "block" }}
                >
                  <path d="M288-600v-72h528v72H288Zm0 156v-72h528v72H288Zm0 156v-72h528v72H288ZM180-600q-14 0-25-11t-11-25.5q0-14.5 11-25t25.5-10.5q14.5 0 25 10.35T216-636q0 14-10.35 25T180-600Zm0 156q-14 0-25-11t-11-25.5q0-14.5 11-25t25.5-10.5q14.5 0 25 10.35T216-480q0 14-10.35 25T180-444Zm0 156q-14 0-25-11t-11-25.5q0-14.5 11-25t25.5-10.5q14.5 0 25 10.35T216-324q0 14-10.35 25T180-288Z"/>
                </svg>
                Choose Topic
              </MenuItem>
              <MenuItem
                onClick={() => handleSplitAction("horizontal")}
                style={{
                  fontSize: "0.8rem",
                  padding: "6px 12px",
                  display: "flex",
                  alignItems: "center",
                  gap: "5px",
                }}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="20"
                  height="20"
                  viewBox="0 0 29 29"
                  fill="#e8eaed"
                  style={{ display: "block" }}
                >
                  <path d="M18 16v-3h-3v9h-2V2h2v9h3V8l4 4-4 4M2 12l4 4v-3h3v9h2V2H9v9H6V8l-4 4z"></path>
                </svg>
                Split Horizontally
              </MenuItem>
              <MenuItem
                onClick={() => handleSplitAction("vertical")}
                style={{
                  fontSize: "0.8rem",
                  padding: "6px 12px",
                  display: "flex",
                  alignItems: "center",
                  gap: "5px",
                }}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="20"
                  height="20"
                  viewBox="0 0 29 29"
                  fill="#e8eaed"
                  style={{ display: "block" }}
                >
                  <path d="M8 18h3v-3H2v-2h20v2h-9v3h3l-4 4-4-4m4-16L8 6h3v3H2v2h20V9h-9V6h3l-4-4z"></path>
                </svg>
                Split Vertically
              </MenuItem>
              <MenuItem
                onClick={handleDeletePanel}
                style={{
                  fontSize: "0.8rem",
                  padding: "6px 12px",
                  display: "flex",
                  alignItems: "center",
                  gap: "5px",
                }}
              >
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  width="20" 
                  height="20" 
                  viewBox="75 -880 960 960" 
                  fill="#e8eaed"
                  style={{ display: "block" }}
                >
                  <path d="M312-144q-29.7 0-50.85-21.15Q240-186.3 240-216v-480h-48v-72h192v-48h192v48h192v72h-48v479.57Q720-186 698.85-165T648-144H312Zm336-552H312v480h336v-480ZM384-288h72v-336h-72v336Zm120 0h72v-336h-72v336ZM312-696v480-480Z"/>
                </svg>
                Delete Panel
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
                  offset: [0, 5],
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
};

export default SplittableCanvas;