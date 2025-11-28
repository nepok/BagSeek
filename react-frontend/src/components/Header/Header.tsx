import React, { useEffect, useRef, useState } from 'react';
import { IconButton, Typography, Box, Tooltip, Popper, Paper, MenuItem, TextField, Button, ButtonGroup } from '@mui/material';
import './Header.css';
import FolderIcon from '@mui/icons-material/Folder';
import IosShareIcon from '@mui/icons-material/IosShare';
import ViewQuiltRoundedIcon from '@mui/icons-material/ViewQuiltRounded';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import { useLocation, useNavigate } from 'react-router-dom';

interface HeaderProps {
  setIsFileInputVisible: (visible: boolean | ((prev: boolean) => boolean)) => void; // Controls visibility of file input dialog
  setIsExportDialogVisible: (visible: boolean | ((prev: boolean) => boolean)) => void; // Controls visibility of export dialog
  selectedRosbag: string | null; // Currently selected ROS bag name
  handleLoadCanvas: (name: string) => void; // Callback to load a canvas by name
  handleAddCanvas: (name: string) => void; // Callback to add a new canvas by name
  handleResetCanvas: () => void; // Callback to reset canvas to empty state
  availableTopics: string[]; // List of available topics for the currently selected rosbag
}

// Generates a consistent color based on rosbag name hash for UI elements
const generateColor = (rosbagName: string) => {
  const hash = rosbagName.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const colors = [
    '#ff5733', '#33ff57', '#3357ff', '#ff33a6', '#ffd633', '#33fff5', // Original colors
    '#ff9966', '#66ff99', '#6699ff', '#ff6699', '#ffee66', '#66ffee', // Softer tints
    '#cc4422', '#22cc44', '#2244cc', '#cc2266', '#ccaa22', '#22ccaa', // Darker shades
    '#ff7744', '#44ff77', '#4477ff', '#ff4477', '#ffdd44', '#44ffdd', // Intermediate tones
    '#ffb366', '#66ffb3', '#b366ff', '#ff66b3', '#ffff66', '#66ffff', // More variety
    '#aa3311', '#11aa33', '#1133aa', '#aa1133', '#aa8811', '#11aa88', // Deeper hues
  ];
  return colors[hash % colors.length];
};

// Normalize topic name: replace / with _ and remove leading _
const normalizeTopic = (topic: string): string => {
  return topic.replace(/\//g, '_').replace(/^_/, '');
};

// Extract all topics from a canvas metadata
const extractTopicsFromCanvas = (canvas: { metadata?: { [id: number]: { nodeTopic: string | null } } }): string[] => {
  const topics: string[] = [];
  if (canvas?.metadata) {
    Object.values(canvas.metadata).forEach((meta) => {
      if (meta.nodeTopic) {
        topics.push(meta.nodeTopic);
      }
    });
  }
  return topics;
};

// Check if a canvas is compatible with the currently selected rosbag based on topics
const isCanvasCompatible = (canvas: any, availableTopics: string[]): boolean => {
  const requiredTopics = extractTopicsFromCanvas(canvas);
  if (requiredTopics.length === 0) return false; // Canvas with no topics is not compatible
  
  // Normalize available topics for comparison (canvas topics are already normalized)
  const normalizedAvailableTopics = availableTopics.map(normalizeTopic);
  
  // Check if all required topics (already normalized) exist in available topics
  return requiredTopics.every(topic => normalizedAvailableTopics.includes(topic));
};

// Header component displays app title and controls for file input, canvas management, and export
const Header: React.FC<HeaderProps> = ({ setIsFileInputVisible, setIsExportDialogVisible, selectedRosbag, handleLoadCanvas, handleAddCanvas, handleResetCanvas, availableTopics }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const viewMode: 'explore' | 'search' = location.pathname.startsWith('/search') ? 'search' : 'explore';
  // State to show/hide the canvas selection popper menu
  const [showCanvasPopper, setShowCanvasPopper] = useState(false);
  // List of canvases loaded from backend, storing full canvas data for topic checking
  const [canvasList, setCanvasList] = useState<{ name: string, canvas: any, rosbag: string, color: string }[]>([]);
  // Controls visibility of TextField for adding a new canvas
  const [showTextField, setShowTextField] = useState(false);
  // Holds the user input for new canvas name
  const [newCanvasName, setNewCanvasName] = useState('');
  // Ref for the canvas icon button to anchor the popper menu
  const canvasIconRef = useRef<HTMLButtonElement | null>(null);

  // Fetch canvas list from backend on component mount to populate menu
  const fetchCanvasList = async () => {
    try {
      const response = await fetch('/api/load-canvases');
      const data = await response.json();
      // Map backend data to local state including full canvas data and generated colors
      setCanvasList(Object.keys(data).map((name) => ({
        name,
        canvas: data[name],
        rosbag: data[name].rosbag,
        color: generateColor(data[name].rosbag || ''),
      })));
    } catch (error) {
      console.error('Error fetching canvas list:', error);
    }
  };

  // Initial fetch of canvas list when component mounts
  useEffect(() => {
    fetchCanvasList();
  }, []);

  // Toggle the visibility of the canvas selection popper
  const toggleCanvasOptions = () => {
    setShowCanvasPopper(!showCanvasPopper);
  };

  // Add a new canvas using the entered name, update list and reset input field
  const onAddCanvas = async () => {
    if (!newCanvasName.trim()) return; // Ignore empty names
    handleAddCanvas(newCanvasName);
    // Fetch updated canvas list to get the full canvas data
    await fetchCanvasList();
    setNewCanvasName('');
    setShowTextField(false);
  };
  
  // Load a canvas by name via callback
  const onLoadCanvas = (name: string) => {
    handleLoadCanvas(name);
  };

  // Delete a canvas by name, update state and notify backend
  const onDeleteCanvas = async (name: string) => {
    try {
      await fetch('/api/delete-canvas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      // Refresh canvas list to get updated data
      await fetchCanvasList();
    } catch (error) {
      console.error('Error deleting canvas:', error);
    }
  };
  
  return (
    <Box className="header-container" display="flex" justifyContent="space-between" alignItems="center" padding="8px 16px">
      <Typography variant="h4" className="header-title">
        BagSeek
      </Typography>

      <Box display="flex" alignItems="center">
        {/* Popper menu anchored to canvas icon, lists canvases and add option */}
        <Popper open={showCanvasPopper} anchorEl={canvasIconRef.current} placement="bottom-end" sx={{ zIndex: 10000, width: '300px' }}>
          <Paper sx={{ padding: '8px', background: '#202020', borderRadius: '8px' }}>
            {/* RESET canvas button */}
            <Button
              onClick={() => {
                handleResetCanvas();
                setShowCanvasPopper(false);
              }}
              variant="contained"
              color="primary"
              fullWidth
              sx={{ 
                fontSize: '0.8rem', 
                marginBottom: '8px',
                textTransform: 'none'
              }}
            >
              RESET
            </Button>
            
            {canvasList.map((canvas, index) => {
              const isExactMatch = selectedRosbag === canvas.rosbag;
              const isCompatible = isCanvasCompatible(canvas.canvas, availableTopics);
              const canLoad = isExactMatch || isCompatible;
              
              return (
                <MenuItem 
                  key={index} 
                  style={{ 
                    fontSize: '0.8rem', 
                    padding: '0px 8px', 
                    display: 'flex', 
                    alignItems: 'center',
                    color: canLoad ? 'white' : 'gray' // Gray out if not compatible
                  }}
                  onClick={() => canLoad && onLoadCanvas(canvas.name)} // Load if compatible
                  disabled={!canLoad} // Disable if not compatible
                >
                  {/* Colored circle representing canvas */}
                  <Box sx={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: canvas.color, marginRight: '8px', marginBottom: '3.5px' }} />
                  {canvas.name}
                  {/* Delete button for canvas */}
                  <IconButton onClick={() => onDeleteCanvas(canvas.name)} sx={{ marginLeft: 'auto', color: 'white' }}>
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </MenuItem>
              );
            })}

            {/* TextField shown when adding a new canvas */}
            {showTextField ? (
              <TextField
                autoFocus
                fullWidth
                size="small"
                variant="outlined"
                value={newCanvasName}
                onChange={(e) => setNewCanvasName(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && onAddCanvas()} // Submit on Enter key
                onBlur={() => setShowTextField(false)} // Hide TextField on blur
                sx={{
                  backgroundColor: '#303030',
                  borderRadius: '4px',
                  marginTop: '4px',
                  input: { color: 'white' },
                  '& .MuiOutlinedInput-root': {
                    '& fieldset': { borderColor: '#555' },
                    '&:hover fieldset': { borderColor: '#777' },
                    '&.Mui-focused fieldset': { borderColor: '#aaa' },
                  },
                }}
              />
            ) : (
              // Add button to show TextField for new canvas name
              <MenuItem onClick={() => setShowTextField(true)} sx={{ fontSize: '0.8rem', padding: '0px 8px', display: 'flex', justifyContent: 'center' }}>
                <AddIcon fontSize="small" />
              </MenuItem>
            )}
          </Paper>
        </Popper>

        {viewMode === 'explore' && (
          <>
            {/* Button to toggle file input dialog */}
            <Tooltip title="Open Rosbag" arrow>
              <IconButton 
                className="header-icon" 
                onClick={() => {
                  setShowCanvasPopper(false); // Close canvas popper if open
                  setIsFileInputVisible((prev) => !prev); // Toggle file input visibility
                }}
              >
                <FolderIcon />
              </IconButton>
            </Tooltip>

            {/* Button to toggle canvas popper menu */}
            <Tooltip title="Load/Save Canvas" arrow>
              <IconButton className="header-icon" ref={canvasIconRef} onClick={toggleCanvasOptions}>
                <ViewQuiltRoundedIcon />
              </IconButton>
            </Tooltip>

            {/* Button to toggle export dialog visibility */}
            <Tooltip title="Export Rosbag" arrow>
              <IconButton className="header-icon" onClick={() => setIsExportDialogVisible((prev: boolean) => !prev)}>
                <IosShareIcon />
              </IconButton>
            </Tooltip>
          </>
        )}
        {/* Mode selection button group */}
        <ButtonGroup variant="outlined" sx={{ marginLeft: 2 }}>
          <Button
            variant={viewMode === 'search' ? 'contained' : 'outlined'}
            onClick={() => {
              // Save current explore query so we can restore it later
              if (location.pathname.startsWith('/explore')) {
                try { sessionStorage.setItem('lastExploreSearch', location.search || ''); } catch {}
              }
              navigate('/search');
            }}
          >
            Search
          </Button>
          <Button
            variant={viewMode === 'explore' ? 'contained' : 'outlined'}
            onClick={() => {
              // Cache current search tab before navigating to explore
              if (location.pathname.startsWith('/search')) {
                try { 
                  const cache = (globalThis as any).__BagSeekGlobalSearchCache;
                  if (cache && cache.viewMode) {
                    sessionStorage.setItem('lastSearchTab', cache.viewMode);
                  }
                } catch {}
              }
              let qs = '';
              try { qs = sessionStorage.getItem('lastExploreSearch') || ''; } catch {}
              navigate(`/explore${qs}`);
            }}
          >
            Explore
          </Button>
        </ButtonGroup>
      </Box>
    </Box>
  );
};

export default Header;
