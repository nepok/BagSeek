import React, { useEffect, useRef, useState } from 'react';
import { IconButton, Typography, Box, Tooltip, Popper, Paper, MenuItem, TextField } from '@mui/material';
import './Header.css';
import FolderIcon from '@mui/icons-material/Folder';
import IosShareIcon from '@mui/icons-material/IosShare';
import SettingsIcon from '@mui/icons-material/Settings';
import ViewQuiltRoundedIcon from '@mui/icons-material/ViewQuiltRounded';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';

interface HeaderProps {
  setIsFileInputVisible: (visible: boolean | ((prev: boolean) => boolean)) => void;
  setIsExportDialogVisible: (visible: boolean | ((prev: boolean) => boolean)) => void;
  selectedRosbag: string | null; // The currently selected ROS bag
  handleLoadCanvas: (name: string) => void;
  handleAddCanvas: (name: string) => void;
}

const generateColor = (rosbagName: string) => {
  // Generate a hash-based color from the rosbag name
  const hash = rosbagName.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const colors = ['#ff5733', '#33ff57', '#3357ff', '#ff33a6', '#ffd633', '#33fff5'];
  return colors[hash % colors.length]; // Pick a color based on hash
};

const Header: React.FC<HeaderProps> = ({ setIsFileInputVisible, setIsExportDialogVisible, selectedRosbag, handleLoadCanvas, handleAddCanvas }) => {
  const [showCanvasPopper, setShowCanvasPopper] = useState(false);
  const [canvasList, setCanvasList] = useState<{ name: string, rosbag: string, color: string }[]>([]);
  const [showTextField, setShowTextField] = useState(false);
  const [newCanvasName, setNewCanvasName] = useState('');
  const canvasIconRef = useRef<HTMLButtonElement | null>(null);

  const fetchCanvasList = async () => {
    try {
      const response = await fetch('/api/load-canvases');
      const data = await response.json();
      
      setCanvasList(Object.keys(data).map((name) => ({
        name,
        rosbag: data[name].rosbag, // Store the rosbag name
        color: generateColor(data[name].rosbag || ''),
      })));
    } catch (error) {
      console.error('Error fetching canvas list:', error);
    }
  };

  useEffect(() => {
    fetchCanvasList();
  }, []);

  const toggleCanvasOptions = () => {
    setShowCanvasPopper(!showCanvasPopper);
  };

  const onAddCanvas = async () => {
    if (!newCanvasName.trim()) return;
    handleAddCanvas(newCanvasName);
    setCanvasList([...canvasList, { name: newCanvasName, rosbag: selectedRosbag || '', color: generateColor(selectedRosbag || '') }]);
    setNewCanvasName('');
    setShowTextField(false);
  };
  

  const onLoadCanvas = (name: string) => {
    handleLoadCanvas(name);
  };

  const onDeleteCanvas = async (name: string) => {
    const updatedCanvasList = canvasList.filter((canvas) => canvas.name !== name); // Compare canvas.name to name
    setCanvasList(updatedCanvasList);
  
    try {
      await fetch('/api/delete-canvas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
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
        <Popper open={showCanvasPopper} anchorEl={canvasIconRef.current} placement="bottom-end" sx={{ zIndex: 100, width: '300px' }}>
          <Paper sx={{ padding: '8px', background: '#202020', borderRadius: '8px' }}>
            {canvasList.map((canvas, index) => (
              <MenuItem 
              key={index} 
              style={{ 
                fontSize: '0.8rem', 
                padding: '0px 8px', 
                display: 'flex', 
                alignItems: 'center',
                color: selectedRosbag === canvas.rosbag ? 'white' : 'gray' // Gray out if not matching
              }}
              onClick={() => selectedRosbag === canvas.rosbag && onLoadCanvas(canvas.name)}
              disabled={selectedRosbag !== canvas.rosbag} // Disable if rosbag doesn't match
            >
              {/* Color Circle */}
              <Box sx={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: canvas.color, marginRight: '8px' }} />
              {canvas.name}
              <IconButton onClick={() => onDeleteCanvas(canvas.name)} sx={{ marginLeft: 'auto', color: 'white' }}>
                <DeleteIcon fontSize="small" />
              </IconButton>
            </MenuItem>
            ))}

            {showTextField ? (
              <TextField
                autoFocus
                fullWidth
                size="small"
                variant="outlined"
                value={newCanvasName}
                onChange={(e) => setNewCanvasName(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && onAddCanvas()}
                onBlur={() => setShowTextField(false)}
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
              <MenuItem onClick={() => setShowTextField(true)} sx={{ fontSize: '0.8rem', padding: '0px 8px', display: 'flex', justifyContent: 'center' }}>
                <AddIcon fontSize="small" />
              </MenuItem>
            )}
          </Paper>
        </Popper>

        <Tooltip title="Load/Save Canvas" arrow>
          <IconButton className="header-icon" ref={canvasIconRef} onClick={toggleCanvasOptions}>
            <ViewQuiltRoundedIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Open Rosbag" arrow>
          <IconButton 
            className="header-icon" 
            onClick={() => {
              setShowCanvasPopper(false); // Close canvas popper
              setIsFileInputVisible((prev) => !prev); // Toggle file input
            }}
          >
            <FolderIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Export Rosbag" arrow>
          <IconButton className="header-icon" onClick={() => setIsExportDialogVisible((prev: boolean) => !prev)}>
            <IosShareIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="Settings" arrow>
          <IconButton className="header-icon">
            <SettingsIcon />
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  );
};

export default Header;