import React, { useState, useEffect } from 'react';
import { Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent } from '@mui/material';

interface FileInputProps {
  isVisible: boolean; // Controls visibility of the dialog
  onClose: () => void; // Callback to close the dialog
  onAvailableTopicsUpdate: () => void; // Callback for refreshing topics list after file selection
  onAvailableTopicTypesUpdate: () => void; // Callback for refreshing topic types after file selection
  onAvailableTimestampsUpdate: () => void; // Callback for refreshing timestamps after file selection
  onSelectedRosbagUpdate: () => void; // Callback for refreshing selected rosbag state
}

const generateColor = (selectedRosbag: string) => {
  // Generate a consistent color based on rosbag filename hash for UI indication
  const hash = selectedRosbag.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const colors = [
    '#ff5733', '#33ff57', '#3357ff', '#ff33a6', '#ffd633', '#33fff5', // Original colors
    '#ff9966', '#66ff99', '#6699ff', '#ff6699', '#ffee66', '#66ffee', // Softer tints
    '#cc4422', '#22cc44', '#2244cc', '#cc2266', '#ccaa22', '#22ccaa', // Darker shades
    '#ff7744', '#44ff77', '#4477ff', '#ff4477', '#ffdd44', '#44ffdd', // Intermediate tones
    '#ffb366', '#66ffb3', '#b366ff', '#ff66b3', '#ffff66', '#66ffff', // More variety
    '#aa3311', '#11aa33', '#1133aa', '#aa1133', '#aa8811', '#11aa88', // Deeper hues
  ];
  return colors[hash % colors.length]; // Pick a color based on hash index
};

const FileInput: React.FC<FileInputProps> = ({ isVisible, onClose, onAvailableTopicsUpdate, onAvailableTopicTypesUpdate, onAvailableTimestampsUpdate, onSelectedRosbagUpdate }) => {
  // State to hold list of available rosbag file paths fetched from backend
  const [filePaths, setFilePaths] = useState<string[]>([]);
  // State for currently selected rosbag file path from dropdown
  const [selectedRosbagPath, setSelectedRosbagPath] = useState<string>('');
  
  useEffect(() => {
    if (isVisible) {
      // Fetch available rosbag file paths from backend API when dialog becomes visible
      fetch('/api/get-file-paths')
        .then((response) => response.json())
        .then((data) => {
          setFilePaths(data.paths) // Update state with fetched file paths
        })
        .catch((error) => {
          console.error('Error fetching file paths:', error) // Log fetch errors
        });
    }
  }, [isVisible]); // Re-run when dialog visibility changes

  // Update selected rosbag path state when user selects a different option
  const handleRosbagSelection = (event: SelectChangeEvent<string>) => {
    setSelectedRosbagPath(event.target.value);
  };

  // Called when user clicks "Apply" button to confirm selection
  const handleApply = () => {
    if (selectedRosbagPath) {
      handleChange(selectedRosbagPath);  // Apply the selected file path
    }
    onClose(); // Close the dialog after applying selection
  };

  // Send selected file path to backend and trigger update callbacks
  const handleChange = (path: string) => {
    setSelectedRosbagPath(path);

    // POST selected path to backend API to update file path
    fetch('/api/set-file-paths', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ path }), // Send selected path as JSON payload
    })
      .then((response) => response.json())
      .then((data) => {
        // Trigger callbacks to refresh topics, topic types, timestamps, and selected rosbag state
        onAvailableTopicsUpdate();      
        onAvailableTopicTypesUpdate();  
        onAvailableTimestampsUpdate();
        onSelectedRosbagUpdate();
      })
      .catch((error) => {
        console.error('Error setting file path:', error) // Log errors on setting file path
      });
  };

  if (!isVisible) return null; // Do not render dialog if not visible

  return (
    <Dialog 
      open={true} 
      onClose={onClose} 
      aria-labelledby="form-dialog-title"
      fullWidth
      maxWidth="md" // Dialog width control
    >
      <DialogTitle id="form-dialog-title">Select RosBag File</DialogTitle>
      <DialogContent>
        <FormControl sx={{ width: '100%', overflow: 'hidden' }}>
          <InputLabel id="rosbag-select-label"></InputLabel>
          <div style={{ width: '100%', display: 'flex', overflow: 'hidden' }}>
            <Select
              labelId="rosbag-select-label"
              id="rosbag-select"
              value={selectedRosbagPath}
              onChange={handleRosbagSelection} // Handle dropdown selection changes
              sx={{
                width: '100%',
                minWidth: 0,
                maxWidth: '100%',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
              MenuProps={{
                PaperProps: {
                  sx: {
                    maxWidth: '100%',
                    overflowX: 'hidden',
                  },
                },
                sx: {
                  maxWidth: '100%',
                  overflowX: 'hidden',
                },
              }}
              renderValue={(selected) => {
                // Render selected value with colored dot and truncated text
                const color = generateColor(selected.split('/').pop() || '');
                return (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box
                      sx={{
                        width: 10,
                        height: 10,
                        borderRadius: '50%',
                        backgroundColor: color,
                        flexShrink: 0,
                      }}
                    />
                    <Box
                      sx={{
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                    >
                      {selected}
                    </Box>
                  </Box>
                );
              }}
            >
              {/* Render dropdown menu items with colored dot and truncated path */}
              {filePaths.map((path, index) => (
                <MenuItem
                  key={index}
                  value={path}
                  sx={{
                    display: 'flex',
                    alignItems: 'center', // Align dot and text horizontally
                    width: '100%',
                    maxWidth: '100%',
                    gap: 1,
                    overflow: 'hidden',
                  }}
                >
                  <Box
                    sx={{
                      flexShrink: 0, // Prevent dot from shrinking
                      width: 10,
                      height: 10,
                      borderRadius: '50%',
                      backgroundColor: generateColor(path.split('/').pop() || ''),
                    }}
                  />
                  <Box
                    sx={{
                      flexGrow: 1, // Allow text to fill remaining space
                      minWidth: 0, // Prevent text from forcing width expansion
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {path}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </div>
        </FormControl>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">
          Cancel
        </Button>
        <Button onClick={handleApply} color="primary">
          Apply
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default FileInput;