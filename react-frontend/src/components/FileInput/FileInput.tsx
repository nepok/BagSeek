import React, { useState, useEffect } from 'react';
import { Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent } from '@mui/material';

interface FileInputProps {
  isVisible: boolean;
  onClose: () => void;
  onTopicsUpdate: () => void; // Callback for refreshing topics
  onTimestampsUpdate: () => void; //Callback for refreshing timestamps
  onRosbagUpdate: () => void; // Callback for refreshing rosbag
}

const generateColor = (selectedRosbag: string) => {
  // Generate a hash-based color from the rosbag name
  const hash = selectedRosbag.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const colors = ['#ff5733', '#33ff57', '#3357ff', '#ff33a6', '#ffd633', '#33fff5'];
  return colors[hash % colors.length]; // Pick a color based on hash
};

const FileInput: React.FC<FileInputProps> = ({ isVisible, onClose, onTopicsUpdate, onTimestampsUpdate, onRosbagUpdate }) => {
  const [filePaths, setFilePaths] = useState<string[]>([]);
  const [selectedRosbagPath, setSelectedRosbagPath] = useState<string>('');
  
  useEffect(() => {
    if (isVisible) {
      // Fetch file paths from the API when the component becomes visible
      fetch('/api/get-file-paths')
        .then((response) => response.json())
        .then((data) => {
          setFilePaths(data.paths); // Set file paths state
        })
        .catch((error) => {
          console.error('Error fetching file paths:', error);
        });
    }
  }, [isVisible]);

  const handleChange = (event: SelectChangeEvent<string>) => {
    const path = event.target.value;
    setSelectedRosbagPath(path);

    // Post the selected path to the API immediately
    fetch('/api/set-file-paths', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ path }), // Send selected path to the API
    })
      .then((response) => response.json())
      .then((data) => {
        //console.log('File path updated:', data);

        // Trigger the topics update callback after setting the file path
        onTopicsUpdate();
        onTimestampsUpdate();
        onRosbagUpdate();
      })
      .catch((error) => {
        console.error('Error setting file path:', error);
      });
  };

  if (!isVisible) return null;

  return (
    <Dialog 
      open={true} 
      onClose={onClose} 
      aria-labelledby="form-dialog-title"
      fullWidth
      maxWidth="lg" // Adjust the maxWidth as needed
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
              onChange={handleChange}
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
            >
              {filePaths.map((path, index) => (
                <MenuItem
                  key={index}
                  value={path}
                  sx={{
                    display: 'flex',
                    alignItems: 'center', // Ensures items stay on the same height level
                    width: '100%',
                    maxWidth: '100%',
                    gap: 1,
                    overflow: 'hidden',
                  }}
                >
                  <Box
                    sx={{
                      flexShrink: 0, // Prevents the circle from shrinking
                      width: 10,
                      height: 10,
                      borderRadius: '50%',
                      backgroundColor: generateColor(path.split('/').pop() || ''),
                    }}
                  />
                  <Box
                    sx={{
                      flexGrow: 1, // Allows text to take up remaining space
                      minWidth: 0, // <== KEY FIX: Prevents text from forcing a
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
        <Button onClick={onClose} color="primary">
          Apply
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default FileInput;