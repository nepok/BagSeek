import React, { useState, useEffect } from 'react';
import { Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, FormControl, IconButton, InputLabel, MenuItem, Select, SelectChangeEvent } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

interface FileInputProps {
  isVisible: boolean;
  onClose: () => void;
  onTopicsUpdate: () => void; // Callback for refreshing topics
  onTimestampsUpdate: () => void; //Callback for refreshing timestamps
}

const FileInput: React.FC<FileInputProps> = ({ isVisible, onClose, onTopicsUpdate, onTimestampsUpdate }) => {
  const [filePaths, setFilePaths] = useState<string[]>([]);
  const [rosbag, setRosbag] = useState<string>('');
  const [selectedPath, setSelectedPath] = useState<string>('');

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
    setRosbag(path);
    setSelectedPath(path); // Set the selected path

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
        console.log('File path updated:', data);

        // Trigger the topics update callback after setting the file path
        onTopicsUpdate();
        onTimestampsUpdate();
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
      <FormControl sx={{ m: 1, width: '100%' }}>
        <InputLabel id="demo-simple-select-helper-label">Select RosBag</InputLabel>
        <Select
          labelId="demo-simple-select-helper-label"
          id="demo-simple-select-helper"
          value={rosbag}
          label="Select File"
          onChange={handleChange}
          sx={{
            width: '100%',
            whiteSpace: 'nowrap', // Prevent text wrapping
            overflow: 'hidden', // Hide overflow text
            textOverflow: 'ellipsis', // Show ellipsis when text overflows
          }}
        >
          {filePaths.map((path, index) => (
            <MenuItem
              key={index}
              value={path}
              sx={{
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                width: '100%',
              }}
            >
              {path}
            </MenuItem>
          ))}
        </Select>
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