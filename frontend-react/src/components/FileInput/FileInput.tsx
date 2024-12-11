import React, { useState, useEffect } from 'react';
import { Box, FormControl, InputLabel, MenuItem, Select, SelectChangeEvent } from '@mui/material';

interface FileInputProps {
  isVisible: boolean;
  onClose: () => void;
}

const FileInput: React.FC<FileInputProps> = ({ isVisible, onClose }) => {
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
      })
      .catch((error) => {
        console.error('Error updating file path:', error);
      });
  };

  if (!isVisible) return null;

  return (
    <Box
      sx={{
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        color: '#fff',
        padding: '20px',
        borderRadius: '8px',
        zIndex: 1000,
        width: '300px', // Optional: Set a fixed width for consistency
      }}
    >
      <FormControl sx={{ m: 1, minWidth: 200 }}>
        <InputLabel id="demo-simple-select-helper-label">Select File</InputLabel>
        <Select
          labelId="demo-simple-select-helper-label"
          id="demo-simple-select-helper"
          value={rosbag}
          label="Select File"
          onChange={handleChange}
          sx={{
            maxWidth: '100%', // Apply max width to prevent overflow
            whiteSpace: 'nowrap', // Prevent text wrapping
            overflow: 'hidden', // Hide overflow text
            textOverflow: 'ellipsis', // Show ellipsis when text overflows
          }}
        >
          <MenuItem value="">
            <em>None</em>
          </MenuItem>
          {filePaths.map((path, index) => (
            <MenuItem key={index} value={path} sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {path}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </Box>
  );
};

export default FileInput;