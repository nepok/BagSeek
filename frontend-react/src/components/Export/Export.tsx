import React, { useState } from 'react';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField, FormControl, InputLabel, Select, MenuItem, IconButton, Box } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';


interface ExportProps {
  isVisible: boolean;
  onClose: () => void;
}

const Export: React.FC<ExportProps> = ({ isVisible, onClose }) => {
  const [rosbagName, setRosbagName] = useState('');
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [topics, setTopics] = useState('');

  const handleExport = () => {
    // Add your export logic here
    console.log('Exporting rosbag:', rosbagName, startTime, endTime, topics);
    onClose();
  };

  if (!isVisible) return null;

  return (
    <Box
      sx={{
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        color: '#fff',
        padding: '20px',
        borderRadius: '8px',
        zIndex: 1000,
        width: '800px', // Optional: Set a fixed width for consistency
      }}
    >
      <IconButton
        sx={{
          position: 'absolute',
          top: '8px',
          right: '8px',
          color: '#fff',
        }}
        onClick={onClose}
      >
        <CloseIcon />
      </IconButton>
    </Box>
  );
};

export default Export;
