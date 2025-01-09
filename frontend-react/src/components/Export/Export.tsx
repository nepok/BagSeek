import React, { useState } from 'react';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField, FormControl, InputLabel, Select, MenuItem, Box, Slider, Checkbox, ListItemText, SelectChangeEvent } from '@mui/material';

interface ExportProps {
  timestamps: number[];
  topics: string[];
  isVisible: boolean;
  onClose: () => void;
}

const Export: React.FC<ExportProps> = ({ timestamps, topics, isVisible, onClose }) => {
  const [rosbagName, setRosbagName] = useState('');
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  const [exportRange, setExportRange] = useState<number[]>([0, timestamps.length - 1]);

  const formatDate = (timestamp: number): string => {
    if (!timestamp || isNaN(timestamp)) {
      return 'Invalid Timestamp';
    }
    // TODO: automatic conversion for time unit seconds, milli, micro and nano seconds
    const date = new Date(timestamp / 1000000); // Divide by 1,000,000 to convert to seconds
    const berlinTime = new Intl.DateTimeFormat('de-DE', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3,
      hour12: false, // 24-hour format
    }).format(date);

    return berlinTime;
  };

  const valueLabelingFormat = (value: number) => {
    const timestamp = timestamps[value];
    const formattedDate = formatDate(timestamp);
    return { formattedDate, timestamp };
  };

  const handleExport = () => {
    // Add your export logic here
    console.log('Exporting rosbag:', rosbagName, selectedTopics, exportRange.map(index => timestamps[index]));
    onClose();
  };

  const handleSliderChange = (event: Event, newValue: number | number[]) => {
    setExportRange(newValue as number[]);
  };

  const handleTopicChange = (event: SelectChangeEvent<string[]>) => {
    setSelectedTopics(event.target.value as string[]);
  };

  const MenuProps = {
    PaperProps: {
      style: {
        maxHeight: 48 * 4.5 + 8,
        width: 250,
      },
    },
  };

  if (!isVisible) return null;

  return (
    <Dialog open={true} onClose={onClose} aria-labelledby="form-dialog-title">
      <DialogTitle id="form-dialog-title">Export Content of {rosbagName} </DialogTitle>
      <DialogContent>
        <TextField
          autoFocus
          margin="dense"
          id="rosbagName"
          label="Name"
          type="text"
          fullWidth
          value={rosbagName}
          onChange={(e) => setRosbagName(e.target.value)}
        />
        <FormControl fullWidth margin="dense">
          <InputLabel id="topics-label">Topics</InputLabel>
          <Select
            labelId="topics-label"
            id="topics"
            multiple
            value={selectedTopics}
            onChange={handleTopicChange}
            renderValue={(selected) => (selected as string[]).join(', ')}
            MenuProps={MenuProps}
          >
            {topics.map((topic) => (
              <MenuItem key={topic} value={topic}>
                <Checkbox checked={selectedTopics.includes(topic)} />
                <ListItemText primary={topic} />
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Box sx={{ mt: 2 }}>
          <Slider
            value={exportRange}
            defaultValue={[0, timestamps.length - 1]}
            onChange={handleSliderChange}
            valueLabelDisplay="auto"
            min={0}
            max={timestamps.length - 1}
            valueLabelFormat={(value) => {
              const { formattedDate, timestamp } = valueLabelingFormat(value);
              return `${formattedDate} (${timestamp})`;
            }}          
          />
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">
          Cancel
        </Button>
        <Button onClick={handleExport} color="primary">
          Export
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default Export;
