import React, { useState, useEffect, useRef } from 'react';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField, FormControl, InputLabel, Select, MenuItem, Box, Slider, Checkbox, ListItemText, SelectChangeEvent } from '@mui/material';
import * as ol from 'ol';
import 'ol/ol.css';
import { Map, View } from 'ol';
import OSM from 'ol/source/OSM';
import { Tile as TileLayer } from 'ol/layer';
import { Draw } from 'ol/interaction';
import { Polygon } from 'ol/geom';
import { Style, Fill, Stroke } from 'ol/style';
import VectorSource from 'ol/source/Vector';

interface ExportProps {
  timestamps: number[];
  topics: string[];
  isVisible: boolean;
  onClose: () => void;
}

const Export: React.FC<ExportProps> = ({ timestamps, topics, isVisible, onClose }) => {
  const [selectedRosbag, setSelectedRosbag] = useState('');
  const [newRosbagName, setNewRosbagName] = useState('');
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  const [exportRange, setExportRange] = useState<number[]>([0, 100]);
  const mapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const fetchSelectedRosbag = async () => {
      try {
        const response = await fetch('/api/get-selected-rosbag');
        const result = await response.json();
        if (response.ok) {
          setSelectedRosbag(result.selected_rosbag);
        } else {
          console.error('Failed to fetch selected rosbag:', result.error);
        }
      } catch (error) {
        console.error('Error fetching selected rosbag:', error);
      }
    };

    fetchSelectedRosbag();
  }, []);

  const handleExport = async () => {
    const exportData = {
      new_rosbag_name: newRosbagName,
      topics: selectedTopics,
      start_timestamp: timestamps[exportRange[0]],
      end_timestamp: timestamps[exportRange[1]],
    };

    try {
      const response = await fetch('/api/export-rosbag', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(exportData),
      });

      const result = await response.json();
      if (response.ok) {
        console.log('Export successful:', result);
      } else {
        console.error('Export failed:', result.error);
      }
    } catch (error) {
      console.error('Error exporting rosbag:', error);
    }

    onClose();
  };

  const handleSliderChange = (event: Event, newValue: number | number[]) => {
    setExportRange(newValue as number[]);
  };

  const handleTopicChange = (event: SelectChangeEvent<string[]>) => {
    setSelectedTopics(event.target.value as string[]);
  };

  const formatDate = (timestamp: number): string => {
    if (!timestamp || isNaN(timestamp)) {
      return 'Invalid Timestamp';
    }
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

  const valueLabelFormat = (value: number) => {
    const timestamp = timestamps[value];
    const formattedDate = formatDate(timestamp);
    return `${formattedDate} (${timestamp})`;
  };

  // OpenLayers Map setup
  useEffect(() => {
    if (mapRef.current) {
      const map = new Map({
        target: mapRef.current,
        layers: [
          new TileLayer({
            source: new OSM(),
          }),
        ],
        view: new View({
          center: [0, 0],
          zoom: 4,
        }),
      });

      const draw = new Draw({
        source: new VectorSource(),
        type: 'Polygon',
      });

      map.addInteraction(draw);

      const style = new Style({
        fill: new Fill({
          color: 'rgba(255, 255, 255, 0.2)',
        }),
        stroke: new Stroke({
          color: '#ffcc33',
          width: 2,
        }),
      });

      draw.on('drawend', (e) => {
        const feature = e.feature;
        feature.setStyle(style);
      });
    }
  }, []);

  if (!isVisible) return null;

  return (
    <Dialog open={true} onClose={onClose} aria-labelledby="form-dialog-title" fullWidth maxWidth="md">
      <DialogTitle id="form-dialog-title">Export Content of {selectedRosbag}</DialogTitle>
      <DialogContent style={{ overflow: 'hidden' }}>
        <TextField
          autoFocus
          margin="dense"
          id="rosbagName"
          label="Name"
          type="text"
          fullWidth
          value={newRosbagName}
          onChange={(e) => setNewRosbagName(e.target.value)}
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
            onChange={handleSliderChange}
            valueLabelDisplay="auto"
            min={0}
            max={timestamps.length - 1}
            valueLabelFormat={valueLabelFormat}
          />
        </Box>
        <Box sx={{ mt: 2, height: '400px' }} ref={mapRef} />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="primary">Cancel</Button>
        <Button onClick={handleExport} color="primary">Export</Button>
      </DialogActions>
    </Dialog>
  );
};

export default Export;