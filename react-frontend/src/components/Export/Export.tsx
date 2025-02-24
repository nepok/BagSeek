import React, { useState, useEffect, useRef } from 'react';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField, FormControl, InputLabel, Select, MenuItem, Box, Slider, Checkbox, ListItemText, SelectChangeEvent } from '@mui/material';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css'; // Ensure Leaflet CSS is loaded

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
  const [exportRange, setExportRange] = useState<number[]>([0, Math.max(0, timestamps.length - 1)]);
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isVisible) return;
  
    const initializeMap = () => {
      if (!mapContainerRef.current) return;
  
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
  
      mapRef.current = L.map(mapContainerRef.current).setView(
        [51.25757432197879, 12.51589660271899], 16
      );
  
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      }).addTo(mapRef.current);
  
      let points: L.LatLng[] = [];
      let circles: L.CircleMarker[] = [];
      let polygon: L.Polygon | null = null;
  
      const onMapClick = (e: L.LeafletMouseEvent) => {
        const clickedLatLng = e.latlng;
  
        // If user clicks near the first point, close the polygon
        if (points.length > 2 && clickedLatLng.distanceTo(points[0]) < 10) {
          if (polygon) {
            polygon.remove();
          }
          polygon = L.polygon(points, { color: 'blue', fillOpacity: 0.5 }).addTo(mapRef.current!);
          
          // Reset points and circles
          circles.forEach(circle => circle.remove());
          points = [];
          circles = [];
          return;
        }
  
        // Add small circle instead of default marker
        const circle = L.circleMarker(clickedLatLng, {
          radius: 5,
          color: 'blue',
          fillColor: 'blue',
          fillOpacity: 0.8,
        }).addTo(mapRef.current!);
  
        circles.push(circle);
        points.push(clickedLatLng);
      };
  
      const onRightClick = (e: L.LeafletMouseEvent) => {
        // Remove polygon if it exists
        if (polygon) {
          polygon.remove();
          polygon = null;
        }
        // Remove all circles
        circles.forEach(circle => circle.remove());
        circles = [];
        points = [];
      };
  
      mapRef.current.on('click', onMapClick);
      mapRef.current.on('contextmenu', onRightClick); // Right-click event
  
      return () => {
        if (mapRef.current) {
          mapRef.current.off('click', onMapClick);
          mapRef.current.off('contextmenu', onRightClick);
        }
      };
    };
  
    setTimeout(initializeMap, 100);
  
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [isVisible]);

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
    if (timestamps.length === 0) return;

    const exportData = {
      new_rosbag_name: newRosbagName,
      topics: selectedTopics,
      start_timestamp: timestamps[Math.min(exportRange[0], timestamps.length - 1)],
      end_timestamp: timestamps[Math.min(exportRange[1], timestamps.length - 1)],
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
    const date = new Date(timestamp / 1000000);
    return new Intl.DateTimeFormat('de-DE', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3,
      hour12: false,
    }).format(date);
  };

  const valueLabelFormat = (value: number) => {
    if (value < 0 || value >= timestamps.length) return 'Invalid';
    return `${formatDate(timestamps[value])} (${timestamps[value]})`;
  };

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
            max={Math.max(0, timestamps.length - 1)}
            valueLabelFormat={valueLabelFormat}
          />
        </Box>
        <Box sx={{ mt: 2, height: 400 }}>
          <div ref={mapContainerRef} style={{ height: '400px', width: '100%' }}></div>
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