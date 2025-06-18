import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField, FormControl, InputLabel, Select, MenuItem, Box, Slider, Checkbox, ListItemText, SelectChangeEvent, Typography, LinearProgress, ButtonGroup, alpha } from '@mui/material';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css'; // Ensure Leaflet CSS is loaded
import { CustomTrack } from '../CustomTrack.tsx/CustomTrack';
import { times } from 'lodash';

interface ExportProps {
  timestamps: number[];
  timestampDensity: number[];
  topics: string[];
  isVisible: boolean;
  onClose: () => void;
  searchMarks: { value: number; label: string }[];
  topicTypes: Record<string, string>;
}

const Export: React.FC<ExportProps> = ({ timestamps, timestampDensity, topics, isVisible, onClose, searchMarks, topicTypes }) => {

  const [selectedRosbag, setSelectedRosbag] = useState('');
  const [newRosbagName, setNewRosbagName] = useState('');
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  const [selectionMode, setSelectionMode] = useState<'topic' | 'type'>('topic');
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  const [exportRange, setExportRange] = useState<number[]>([0, Math.max(0, timestamps.length - 1)]);
  const [exportStatus, setExportStatus] = useState<{progress: number, message: string, status: string} | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!exportStatus || exportStatus.status !== "running") return;
  
    const interval = setInterval(async () => {
      const res = await fetch("/api/export-status");
      const data = await res.json();
      setExportStatus(data);
  
      if (data.status === "done") clearInterval(interval);
    }, 100);
  
    return () => clearInterval(interval);
  }, [exportStatus]);

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

    setExportStatus({ status: "running", progress: 0, message: "Export started..." });

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
    const topicsSelected = event.target.value as string[];
    setSelectedTopics(topicsSelected);
    // Falls im Type-Modus: synchronisiere selectedTypes
    if (selectionMode === 'type') {
      // Finde alle Typen, die in den neuen Topics enthalten sind
      const types = Array.from(new Set(topicsSelected.map(t => topicTypes[t])));
      setSelectedTypes(types);
    }
  };

  const handleTypeChange = (event: SelectChangeEvent<string[]>) => {
    const types = event.target.value as string[];
    setSelectedTypes(types);
    const filteredTopics = topics.filter(t => types.includes(topicTypes[t]));
    setSelectedTopics(filteredTopics);
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

  if (!isVisible) {
    return (
      <>
        {exportStatus && exportStatus.status === "running" && (
          <Box sx={{
            position: 'fixed',
            bottom: 80,
            left: 20,
            backgroundColor: '#202020',
            color: 'white',
            padding: '8px 16px',
            borderRadius: '8px',
            zIndex: 9999,
            boxShadow: '0px 0px 10px rgba(0,0,0,0.5)'
          }}>
            <Typography variant="body2">{exportStatus.message}</Typography>
            <Box sx={{ width: 200, mt: 1 }}>
              <LinearProgress variant="determinate" value={exportStatus.progress * 100} />
            </Box>
          </Box>
        )}
      </>
    );
  }

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
        <ButtonGroup
          sx={{ my: 2, display: 'flex', width: '100%' }}
          variant="outlined"
        >
          <Button
            sx={{ flex: 1 }}
            variant={selectionMode === 'topic' ? 'contained' : 'outlined'}
            onClick={() => setSelectionMode('topic')}
          >
            Topics
          </Button>
          <Button
            sx={{ flex: 1 }}
            variant={selectionMode === 'type' ? 'contained' : 'outlined'}
            onClick={() => setSelectionMode('type')}
          >
            Types
          </Button>
        </ButtonGroup>
        {selectionMode === 'topic' && (
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
        )}
        {selectionMode === 'type' && (
          <FormControl fullWidth margin="dense">
            <InputLabel id="types-label">Types</InputLabel>
            <Select
              labelId="types-label"
              id="types"
              multiple
              value={selectedTypes}
              onChange={handleTypeChange}
              renderValue={(selected) => (selected as string[]).join(', ')}
            >
              {Array.from(new Set(Object.values(topicTypes))).map((type) => (
                <MenuItem key={type} value={type}>
                  <Checkbox checked={selectedTypes.includes(type)} />
                  <ListItemText primary={type} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}
        <Box sx={{ mt: 2, position: 'relative' }}>
          <Slider
            value={exportRange}
            onChange={handleSliderChange}
            valueLabelDisplay="auto"
            min={0}
            max={Math.max(0, timestamps.length - 1)}
            valueLabelFormat={valueLabelFormat}
            components={{
              Track: (props) => (
                <CustomTrack
                  {...props}
                  timestampCount={timestamps.length}
                  searchMarks={searchMarks}
                  timestampDensity={timestampDensity}
                  bins={1000}
                  windowSize={50}
                />
              ),
            }}
            sx={{
              '& .MuiSlider-thumb': {
                backgroundColor: 'primary', // or another solid color
                zIndex: 2, // ensure they sit above the highlight
              }
            }}
          />
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              transform: 'translateY(-10px)', // half of the defined height
              left: `${(exportRange[0] / (timestamps.length - 1)) * 100}%`,
              width: `${((exportRange[1] - exportRange[0]) / (timestamps.length - 1)) * 100}%`,
              height: '20px', // fixed height in pixels
              backgroundColor: (theme) => alpha(theme.palette.primary.main, 0.2), // semi-transparent primary color
              pointerEvents: 'none',
              zIndex: 1,
            }}
          />
        </Box>
        {/* <Box sx={{ mt: 2, height: 400 }}>
          <div ref={mapContainerRef} style={{ height: '400px', width: '100%' }}></div>
        </Box>*/} 
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