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

  // State for selected rosbag name
  const [selectedRosbag, setSelectedRosbag] = useState('');
  // State for new rosbag export name input
  const [newRosbagName, setNewRosbagName] = useState('');
  // State for selected topics when filtering by topic
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  // Mode to select by 'topic' or by 'type'
  const [selectionMode, setSelectionMode] = useState<'topic' | 'type'>('topic');
  // State for selected types when filtering by type
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  // Range of timestamps selected for export (indices)
  const [exportRange, setExportRange] = useState<number[]>([0, Math.max(0, timestamps.length - 1)]);
  // Export progress and status information
  const [exportStatus, setExportStatus] = useState<{progress: number, status: string} | null>(null);
  // Ref for Leaflet map instance
  const mapRef = useRef<L.Map | null>(null);
  // Ref for map container DOM element
  const mapContainerRef = useRef<HTMLDivElement | null>(null);

  // Initialize Leaflet map when dialog becomes visible
  /*
  useEffect(() => {
    if (!isVisible) return;
  
    const initializeMap = () => {
      if (!mapContainerRef.current) return;
  
      // Clean up existing map instance if any
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
  
      // Create new map centered on given coordinates
      mapRef.current = L.map(mapContainerRef.current).setView(
        [51.25757432197879, 12.51589660271899], 16
      );
  
      // Add OpenStreetMap tile layer
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      }).addTo(mapRef.current);
  
      // Variables to store polygon points and markers
      let points: L.LatLng[] = [];
      let circles: L.CircleMarker[] = [];
      let polygon: L.Polygon | null = null;
  
      // Handler for map clicks to add points or close polygon
      const onMapClick = (e: L.LeafletMouseEvent) => {
        const clickedLatLng = e.latlng;
  
        // Close polygon if clicked near the first point and enough points exist
        if (points.length > 2 && clickedLatLng.distanceTo(points[0]) < 10) {
          if (polygon) {
            polygon.remove();
          }
          polygon = L.polygon(points, { color: 'blue', fillOpacity: 0.5 }).addTo(mapRef.current!);
          
          // Clear points and markers after closing polygon
          circles.forEach(circle => circle.remove());
          points = [];
          circles = [];
          return;
        }
  
        // Add small circle marker instead of default marker for each click
        const circle = L.circleMarker(clickedLatLng, {
          radius: 5,
          color: 'blue',
          fillColor: 'blue',
          fillOpacity: 0.8,
        }).addTo(mapRef.current!);
  
        circles.push(circle);
        points.push(clickedLatLng);
      };
  
      // Handler for right-click to clear polygon and points
      const onRightClick = (e: L.LeafletMouseEvent) => {
        if (polygon) {
          polygon.remove();
          polygon = null;
        }
        circles.forEach(circle => circle.remove());
        circles = [];
        points = [];
      };
  
      // Attach event listeners for clicks and right-clicks
      mapRef.current.on('click', onMapClick);
      mapRef.current.on('contextmenu', onRightClick); // Right-click event
  
      // Cleanup event listeners on unmount
      return () => {
        if (mapRef.current) {
          mapRef.current.off('click', onMapClick);
          mapRef.current.off('contextmenu', onRightClick);
        }
      };
    };
  
    // Delay map initialization slightly to ensure container is ready
    setTimeout(initializeMap, 100);
  
    // Cleanup map on dialog close
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [isVisible]);
  */

  // Fetch currently selected rosbag on component mount
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

  // Reset export status message after completion delay
  useEffect(() => {
  if (exportStatus?.status === 'done') {
    const timer = setTimeout(() => {
      setExportStatus(null);
    }, 3000);
    return () => clearTimeout(timer);
  }
}, [exportStatus]);

  // Handle export button click: send export request and poll for status
  const handleExport = async () => {
    if (timestamps.length === 0) return;

    setExportStatus({status: 'starting', progress: -1});

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
        // Start polling export status every second after successful request
        const pollInterval = setInterval(async () => {
          try {
            const statusRes = await fetch("/api/export-status");
            const statusData = await statusRes.json();
            setExportStatus(statusData);

            if (statusData.status === "done" || statusData.status === "idle") {
              clearInterval(pollInterval);
            }
          } catch (err) {
            console.error("Polling error:", err);
            clearInterval(pollInterval);
          }
        }, 1000);
      } else {
        console.error('Export failed:', result.error);
      }
    } catch (error) {
      console.error('Error exporting rosbag:', error);
    }

    onClose();
  };

  // Handle slider value changes for export range
  const handleSliderChange = (event: Event, newValue: number | number[]) => {
    setExportRange(newValue as number[]);
  };

  // Handle topic selection changes and synchronize types if in type mode
  const handleTopicChange = (event: SelectChangeEvent<string[]>) => {
    const topicsSelected = event.target.value as string[];
    setSelectedTopics(topicsSelected);
    // If in type selection mode, update selected types based on selected topics
    if (selectionMode === 'type') {
      // Find unique types for selected topics
      const types = Array.from(new Set(topicsSelected.map(t => topicTypes[t])));
      setSelectedTypes(types);
    }
  };

  // Handle type selection changes and filter topics accordingly
  const handleTypeChange = (event: SelectChangeEvent<string[]>) => {
    const types = event.target.value as string[];
    setSelectedTypes(types);
    const filteredTopics = topics.filter(t => types.includes(topicTypes[t]));
    setSelectedTopics(filteredTopics);
  };

  // Format timestamp to German locale date string with milliseconds
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

  // Format slider value label with date and raw timestamp
  const valueLabelFormat = (value: number) => {
    if (value < 0 || value >= timestamps.length) return 'Invalid';
    return `${formatDate(timestamps[value])} (${timestamps[value]})`;
  };

  // Render only export status popup when dialog is not visible
  if (!isVisible) {
    return (
      <>
        {exportStatus && (
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
            <Typography variant="body2">
              {exportStatus.progress === -1
                ? "Loading Rosbag..."
                : exportStatus.progress === 1
                  ? "Finished exporting!"
                  : `Export progress: ${(exportStatus.progress * 100).toFixed(0)}%`}
            </Typography>
            <Box sx={{ width: 200, mt: 1 }}>
              {exportStatus.progress === -1 ? (
                <LinearProgress />
              ) : (
                <LinearProgress variant="determinate" value={exportStatus.progress * 100} />
              )}
            </Box>
          </Box>
        )}
      </>
    );
  }

  // Main export dialog UI
  return (
    <Dialog open={true} onClose={onClose} aria-labelledby="form-dialog-title" fullWidth maxWidth="md">
      <DialogTitle id="form-dialog-title">Export Content of {selectedRosbag}</DialogTitle>
      <DialogContent style={{ overflow: 'hidden' }}>
        {/* Input for new rosbag name */}
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
        {/* Toggle button group for selecting filter mode */}
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
        {/* Topic selection UI */}
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
        {/* Type selection UI */}
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
        {/* Slider to select export timestamp range */}
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
          {/* Highlight box showing selected export range */}
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
        {/* Leaflet map container (currently commented out) */}
        {/* <Box sx={{ mt: 2, height: 400 }}>
          <div ref={mapContainerRef} style={{ height: '400px', width: '100%' }}></div>
        </Box>*/} 
      </DialogContent>
      <DialogActions>
        {/* Cancel button closes dialog */}
        <Button onClick={onClose} color="primary">
          Cancel
        </Button>
        {/* Export button triggers export logic */}
        <Button onClick={handleExport} color="primary">
          Export
        </Button>
      </DialogActions>
    </Dialog>

    
  );
};

export default Export;