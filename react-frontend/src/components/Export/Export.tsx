import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField, FormControl, InputLabel, Select, MenuItem, Box, Slider, Checkbox, ListItemText, SelectChangeEvent, Typography, LinearProgress, ButtonGroup, alpha } from '@mui/material';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css'; // Ensure Leaflet CSS is loaded
import { CustomTrack } from '../CustomTrack/CustomTrack';
import { times } from 'lodash';

interface ExportProps {
  timestamps: number[];
  timestampDensity: number[];
  availableTopics: Record<string, string>; // Unified: { topicName: messageType }
  isVisible: boolean;
  onClose: () => void;
  searchMarks: { value: number; label: string }[];
  selectedRosbag: string | null;
}

const Export: React.FC<ExportProps> = ({ timestamps, timestampDensity, availableTopics, isVisible, onClose, searchMarks, selectedRosbag: selectedRosbagProp }) => {
  // Derive topics array and topicTypes map from unified availableTopics
  const topics = Object.keys(availableTopics);
  const topicTypes = availableTopics;

  // State for selected rosbag name (fallback if prop not provided)
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
  const [exportStatus, setExportStatus] = useState<{progress: number, status: string, message?: string} | null>(null);
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

  // Fetch currently selected rosbag when dialog becomes visible (fallback if prop not provided)
  useEffect(() => {
    // Use prop if provided, otherwise fetch
    if (selectedRosbagProp) {
      setSelectedRosbag(selectedRosbagProp);
      return;
    }

    // Only fetch if dialog is visible and prop is not provided
    if (!isVisible) return;

    const fetchSelectedRosbag = async () => {
      try {
        const response = await fetch('/api/get-selected-rosbag');
        const result = await response.json();
        if (response.ok) {
          setSelectedRosbag(result.selected_rosbag || result.selectedRosbag || '');
        } else {
          console.error('Failed to fetch selected rosbag:', result.error);
        }
      } catch (error) {
        console.error('Error fetching selected rosbag:', error);
      }
    };

    fetchSelectedRosbag();
  }, [isVisible, selectedRosbagProp]);

  // Poll export status periodically when running, stop after 3 failed fetches
  useEffect(() => {
    let retryCount = 0;
    let interval: NodeJS.Timeout;

    if (exportStatus?.status === 'running' || exportStatus?.status === 'starting') {
      interval = setInterval(async () => {
        try {
          const response = await fetch('/api/export-status');
          const data = await response.json();
          setExportStatus(data);
          retryCount = 0; // reset on success
        } catch (err) {
          console.error('Failed to fetch export status:', err);
          retryCount++;
          if (retryCount >= 3) {
            console.warn('Stopping export status polling after 3 failed attempts');
            clearInterval(interval);
          }
        }
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [exportStatus?.status]);

  // Reset export status message after completion delay
  useEffect(() => {
    if (exportStatus?.status === 'completed' || exportStatus?.status === 'done') {
      const timer = setTimeout(() => {
        setExportStatus(null);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [exportStatus]);

  // Handle export button click: send export request and poll for status
  const handleExport = async () => {
    if (timestamps.length === 0) return;

    // Don't set status here - backend will set it. We'll fetch it after sending the request.

    // Get timestamp indices
    const startIndex = Math.min(exportRange[0], timestamps.length - 1);
    const endIndex = Math.min(exportRange[1], timestamps.length - 1);
    const startTimestamp = timestamps[startIndex];
    const endTimestamp = timestamps[endIndex];

    // Fetch MCAP mapping to get MCAP IDs for the timestamps
    let startMcapId: string | null = null;
    let endMcapId: string | null = null;
    
    try {
      // Fetch the MCAP mapping for the selected rosbag
      const mappingResponse = await fetch(`/api/get-topic-mcap-mapping?relative_rosbag_path=${encodeURIComponent(selectedRosbag)}`);
      if (mappingResponse.ok) {
        const mappingData = await mappingResponse.json();
        const ranges = mappingData.ranges || [];
        
        // Helper function to find MCAP ID for a given index
        const findMcapIdForIndex = (index: number): string | null => {
          for (let i = 0; i < ranges.length; i++) {
            const range = ranges[i];
            const nextStart = i < ranges.length - 1 ? ranges[i + 1].startIndex : mappingData.total;
            
            if (index >= range.startIndex && index < nextStart) {
              return range.mcap_identifier;
            }
          }
          return null;
        };
        
        startMcapId = findMcapIdForIndex(startIndex);
        endMcapId = findMcapIdForIndex(endIndex);
      }
    } catch (error) {
      console.error('Error fetching MCAP mapping:', error);
      alert('Failed to fetch MCAP mapping. Export cannot proceed without MCAP IDs.');
      // Don't set status here - let backend handle it, or clear it since export didn't start
      setExportStatus(null);
      return;
    }

    const exportData = {
      new_rosbag_name: newRosbagName.trim(),
      topics: selectedTopics, // Can be empty array (will export all topics if empty)
      start_timestamp: startTimestamp, // nanoseconds
      end_timestamp: endTimestamp, // nanoseconds
      start_mcap_id: startMcapId, // string
      end_mcap_id: endMcapId, // string
    };

    // Set status to 'starting' BEFORE making API call to trigger polling immediately
    // This matches the pattern used in GlobalSearch
    setExportStatus({ progress: -1, status: 'starting', message: 'Starting export...' });

    // Close dialog immediately so progress can be shown
    onClose();

    try {
      // Send export request (backend will update status to 'running' when it starts processing)
      const response = await fetch('/api/export-rosbag', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(exportData),
      });

      // Check if response is JSON before parsing
      const contentType = response.headers.get('content-type');
      let result;
      if (contentType && contentType.includes('application/json')) {
        try {
          const text = await response.text();
          if (!text) {
            console.error('Empty response from server');
            setExportStatus({status: 'error', progress: -1, message: `Server returned empty response: ${response.status} ${response.statusText}`});
            return;
          }
          result = JSON.parse(text);
        } catch (jsonError) {
          console.error('Failed to parse JSON response:', jsonError);
          setExportStatus({status: 'error', progress: -1, message: `Server error: ${response.status} ${response.statusText}. Failed to parse JSON.`});
          return;
        }
      } else {
        const text = await response.text();
        console.error('Non-JSON response:', text.substring(0, 200));
        setExportStatus({status: 'error', progress: -1, message: `Server returned non-JSON response: ${response.status} ${response.statusText}`});
        return;
      }

      if (response.ok) {
        // Status polling is already running (triggered by 'starting' status above)
        // Backend will update status to 'running' when it starts processing
        // No need to fetch here - polling will pick it up automatically
      } else {
        console.error('Export failed:', result?.error || result);
        // Backend should set error status, but if it didn't, fetch it
        try {
          const statusRes = await fetch('/api/export-status');
          if (statusRes.ok) {
            const statusData = await statusRes.json();
            setExportStatus(statusData);
          } else {
            // Fallback if backend doesn't have status
            setExportStatus({status: 'error', progress: -1, message: result?.error || 'Unknown error'});
          }
        } catch (err) {
          setExportStatus({status: 'error', progress: -1, message: result?.error || 'Unknown error'});
        }
      }
    } catch (error) {
      console.error('Error exporting rosbag:', error);
      // Try to get status from backend first
      try {
        const statusRes = await fetch('/api/export-status');
        if (statusRes.ok) {
          const statusData = await statusRes.json();
          setExportStatus(statusData);
        } else {
          setExportStatus({status: 'error', progress: -1, message: error instanceof Error ? error.message : 'Unknown error'});
        }
      } catch (err) {
        setExportStatus({status: 'error', progress: -1, message: error instanceof Error ? error.message : 'Unknown error'});
      }
    }
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
            backgroundColor: exportStatus.status === 'error' ? '#d32f2f' : '#202020',
            color: 'white',
            padding: '8px 16px',
            borderRadius: '8px',
            zIndex: 9999,
            boxShadow: '0px 0px 10px rgba(0,0,0,0.5)',
            maxWidth: '400px'
          }}>
            <Typography variant="body2" sx={{ 
              color: exportStatus.status === 'error' ? '#fff' : 'inherit',
              fontWeight: exportStatus.status === 'error' ? 'bold' : 'normal'
            }}>
              {exportStatus.status === 'error'
                ? exportStatus.message || "Export failed!"
                : exportStatus.progress === -1
                  ? exportStatus.message || "Loading Rosbag..."
                  : exportStatus.progress === 1 || exportStatus.status === "completed"
                    ? exportStatus.message || "Finished exporting!"
                    : exportStatus.message || `Export progress: ${(exportStatus.progress * 100).toFixed(0)}%`}
            </Typography>
            <Box sx={{ width: 200, mt: 1 }}>
              {exportStatus.status === 'error' ? (
                <LinearProgress variant="determinate" value={100} color="error" />
              ) : exportStatus.progress === -1 ? (
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