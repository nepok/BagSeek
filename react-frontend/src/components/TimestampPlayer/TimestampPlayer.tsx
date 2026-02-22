// React component for interacting with and controlling timestamp-based playback, search, and model filtering
import React, { useEffect, useState, useRef } from 'react';
import './TimestampPlayer.css'; // Import the CSS file
import { Box, FormControl, IconButton, InputLabel, MenuItem, Select, Slider, SelectChangeEvent, Tooltip, Typography } from '@mui/material';
import HelpPopover from '../HelpPopover/HelpPopover';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from "@mui/icons-material/Pause";
//import FilterAltIcon from '@mui/icons-material/FilterAlt';
//import L from 'leaflet';
import { CustomTrack } from '../CustomTrack/CustomTrack';
import 'leaflet/dist/leaflet.css'; // Ensure Leaflet CSS is loaded
import { useError } from '../ErrorContext/ErrorContext'; // adjust path as needed

interface TimestampPlayerProps {
  timestampCount: number;
  firstTimestampNs: string | null;
  lastTimestampNs: string | null;
  selectedTimestampIndex: number | null;
  selectedTimestamp: string | null;
  onSliderChange: (index: number) => void;
  selectedRosbag: string | null;
  searchMarks: { value: number; rank?: number }[];
  setSearchMarks: React.Dispatch<React.SetStateAction<{ value: number; rank?: number }[]>>;
  mcapBoundaries?: number[];
  mcapHighlightMask?: boolean[];
}

const TimestampPlayer: React.FC<TimestampPlayerProps> = (props) => {
  const {
    timestampCount,
    firstTimestampNs,
    lastTimestampNs,
    selectedTimestampIndex,
    selectedTimestamp,
    onSliderChange,
    selectedRosbag,
    searchMarks,
    setSearchMarks,
    mcapBoundaries = [],
    mcapHighlightMask,
  } = props;
  
  const formatDate = (timestampNs: string): string => {
    if (!timestampNs) return 'Invalid Timestamp';
    try {
      const ms = Number(BigInt(timestampNs) / BigInt(1000000));
      const date = new Date(ms);
      return new Intl.DateTimeFormat('de-DE', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      }).format(date);
    } catch {
      return 'Invalid Timestamp';
    }
  };

  const formatDuration = (totalSeconds: number): string => {
    const sign = totalSeconds < 0 ? '-' : '';
    const abs = Math.floor(Math.abs(totalSeconds));
    const hours = Math.floor(abs / 3600);
    const minutes = Math.floor((abs % 3600) / 60);
    const seconds = abs % 60;
    const mm = minutes.toString().padStart(2, '0');
    const ss = seconds.toString().padStart(2, '0');
    return hours > 0 ? `${sign}${hours}:${mm}:${ss}` : `${sign}${minutes}:${ss}`;
  };

  const getDurationDisplay = (): string => {
    if (!firstTimestampNs || !lastTimestampNs || !selectedTimestamp) return '0:00/0:00';
    try {
      const first = BigInt(firstTimestampNs);
      const last = BigInt(lastTimestampNs);
      const current = BigInt(selectedTimestamp);
      const totalSeconds = Number(last - first) / 1e9;
      const elapsedSeconds = Number(current - first) / 1e9;
      const totalFormatted = formatDuration(totalSeconds);
      if (showRemaining) {
        return `${formatDuration(elapsedSeconds - totalSeconds)}/${totalFormatted}`;
      }
      return `${formatDuration(elapsedSeconds)}/${totalFormatted}`;
    } catch {
      return '0:00/0:00';
    }
  };

  const [sliderValue, setSliderValue] = useState(selectedTimestampIndex ?? 0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1); // playback multiplier (e.g., 0.5x, 1x, 2x)
  const [isPlaying, setIsPlaying] = useState(false); // whether playback is running
  const [timestampUnit, setTimestampUnit] = useState<'ROS' | 'TOD'>('ROS'); // display mode: ROS or formatted time
  const [showRemaining, setShowRemaining] = useState(false);
  const [tsSelectOpen, setTsSelectOpen] = useState(false); // toggle elapsed vs remaining time
  //const [showFilter, setShowFilter] = useState(false); // toggle polygon filter view

  const intervalRef = useRef<NodeJS.Timeout | null>(null); // interval reference for playback
  //const filterIconRef = useRef<HTMLButtonElement | null>(null); // anchor for filter input

  //const mapRef = useRef<L.Map | null>(null); // Leaflet map instance
  //const mapContainerRef = useRef<HTMLDivElement | null>(null); // container for Leaflet map

  const { setError } = useError();

  // Initialize Leaflet map with click-to-polygon and right-click reset functionality
  /*useEffect(() => {
      if (!showFilter) return;
    
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
    }, [showFilter]);*/


  // Keep slider position in sync with selected index
  useEffect(() => {
    if (selectedTimestampIndex !== null) {
      setSliderValue(selectedTimestampIndex);
    }
  }, [selectedTimestampIndex]);


  // Handle slider movement: update index and propagate to parent
  const handleSliderChange = (event: Event, value: number | number[]) => {
    const newIndex = Array.isArray(value) ? value[0] : value;
    setSliderValue(newIndex);
    onSliderChange(newIndex);
  };

  // Update playback speed and restart playback timer if running
  const handlePlaybackSpeedChange = (event: SelectChangeEvent<number>) => {
    const newSpeed = event.target.value as number;
    setPlaybackSpeed(newSpeed);

    if (isPlaying) {
      clearInterval(intervalRef.current!);
      const step = newSpeed <= 0.5 ? 1 : newSpeed === 1 ? 2 : newSpeed === 1.5 ? 3 : 4;
      const intervalTime = newSpeed <= 0.5 ? 150 / newSpeed : 300;
      intervalRef.current = setInterval(() => {
        setSliderValue(prevSliderValue => {
          const nextIndex = (prevSliderValue + step) % timestampCount;
          onSliderChange(nextIndex);
          return nextIndex;
        });
      }, intervalTime);
    }
  };

  // Change displayed timestamp unit (ROS or formatted time)
  const handleTimestampUnitChange = (event: SelectChangeEvent<string>) => {
    setTimestampUnit(event.target.value as 'ROS' | 'TOD');
  };

  // Start or pause playback timer
  const togglePlayback = () => {
    if (isPlaying) {
      clearInterval(intervalRef.current!);
      setIsPlaying(false);
    } else {
      const step = playbackSpeed <= 0.5 ? 1 : playbackSpeed === 1 ? 2 : playbackSpeed === 1.5 ? 3 : 4;
      const intervalTime = playbackSpeed <= 0.5 ? 150 / playbackSpeed : 300;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      intervalRef.current = setInterval(() => {
        setSliderValue(prevSliderValue => {
          const nextIndex = (prevSliderValue + step) % timestampCount;
          onSliderChange(nextIndex);
          return nextIndex;
        });
      }, intervalTime);

      setIsPlaying(true);
    }
  };

  // Clean up playback timer on component unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);


  // Toggle polygon filter map open/close
  /*const toggleFilter = () => {
    setShowFilter(!showFilter);
  };*/
  

  // Render player controls, timestamp slider, search/model selection and result preview
  return (
    <div className="timestamp-player-container">
      
      {/* Select for playback speed */}
      <FormControl sx={{ m: 1, minWidth: 86 }} size="small">
        <InputLabel id="playback-speed-select-label" sx={{ fontSize: '0.8rem' }}>Speed</InputLabel>
        <Select
          labelId="playback-speed-select-label"
          id="playback-speed-select"
          value={playbackSpeed}
          label="Speed"
          onChange={handlePlaybackSpeedChange}
          sx={{ fontSize: '0.8rem' }}
        >
          <MenuItem value={0.125} sx={{ fontSize: '0.8rem' }}>0.125x</MenuItem>
          <MenuItem value={0.25} sx={{ fontSize: '0.8rem' }}>0.25x</MenuItem>
          <MenuItem value={0.5} sx={{ fontSize: '0.8rem' }}>0.5x</MenuItem>
          <MenuItem value={1} sx={{ fontSize: '0.8rem' }}>1x</MenuItem>
          <MenuItem value={1.5} sx={{ fontSize: '0.8rem' }}>1.5x</MenuItem>
          <MenuItem value={2} sx={{ fontSize: '0.8rem' }}>2x</MenuItem>
        </Select>
      </FormControl>

      {/* Play icon button */}
      <Tooltip title={isPlaying ? "Pause" : "Play"} arrow>
        <IconButton
          aria-label={isPlaying ? "pause" : "play"}
          color="primary"
          onClick={togglePlayback}>
          {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
        </IconButton>
      </Tooltip>

      {/* Duration display */}
      <Tooltip title="Click to toggle elapsed / remaining time" arrow>
        <Typography
          variant="body2"
          onClick={() => setShowRemaining(prev => !prev)}
          sx={{
            fontSize: '0.8rem',
            whiteSpace: 'nowrap',
            cursor: 'pointer',
            userSelect: 'none',
            fontFamily: 'monospace',
            flexShrink: 0,
          }}
        >
          {getDurationDisplay()}
        </Typography>
      </Tooltip>

      {/* Timestamp slider */}
      <Slider
        size="small"
        min={0}
        max={Math.max(0, timestampCount - 1)}
        step={1}
        value={sliderValue}
        onChange={handleSliderChange}
        aria-label="Timestamp"
        components={{
          Track: (props) => (
            <CustomTrack
              {...props}
              timestampCount={timestampCount}
              searchMarks={searchMarks}
              mcapBoundaries={mcapBoundaries}
              mcapHighlightMask={mcapHighlightMask}
              firstTimestampNs={firstTimestampNs}
              lastTimestampNs={lastTimestampNs}
              sliderValue={sliderValue}
              bins={1000} // optional
              windowSize={50} // optional
            />
          ),
        }}
        sx={{
          flex: '1 1 auto',
          minWidth: 0,
          marginRight: '12px',
          marginLeft: '10px',
        }}
        //marks={searchMarks}
      />

      {/* Display the selected timestamp */}
      <Typography
        variant="body2"
        sx={{ fontSize: '0.8rem', whiteSpace: 'nowrap', minWidth: 140 }}
      >
        {timestampUnit === 'ROS'
          ? (selectedTimestamp ?? '')
          : (selectedTimestamp ? formatDate(selectedTimestamp) : '')}
      </Typography>

      {/* Select for Timestamp Unit */}
      <Tooltip title="Timestamp format: ROS nanoseconds or Time of Day" arrow disableHoverListener={tsSelectOpen}>
        <FormControl sx={{ m: 1, minWidth: 80 }} size="small">
        <InputLabel id="timestamp-unit-select-label" sx={{ fontSize: '0.8rem' }}></InputLabel>
        <Select
          labelId="timestamp-unit-select-label"
          id="timestamp-unit-select"
          value={timestampUnit}
          onChange={handleTimestampUnitChange}
          onOpen={() => setTsSelectOpen(true)}
          onClose={() => setTsSelectOpen(false)}
          sx={{ fontSize: '0.8rem' }}
        >
          <MenuItem value="ROS" sx={{ fontSize: '0.8rem' }}>ROS</MenuItem>
          <MenuItem value="TOD" sx={{ fontSize: '0.8rem' }}>TOD</MenuItem>
        </Select>
      </FormControl>
      </Tooltip>

      <HelpPopover
        title="Timeline"
        content={
          <Box component="ul" sx={{ m: 0, pl: 2 }}>
            <Box component="li" sx={{ mb: 0.5 }}>Drag the slider to scrub through the sensory data of the rosbag.</Box>
            <Box component="li" sx={{ mb: 0.5 }}><strong>Thin vertical lines</strong> mark MCAP file boundaries within the rosbag.</Box>
            <Box component="li" sx={{ mb: 0.5 }}>Click the <strong>Play/Pause button</strong> to start/pause the playback of the rosbag.</Box>
            <Box component="li" sx={{ mb: 0.5 }}>Click the <strong>duration display</strong> to toggle between elapsed and remaining time.</Box>
            <Box component="li"><strong>ROS</strong> shows the raw nanosecond timestamp; <strong>TOD</strong> shows wall-clock time of day.</Box>
          </Box>
        }
        iconSx={{ ml: 0.5 }}
      />
      
      {/* Filter icon button */}  
      {/*<IconButton
        ref={filterIconRef}
        aria-label='filter'
        onClick={toggleFilter}
        sx={{ fontSize: '1.2rem' }}
      >
        <FilterAltIcon />
      </IconButton>*/} 
      
      {/* Popper for Filter */}
      {/*<Popper
        open={showFilter}
        anchorEl={filterIconRef.current} // Position relative to the search icon
        placement="top-end" // Places the popper above the icon
        sx={{
          zIndex: 100, // Ensure it appears above other content
          width: "600px", // Adjust width to make the input box thinner
        }}
        modifiers={[
          {
            name: 'offset',
            options: {
              offset: [0, 12], // Move the popper 10px higher above the search icon (increase value for higher)
            },
          },
        ]}
      >
        <Paper sx={{ padding: '8px', background: '#202020', borderRadius: '8px' }}>
            <div ref={mapContainerRef} style={{ height: '400px', width: '600px' }}></div>
        </Paper>
      </Popper> */}


    </div>
  );
};

export default TimestampPlayer;