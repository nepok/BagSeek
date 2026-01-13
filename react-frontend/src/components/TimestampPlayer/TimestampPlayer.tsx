// React component for interacting with and controlling timestamp-based playback, search, and model filtering
import React, { useEffect, useState, useRef } from 'react';
import './TimestampPlayer.css'; // Import the CSS file
import { FormControl, IconButton, InputLabel, MenuItem, Select, Slider, SelectChangeEvent, Typography } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from "@mui/icons-material/Pause";
//import FilterAltIcon from '@mui/icons-material/FilterAlt';
//import L from 'leaflet';
import { CustomTrack } from '../CustomTrack/CustomTrack';
import 'leaflet/dist/leaflet.css'; // Ensure Leaflet CSS is loaded
import { useError } from '../ErrorContext/ErrorContext'; // adjust path as needed

interface TimestampPlayerProps {
  availableTimestamps: number[];
  timestampDensity: number[];
  selectedTimestamp: number | null;
  onSliderChange: (value: number) => void;
  selectedRosbag: string | null;
  searchMarks: { value: number; label: string }[];
  setSearchMarks: React.Dispatch<React.SetStateAction<{ value: number; label: string }[]>>;
}

const TimestampPlayer: React.FC<TimestampPlayerProps> = (props) => {
  const {
    availableTimestamps,
    timestampDensity,
    selectedTimestamp,
    onSliderChange,
    selectedRosbag,
    searchMarks,
    setSearchMarks,
  } = props;
  
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
      hour12: false, // 24-hour format

    }).format(date);

    return berlinTime;
  };

  const [sliderValue, setSliderValue] = useState(
    selectedTimestamp ? availableTimestamps.indexOf(selectedTimestamp) : 0
  ); // current index in timestamp list
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1); // playback multiplier (e.g., 0.5x, 1x, 2x)
  const [isPlaying, setIsPlaying] = useState(false); // whether playback is running
  const [timestampUnit, setTimestampUnit] = useState<'ROS' | 'TOD'>('ROS'); // display mode: ROS or formatted time
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


  // Keep slider position in sync with selected timestamp
  useEffect(() => {
    if (selectedTimestamp !== null) {
      const index = availableTimestamps.indexOf(selectedTimestamp);
      setSliderValue(index);
    }
  }, [selectedTimestamp, availableTimestamps]);


  // Handle slider movement: update index and propagate to parent
  const handleSliderChange = (event: Event, value: number | number[]) => {
    const newValue = Array.isArray(value) ? value[0] : value;
    setSliderValue(newValue);
    onSliderChange(availableTimestamps[newValue]);
  };

  // Update playback speed and restart playback timer if running
  const handlePlaybackSpeedChange = (event: SelectChangeEvent<number>) => {
    const newSpeed = event.target.value as number;
    setPlaybackSpeed(newSpeed);

    if (isPlaying) {
      clearInterval(intervalRef.current!);
      // Updated logic for intervalTime and step
      const step = newSpeed <= 0.5 ? 1 : newSpeed === 1 ? 2 : newSpeed === 1.5 ? 3 : 4;
      const intervalTime = newSpeed <= 0.5 ? 150 / newSpeed : 300;
      intervalRef.current = setInterval(() => {
        setSliderValue(prevSliderValue => {
          const nextIndex = (prevSliderValue + step) % availableTimestamps.length;
          onSliderChange(availableTimestamps[nextIndex]);
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
      // Updated logic for intervalTime and step
      const step = playbackSpeed <= 0.5 ? 1 : playbackSpeed === 1 ? 2 : playbackSpeed === 1.5 ? 3 : 4;
      const intervalTime = playbackSpeed <= 0.5 ? 150 / playbackSpeed : 300;
      console.log(`Starting playback at speed ${playbackSpeed} with step ${step} and interval time ${intervalTime}ms`);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      intervalRef.current = setInterval(() => {
        setSliderValue(prevSliderValue => {
          const nextIndex = (prevSliderValue + step) % availableTimestamps.length;
          onSliderChange(availableTimestamps[nextIndex]);
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
      <IconButton 
        aria-label={isPlaying ? "pause" : "play"} 
        color="primary" 
        onClick={togglePlayback}>
        {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
      </IconButton>

      {/* Timestamp slider */}
      <Slider
        size="small"
        min={0}
        max={availableTimestamps.length - 1}
        step={1}
        value={sliderValue}
        onChange={handleSliderChange}
        aria-label="Timestamp"
        components={{
          Track: (props) => (
            <CustomTrack
              {...props}
              timestampCount={availableTimestamps.length}
              searchMarks={searchMarks}
              timestampDensity={timestampDensity}
              bins={1000} // optional
              windowSize={50} // optional
            />
          ),
        }}
        sx={{
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
          ? selectedTimestamp
          : selectedTimestamp && formatDate(selectedTimestamp)}
      </Typography>

      {/* Select for Timestamp Unit */}
      <FormControl sx={{ m: 1, minWidth: 80 }} size="small">
        <InputLabel id="timestamp-unit-select-label" sx={{ fontSize: '0.8rem' }}></InputLabel>
        <Select
          labelId="timestamp-unit-select-label"
          id="timestamp-unit-select"
          value={timestampUnit}
          onChange={handleTimestampUnitChange}
          sx={{ fontSize: '0.8rem' }}
        >
          <MenuItem value="ROS" sx={{ fontSize: '0.8rem' }}>ROS</MenuItem>
          <MenuItem value="TOD" sx={{ fontSize: '0.8rem' }}>TOD</MenuItem>
        </Select>
      </FormControl>
      
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