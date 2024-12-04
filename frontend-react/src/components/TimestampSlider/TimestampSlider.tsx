import React, { useEffect, useState, useRef } from 'react';
import './TimestampSlider.css'; // Import the CSS file
import { FormControl, IconButton, InputLabel, MenuItem, Select, Slider, SelectChangeEvent, Typography, TextField, Popper } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import SearchIcon from '@mui/icons-material/Search';

interface TimestampSliderProps {
  timestamps: number[];
  selectedTimestamp: number | null;
  onSliderChange: (value: number) => void;
}

const TimestampSlider: React.FC<TimestampSliderProps> = ({
  timestamps,
  selectedTimestamp,
  onSliderChange,
}) => {
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
      hour12: false, // 24-hour format
    }).format(date);

    return berlinTime;
  };

  const [sliderValue, setSliderValue] = useState(
    selectedTimestamp ? timestamps.indexOf(selectedTimestamp) : 0
  );

  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [timestampUnit, setTimestampUnit] = useState<'ROS' | 'TOD'>('ROS');
  const [isPlaying, setIsPlaying] = useState(false);
  const [showSearchInput, setShowSearchInput] = useState(false); // State to control the visibility of the input
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]); // Initialize as an empty array

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const searchIconRef = useRef<HTMLButtonElement | null>(null); // Reference to the search icon

  useEffect(() => {
    if (selectedTimestamp !== null) {
      const index = timestamps.indexOf(selectedTimestamp);
      setSliderValue(index);
    }
  }, [selectedTimestamp, timestamps]);

  // Function to fetch the API search results
  const fetchSearchResults = async (query: string) => {
    if (!query.trim()) return;

    try {
      const response = await fetch(`/api/search?query=${query}`, { method: 'GET' });
      const data = await response.json();
      setSearchResults(Array.isArray(data) ? data : []); // Ensure data is always an array
    } catch (error) {
      console.error('Error fetching search results:', error);
      setSearchResults([]); // In case of an error, default to an empty array
    }
  };

  const handleSliderChange = (event: Event, value: number | number[]) => {
    const newValue = Array.isArray(value) ? value[0] : value;
    setSliderValue(newValue);
    onSliderChange(timestamps[newValue]);
  };

  const handlePlaybackSpeedChange = (event: SelectChangeEvent<number>) => {
    const newSpeed = event.target.value as number;
    setPlaybackSpeed(newSpeed);

    if (isPlaying) {
      clearInterval(intervalRef.current!);
      const intervalTime = 83.33 / newSpeed;
      intervalRef.current = setInterval(() => {
        setSliderValue(prevSliderValue => {
          const nextIndex = (prevSliderValue + 1) % timestamps.length;
          onSliderChange(timestamps[nextIndex]);
          return nextIndex;
        });
      }, intervalTime);
    }
  };

  const handleTimestampUnitChange = (event: SelectChangeEvent<string>) => {
    setTimestampUnit(event.target.value as 'ROS' | 'TOD');
  };

  const togglePlayback = () => {
    if (isPlaying) {
      clearInterval(intervalRef.current!);
      setIsPlaying(false);
    } else {
      const intervalTime = 83.33 / playbackSpeed;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      intervalRef.current = setInterval(() => {
        setSliderValue(prevSliderValue => {
          const nextIndex = (prevSliderValue + 1) % timestamps.length;
          onSliderChange(timestamps[nextIndex]);
          return nextIndex;
        });
      }, intervalTime);

      setIsPlaying(true);
    }
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Toggle the input box visibility
  const toggleSearchInput = () => {
    setShowSearchInput(!showSearchInput);
  };

  // Function to handle search input change and trigger the API call only when Enter is pressed
  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      fetchSearchResults(searchQuery);
    }
  };

  return (
    <div className="timestamp-slider-container">
      {/* Popper for Search Results */}
      <Popper
        open={searchResults.length > 0} // Show the Popper only when there are results
        anchorEl={searchIconRef.current} // Position relative to the search icon
        placement="top" // Display results above the icon
        sx={{
          zIndex: 1400,
          width: '300px',
          left: '50%', // Center the Popper horizontally
          transform: 'translateX(-50%)', // Center the Popper horizontally
          padding: '8px',
          background: '#202020', // Dark background color
          borderRadius: '8px',
          boxShadow: '0 2px 10px rgba(0, 0, 0, 0.3)', // Optional: shadow for better visibility
        }}
      >
        <div>
          {searchResults.map((result: any) => (
            <Typography key={result.timestamp} variant="body2" sx={{ color: 'white' }}>
              {result.timestamp}
            </Typography>
          ))}
        </div>
      </Popper>
      {/* Popper for Search Input */}
      <Popper
        open={showSearchInput}
        anchorEl={searchIconRef.current} // Position relative to the search icon
        placement="top-end" // Places the popper above the icon
        sx={{
          zIndex: 1300, // Ensure it appears above other content
          width: "200px", // Adjust width to make the input box thinner
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
        <TextField
          variant="outlined"
          label="Search"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}  // Update the query as the user types
          onKeyDown={handleSearchKeyDown} // Trigger API call only when Enter is pressed
          size="small"
          sx={{
            fontSize: '0.7rem', // Smaller font size
            background: "#202020", // Dark background color for input box
            width: '100%', // Ensure the width fits the parent container (Popper)
            borderRadius: '4px', // Optional: Add rounded corners for a smoother look
          }}
        />
      </Popper>

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
      <IconButton aria-label="play" color="primary" onClick={togglePlayback}>
        <PlayArrowIcon />
      </IconButton>

      {/* Timestamp slider */}
      <Slider
        size="small"
        min={0}
        max={timestamps.length - 1}
        step={1}
        value={sliderValue}
        onChange={handleSliderChange}
        aria-label="Timestamp"
        sx={{ marginRight: '12px' }}
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
        <InputLabel id="timestamp-unit-select-label" sx={{ fontSize: '0.8rem' }}>Unit</InputLabel>
        <Select
          labelId="timestamp-unit-select-label"
          id="timestamp-unit-select"
          value={timestampUnit}
          label="Unit"
          onChange={handleTimestampUnitChange}
          sx={{ fontSize: '0.8rem' }}
        >
          <MenuItem value="ROS" sx={{ fontSize: '0.8rem' }}>ROS</MenuItem>
          <MenuItem value="TOD" sx={{ fontSize: '0.8rem' }}>TOD</MenuItem>
        </Select>
      </FormControl>

      {/* Search icon button to toggle the search input */}
      <IconButton 
        ref={searchIconRef} 
        aria-label="search" 
        onClick={toggleSearchInput} 
        sx={{ fontSize: '1.2rem' }}
      >
        <SearchIcon />
      </IconButton>
      
      {/* Display search results */}
      {searchResults.length > 0 && (
        <div className="search-results">
          {searchResults.map((result: any) => (
            <div key={result.timestamp}>
              <Typography variant="body2">{result.timestamp}</Typography>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default TimestampSlider;