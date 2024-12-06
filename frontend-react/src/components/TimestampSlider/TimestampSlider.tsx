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
    selectedTimestamp ? timestamps.indexOf(selectedTimestamp) : 0
  );

  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [timestampUnit, setTimestampUnit] = useState<'ROS' | 'TOD'>('ROS');
  const [showSearchInput, setShowSearchInput] = useState(false); // State to control the visibility of the input
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<string[]>([]); // Initialize as an empty array
  const [searchMarks, setSearchMarks] = useState<{ value: number; label: string }[]>([]);  
  const [imageGallery, setImageGallery] = useState<string[]>([]); // Store multiple images

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const searchIconRef = useRef<HTMLButtonElement | null>(null); // Reference to the search icon

  // update selected timestamp if slider or possible timestamps change
  useEffect(() => {
    if (selectedTimestamp !== null) {
      const index = timestamps.indexOf(selectedTimestamp);
      setSliderValue(index);
    }
  }, [selectedTimestamp, timestamps]);

  // Method for fetching the API search results
  const fetchSearchResults = async (query: string) => {
    if (!query.trim()) return;

    try {
      const response = await fetch(`/api/search?query=${query}`, { method: 'GET' });
      const data = await response.json();

      // TODO: maybe move to backend?
      var results = [];
      for (var result of data.results){
        results.push(result.path.substring(56,82));
      }

      setSearchResults(results);
      setSearchMarks(data.marks)
    } catch (error) {
      console.error('Error fetching search results:', error);
      setSearchResults([]); // In case of an error, default to an empty array
    }
  };

  // Fetch all images for the search results
  const fetchAllImages = async () => {
    try {
      const imagePromises = searchResults.map(async (result) => {
        const response = await fetch(
          `/api/ros?timestamp=${result.substring(7)}&topic=/camera_image/${result.substring(0, 6)}&mode=search`
        );
        const data = await response.json();
        return data.image || null;
      });

      const fetchedImages = await Promise.all(imagePromises);
      setImageGallery(fetchedImages.filter((img) => img !== null)); // Filter out any null values
    } catch (error) {
      console.error('Error fetching images:', error);
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

  useEffect(() => {
    if (searchResults.length > 0) {
      fetchAllImages();
    }
  }, [searchResults]);
  
  const handleSearchKeyDown = async (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      await fetchSearchResults(searchQuery);
    }
  };


  return (
    <div className="timestamp-slider-container">
      
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
        aria-label="play" 
        color="primary" 
        onClick={togglePlayback}>
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
        sx={{ 
          marginRight: '12px',
          '& .MuiSlider-mark': {
            backgroundColor: '#FFA500',
            width: '3px',
            height: '7px'
          },
         }}
        marks={searchMarks}
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

      {/* Search icon button to toggle the search input */}
      <IconButton 
        ref={searchIconRef} 
        aria-label="search" 
        onClick={toggleSearchInput} 
        sx={{ fontSize: '1.2rem' }}
      >
        <SearchIcon />
      </IconButton>


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


      {/* Popper for Search Results */}
      <Popper
        open={searchResults.length > 0 && showSearchInput} // Show the Popper only when there are results
        anchorEl={searchIconRef.current} // Position relative to the search icon
        placement="top" // Display results above the icon
        sx={{
          zIndex: 1400,
          width: '210px',
          left: '50%', // Center the Popper horizontally
          transform: 'translateX(-50%)', // Center the Popper horizontally
          padding: '8px',
          background: '#202020', // Dark background color
          borderRadius: '8px',
          boxShadow: '0 2px 10px rgba(0, 0, 0, 0.3)', // Optional: shadow for better visibility
        }}
        modifiers={[
          {
            name: 'offset',
            options: {
              offset: [0, 70], // Move the popper 10px higher above the search icon (increase value for higher)
            },
          },
        ]}
      >
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'flex-start', // Align items to the left
        width: '100%', // Allow images to fill the width of the container
        gap: '8px', // Space between image and text
        overflowY: 'auto', // In case there are many images, allow scrolling
        padding: '8px',
      }}>
        {imageGallery && imageGallery.length > 0 ? (
          imageGallery.map((imgBase64, index) => (
            <div key={index} style={{ textAlign: 'left', width: '100%' }}>
              <img
                src={`data:image/webp;base64,${imgBase64}`}
                alt={`Search result ${index + 1}`}
                style={{
                  width: '100%', // Make the image fill the container's width
                  height: 'auto', // Maintain the aspect ratio
                  objectFit: 'contain', // Keep the image proportion intact
                }}
              />
              <Typography 
                variant="body2" 
                sx={{ 
                  color: 'white', 
                  wordBreak: 'break-word', 
                  marginTop: '4px', // Tiny margin between image and text
                  fontSize: '0.7rem', // Make the font smaller (you can adjust this value)
                }}
              >
                {searchResults[index]} {/* Match the description with the image */}
              </Typography>
            </div>
          ))
        ) : (
          <p style={{ color: "white", fontSize: "0.8rem" }}>
            Loading...
          </p>
        )}
      </div>
      </Popper>

    </div>
  );
};

export default TimestampSlider;