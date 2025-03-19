import React, { useEffect, useState, useRef } from 'react';
import './TimestampSlider.css'; // Import the CSS file
import { FormControl, IconButton, InputLabel, MenuItem, Select, Slider, SelectChangeEvent, Typography, TextField, Popper, Skeleton, Paper, Box, List, ListItem, ListItemText, ListItemButton } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from "@mui/icons-material/Pause";
import SearchIcon from '@mui/icons-material/Search';
import FilterAltIcon from '@mui/icons-material/FilterAlt';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css'; // Ensure Leaflet CSS is loaded

interface TimestampSliderProps {
  timestamps: number[];
  selectedTimestamp: number | null;
  onSliderChange: (value: number) => void;
  selectedRosbag: string | null;
}

const TimestampSlider: React.FC<TimestampSliderProps> = ({
  timestamps,
  selectedTimestamp,
  onSliderChange,
  selectedRosbag
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
  const [showFilter, setShowFilter] = useState(false); // State to control the visibility of the filter
  const [showModelSelection, setShowModelSelection] = useState(false); // State to control the visibility of the model selection
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<{ rank : number; embedding_path: string; distance: number; timestamp: string; topic: string }[]>([]); // Initialize as an empty array of SearchResult objects  
  const [searchMarks, setSearchMarks] = useState<{ value: number; label: string }[]>([]);  
  const [imageGallery, setImageGallery] = useState<string[]>([]); // Store multiple images
  const [models, setModels] = useState<string[]>([]); // State to store the list of models
  const [selectedModel, setSelectedModel] = useState<string>(''); // State to store the selected model

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const searchIconRef = useRef<HTMLButtonElement | null>(null); // Reference to the search icon
  const filterIconRef = useRef<HTMLButtonElement | null>(null); // Reference to the filter icon

  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
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
    }, [showFilter]);


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

      setSearchMarks(data.marks || []);
      setSearchResults(data.results || []);
    } catch (error) {
      console.error('Error fetching search results:', error);
      setSearchResults([]); // In case of an error, default to an empty array
    }
  };

  // Fetch all images for the search results
  const fetchAllImages = async () => {
    try {
      // Generate the image URLs directly without making fetch requests
      const imageUrls = searchResults.map((result) => {
        const imageUrl =
          result.topic && result.timestamp && selectedRosbag
            ? `http://localhost:5000/images/${selectedRosbag}/${result.topic.replaceAll("/", "__")}-${result.timestamp}.webp`
            : undefined;
        return imageUrl;
      });
  
      // Filter out any undefined values (in case any field was missing)
      setImageGallery(imageUrls.filter((url) => url !== undefined) as string[]);
    } catch (error) {
      console.error("Error generating image URLs:", error);
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

  const toggleFilter = () => {
    setShowFilter(!showFilter);
  };
  
  useEffect(() => {
    if (searchResults.length > 0) {
      fetchAllImages();
    }
  }, [searchResults]);
  
  const handleSearchKeyDown = async (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      setSearchResults([]); // Clear the previous results
      setImageGallery([]); // Clear the previous images
      await fetchSearchResults(searchQuery);
    }
  };

  const toggleModelSelection = () => {
    setShowModelSelection(!showModelSelection);
  };

  // Fetch models from the API when the dropdown is opened
  useEffect(() => {
    if (showModelSelection) {
      fetchModels();
    }
  }, [showModelSelection]);

  // Function to fetch models from the API
  const fetchModels = async () => {
    try {
      const response = await fetch('/api/get-models');
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }
      const data = await response.json();
      setModels(data.models); // Update the state with the fetched models
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  // Function to handle model selection
  const handleModelSelect = async (event: SelectChangeEvent<string>) => {
    const selectedValue = event.target.value;
    setSelectedModel(selectedValue); // Update the selected model state

    try {
      const response = await fetch('/api/set-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ models: selectedValue }), // Send the selected model to the API
      });

      if (!response.ok) {
        throw new Error('Failed to set model');
      }

      const data = await response.json();
      console.log(data.message); // Log the success message
    } catch (error) {
      console.error('Error setting model:', error);
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
        aria-label={isPlaying ? "pause" : "play"} 
        color="primary" 
        onClick={togglePlayback}>
        {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
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
      
      {/* Filter icon button */}  
      <IconButton
        ref={filterIconRef}
        aria-label='filter'
        onClick={toggleFilter}
        sx={{ fontSize: '1.2rem' }}
      >
        <FilterAltIcon />
      </IconButton>
      {/* Search icon button to toggle the search input */}
      <IconButton 
        ref={searchIconRef} 
        aria-label="search" 
        onClick={toggleSearchInput} 
        sx={{ fontSize: '1.2rem' }}
      >
        <SearchIcon />
      </IconButton>
      
      {/* Popper for Filter */}
      <Popper
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
      </Popper>

      {/* Popper for Search Input */}
      <Popper
        open={showSearchInput}
        anchorEl={searchIconRef.current} // Position relative to the search icon
        placement="top-end" // Places the popper above the icon
        sx={{
          zIndex: 1300, // Ensure it appears above other content
          width: "252px", // Adjust width to make the input box thinner
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
        <Box sx={{ display: 'flex', flexDirection: 'row', gap: '8px', paddingBlock: '8px'}}>
          <IconButton
            aria-label="model"
            onClick={toggleModelSelection}
            sx={{ fontSize: '1.2rem', background: "#202020"}} 
          >
            <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e8eaed"><path d="M720-140 560-300l160-160 56 56-63 64h167v80H713l63 64-56 56Zm-560-20q-33 0-56.5-23.5T80-240v-120q0-33 23.5-56.5T160-440h240q33 0 56.5 23.5T480-360v120q0 33-23.5 56.5T400-160H160Zm0-80h240v-120H160v120Zm80-260-56-56 63-64H80v-80h167l-63-64 56-56 160 160-160 160Zm320-20q-33 0-56.5-23.5T480-600v-120q0-33 23.5-56.5T560-800h240q33 0 56.5 23.5T880-720v120q0 33-23.5 56.5T800-520H560Zm0-80h240v-120H560v120ZM400-240v-120 120Zm160-360v-120 120Z"/></svg>
          </IconButton>
          <TextField
            variant="outlined"
            label="Search"
            value={searchQuery}
            onChange={(e: { target: { value: React.SetStateAction<string>; }; }) => setSearchQuery(e.target.value)}  // Update the query as the user types
            onKeyDown={handleSearchKeyDown} // Trigger API call only when Enter is pressed
            size="small"
            sx={{
              fontSize: '0.7rem', // Smaller font size
              background: "#202020", // Dark background color for input box
              width: '100%', // Ensure the width fits the parent container (Popper)
              borderRadius: '4px', // Optional: Add rounded corners for a smoother look
            }}
          />
        </Box>
        {/* Nested Popper for Model Selection Dropdown */}
        <Popper
          open={showModelSelection} // Controlled by the toggleModelSelection state
          anchorEl={searchIconRef.current} // Anchor to the same search icon
          placement="top-end" // Place the dropdown below the IconButton
          sx={{
            zIndex: 1300, // Ensure it appears above other content
            width: "200px", // Adjust width to make the input box thinner
            background: "#202020", // Dark background color
            borderRadius: '8px', // Rounded corners
            marginTop: '8px', // Add some spacing from the IconButton
          }}
          modifiers={[
            {
              name: 'offset',
              options: {
                offset: [-213, 70], // Move the popper 10px higher above the search icon (increase value for higher)
              },
            },
          ]}
        >
          <Box sx={{ padding: '8px' }}>
            <Typography variant="body2" sx={{ color: "#e8eaed", marginBottom: '8px' }}>
              Select a Model
            </Typography>
            <Select
              labelId="model-select-label"
              id="model-select"
              value={selectedModel}
              onChange={handleModelSelect}
              sx={{ width: '100%', color: "#e8eaed", background: "#404040" }} // Styling for the Select component
            >
              {models.map((model, index) => (
                <MenuItem key={index} value={model}>
                  {model}
                </MenuItem>
              ))}
            </Select>
          </Box>
        </Popper>

      </Popper>
  
      {/* Popper for Search Results */}
      <Popper
        open={showSearchInput} // Show the Popper only when there are results
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
          imageGallery.map((imgUrl, index) => (
            <div key={index} style={{ textAlign: 'left', width: '100%' }}>
              <img
                src={imgUrl}
                alt={"No result available"}
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
                {
                  searchResults[index] && (
                    <>
                      <div>{searchResults[index].topic}</div>
                      <div>{searchResults[index].timestamp}</div>
                      <div>Distance: {searchResults[index].distance.toFixed(5)}</div>
                    </>
                  )
                }              
              </Typography>
            </div>
          ))
        ) : (
          //<p style={{ color: "white", fontSize: "0.8rem" }}>
          //  Loading...
          //</p>
          //<div className="circular-progress-container">
          //  <CircularProgress />
          //</div>
          <>
            <Skeleton variant="rounded" width={178} height={60} />
            <Skeleton variant="rounded" width={178} height={60} />
            <Skeleton variant="rounded" width={178} height={60} />
            <Skeleton variant="rounded" width={178} height={60} />
            <Skeleton variant="rounded" width={178} height={60} />
          </>
        )}
      </div>
      </Popper>

    </div>
  );
};

export default TimestampSlider;