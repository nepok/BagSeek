// React component for interacting with and controlling timestamp-based playback, search, and model filtering
import React, { useEffect, useState, useRef, useMemo } from 'react';
import './TimestampPlayer.css'; // Import the CSS file
import { FormControl, IconButton, InputLabel, MenuItem, Select, Slider, SelectChangeEvent, Typography, TextField, Popper, Skeleton, Paper, Box, List, ListItem, ListItemText, ListItemButton, LinearProgress, CircularProgress, Icon, Checkbox } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from "@mui/icons-material/Pause";
import SearchIcon from '@mui/icons-material/Search';
import ListIcon from '@mui/icons-material/List';
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
  const [showSearchInput, setShowSearchInput] = useState(false); // toggle search input box
  //const [showFilter, setShowFilter] = useState(false); // toggle polygon filter view
  const [showModelSelection, setShowModelSelection] = useState(false); // toggle model dropdown
  const [showRosbagSelection, setShowRosbagSelection] = useState(false); // toggle folder selection
  const [searchQuery, setSearchQuery] = useState(''); // input text for search query
  const [searchResults, setSearchResults] = useState<{ rank : number; embedding_path: string; similarityScore: number; timestamp: string; topic: string; model?: string, rosbag?: string}[]>([]); // list of returned search results
  //const [searchMarks, setSearchMarks] = useState<{ value: number; label: string }[]>([]);  
  const [imageGallery, setImageGallery] = useState<string[]>([]); // image previews for search results
  const [models, setModels] = useState<string[]>([]); // list of available model names 
  const [availableRosbags, setAvailableRosbags] = useState<string[]>([]); // list of available rosbag files
  const [selectedModel, setSelectedModel] = useState<string>('ViT-B-16-quickgelu__openai'); // current selected model
  const [selectedRosbags, setSelectedRosbags] = useState<string[]>([]); // list of selected rosbag files
  const [isSearching, setIsSearching] = useState(false); // status of async search

  const intervalRef = useRef<NodeJS.Timeout | null>(null); // interval reference for playback
  const searchIconRef = useRef<HTMLButtonElement | null>(null); // anchor for search input
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

  // Fetch semantic search results from backend API
  const fetchSearchResults = async (query: string) => {
    if (!query.trim()) return;

    try {
      const response = await fetch(`/api/search?query=${query}`, { method: 'GET' });
      const data = await response.json();

      setSearchMarks(data.marks || []);
      setSearchResults(data.results || []);
    } catch (error) {
      setError('Error fetching search results.');
      console.error('Error fetching search results:', error);
      setSearchResults([]); // In case of an error, default to an empty array
    }
  };

  // Build list of preview image URLs for current search results
  const fetchAllImages = async (maxResults = 5) => {
    try {
      // Generate the image URLs directly without making fetch requests
      const imageUrls = searchResults.slice(0, maxResults).map((result) => {
        const imageUrl =
          result.topic && result.timestamp && result.rosbag
            ? `http://localhost:5000/images/${result.rosbag}/${result.topic.replace(/\//g, '_').replace(/^_/, '')}/${result.timestamp}.png`
            : undefined;
        return imageUrl;
      });
  
      // Filter out any undefined values (in case any field was missing)
      setImageGallery(imageUrls.filter((url) => url !== undefined) as string[]);
    } catch (error) {
      setError('Error generating image URLs.');
      console.error("Error generating image URLs:", error);
    }
  };

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

  // Toggle the input box visibility
  // Toggle search input box open/close
  const toggleSearchInput = () => {
    setShowSearchInput(!showSearchInput);
  };

  // Toggle polygon filter map open/close
  /*const toggleFilter = () => {
    setShowFilter(!showFilter);
  };*/
  
  // Update preview image gallery when search results change
  useEffect(() => {
    if (searchResults.length > 0) {
      fetchAllImages(5);
    }
  }, [searchResults]);

  // Reset UI state on rosbag change (keep heatmap marks if already provided)
  useEffect(() => {
    setSearchResults([]);
    setImageGallery([]);
    // Do not clear searchMarks here so heatmap can be shown by default when provided via URL
  }, [selectedRosbag]);
  
  // Run search when user presses Enter in search box
  const handleSearchKeyDown = async (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      setIsSearching(true); // Start searching
      setSearchResults([]); // Clear the previous results
      setImageGallery([]); // Clear the previous images
      await fetchSearchResults(searchQuery);
      setIsSearching(false); // End searching
    }
  };

  const toggleRosbagSelection = () => {
    setShowRosbagSelection(!showRosbagSelection);
  }

  useEffect(() => {
    if (showRosbagSelection) {
      fetchRosbags();
    }
  }, [showRosbagSelection]);

  useEffect(() => {
    if (selectedRosbags.length > 0) {
      // Reset search results and image gallery when rosbags are selected
      setSearchResults([]);
    }
  }, [selectedRosbags]);


  // Toggle model selection dropdown open/close
  const toggleModelSelection = () => {
    setShowModelSelection(!showModelSelection);
  };

  // Fetch model list when model selector is opened
  useEffect(() => {
    if (showModelSelection) {
      fetchModels();
    }
  }, [showModelSelection]);

  // Fetch available models from backend API
  const fetchModels = async () => {
    try {
      const response = await fetch('/api/get-models');
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }
      const data = await response.json();
      setModels(data.models); // Update the state with the fetched models
    } catch (error) {
      setError('Error fetching models.');
      console.error('Error fetching models:', error);
    }
  };

  const fetchRosbags = async () => {
    try {
      const response = await fetch('/api/get-file-paths');
      if (!response.ok) {
        throw new Error('Failed to fetch folders');
      }
      const data = await response.json();
      setAvailableRosbags(data.paths || []); // Update the state with the fetched folders
      console.log('Fetched folders:', data.paths); // Log the fetched folders
    } catch (error) {
      setError('Error fetching folders.');
      console.error('Error fetching folders:', error);
    }
  };

  // Handle selection of rosbags from dropdown
  const handleRosbagsSelection = async (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value as string[];

    let newSelection: string[] = [];

    if (value.includes("ALL")) {
      newSelection =
        selectedRosbags.length === availableRosbags.length ? [] : availableRosbags;
    } else if (value.includes("currently selected")) {
      const matched = availableRosbags.find(
        (bag) => selectedRosbag && bag.split("/").pop() === selectedRosbag
      );
      if (matched) {
        newSelection = [matched];
      }
    } else {
      newSelection = value;
    }

    setSelectedRosbags(newSelection); // Update state

    try {
      const response = await fetch("/api/set-searched-rosbags", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ searchedRosbags: newSelection }),
      });

      if (!response.ok) {
        throw new Error("Failed to set searched rosbags");
      }

      const data = await response.json();
      console.log(data.message);
    } catch (error) {
      setError("Error setting searched rosbags.");
      console.error("Error setting searched rosbags:", error);
    }
  };

  // Handle selection of a model from dropdown
  const handleModelSelect = async (event: SelectChangeEvent<string>) => {
    const selectedValue = event.target.value;
    setSelectedModel(selectedValue); // Update the selected model state

    try {
      const response = await fetch('/api/set-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model: selectedValue }), // Send the selected model to the API
      });

      if (!response.ok) {
        throw new Error('Failed to set model');
      }

      const data = await response.json();
      console.log(data.message); // Log the success message
    } catch (error) {
      setError('Error setting model.');
      console.error('Error setting model:', error);
    }
  };

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
      {/* Folder icon button */}
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
        {/* New Folder Button positioned above model + search input */}
        <Box sx={{ display: 'flex', gap: '8px'}}>
          <Box sx={{ width: '40px' }}>
            <IconButton
              aria-label="folder"
              onClick={toggleRosbagSelection}
              sx={{ fontSize: '1.2rem', background: "#202020" }}
            >
              <ListIcon />
            </IconButton>
          </Box>
        </Box>

                {/* Nested Popper for Rosbag Selection */}
        <Popper
          open={showRosbagSelection}
          anchorEl={searchIconRef.current}
          placement="top-end"
          sx={{
            zIndex: 1300,
            width: "250px",
            background: "#202020",
            borderRadius: '8px',
            marginTop: '8px',
          }}
          modifiers={[
            {
              name: 'offset',
              options: {
                offset: [-213, 130],
              },
            },
          ]}
        >
          <Box sx={{ padding: '8px' }}>
            <Typography variant="body2" sx={{ color: "#e8eaed", marginBottom: '8px' }}>
              Select Rosbags
            </Typography>
            <FormControl fullWidth margin="dense">
              <InputLabel id="rosbag-select-label" sx={{ color: '#e8eaed' }}>Rosbags</InputLabel>
              {
                // Find the availableRosbags entry that matches the selectedRosbag substring
              }
              {(() => {
                const matchedRosbag = availableRosbags.find(bag =>
                  selectedRosbag && bag.split('/').pop() === selectedRosbag
                ) || '';
                return (
                  <Select
                    labelId="rosbag-select-label"
                    id="rosbag-select"
                    multiple
                    value={selectedRosbags}
                    onChange={handleRosbagsSelection}
                    renderValue={(selected) => (selected as string[]).join(', ')}
                    sx={{ background: "#404040", color: "#e8eaed" }}
                  >
                    {/* ALL Option */}
                    <MenuItem value="ALL">
                      <Checkbox checked={selectedRosbags.length === availableRosbags.length && availableRosbags.length > 0} />
                      <ListItemText primary="SELECT ALL" />
                    </MenuItem>
                    <MenuItem value="currently selected">
                      <Checkbox checked={selectedRosbags.includes(matchedRosbag)} />
                      <ListItemText primary="CURRENTLY SELECTED" />
                    </MenuItem>
                    {/* Map through available rosbags */}
                    {(Array.isArray(availableRosbags) ? availableRosbags : []).map((bag) => (
                      <MenuItem key={bag} value={bag}>
                        <Checkbox checked={selectedRosbags.includes(bag)} />
                        <ListItemText primary={bag} />
                      </MenuItem>
                    ))}
                  </Select>
                );
              })()}
            </FormControl>
          </Box>
        </Popper>

        {/* Model Button + Search Input Row */}
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
        open={showSearchInput}
        anchorEl={searchIconRef.current}
        placement="top"
        sx={{
          zIndex: 1200,
          width: '210px',
          left: '50%',
          transform: 'translateX(-50%)',
          padding: '8px',
          background: '#202020',
          borderRadius: '8px',
          boxShadow: '0 2px 10px rgba(0, 0, 0, 0.3)',
        }}
        modifiers={[
          {
            name: 'offset',
            options: {
              offset: [0, 70],
            },
          },
        ]}
      >
        <Box sx={{ display: 'flex', flexDirection: 'row', gap: '8px' }}>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'flex-start',
            width: '100%',
            minHeight: '24px',
            gap: '8px',
            padding: '8px',
            maxHeight: 'calc(100vh - 100px)',
            overflowY: 'auto',
          }}>
            {imageGallery && imageGallery.length > 0 ? (
              imageGallery.map((imgUrl, index) => (
                <div key={index} style={{ textAlign: 'left', width: '100%' }}>
                  <img
                    src={imgUrl}
                    alt={"No result available"}
                    style={{
                      width: '100%',
                      height: 'auto',
                      objectFit: 'contain',
                    }}
                  />
                  <Typography
                    variant="body2"
                    sx={{
                      color: 'white',
                      wordBreak: 'break-word',
                      marginTop: '4px',
                      fontSize: '0.7rem',
                    }}
                  >
                    {searchResults[index] && (
                      <>
                        <div>{searchResults[index].topic}</div>
                        <div>{searchResults[index].timestamp}</div>
                        <div>{searchResults[index].rosbag}</div>
                      </>
                    )}
                  </Typography>
                </div>
              ))
            ) : (
              <>
                {isSearching && (
                  <LinearProgress
                    sx={{
                      width: '100%',
                    }}
                  />
                )}
              </>
            )}
          </div>
        </Box>
      </Popper>

    </div>
  );
};

export default TimestampPlayer;