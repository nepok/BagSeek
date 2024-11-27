import React, { useEffect, useState, useRef } from 'react';
import './TimestampSlider.css'; // Import the CSS file
import { FormControl, IconButton, InputLabel, MenuItem, Select, Slider, SelectChangeEvent, Typography } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

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
  // Convert the Unix timestamp to a readable date (Berlin time zone)
  const formatDate = (timestamp: number): string => {
    if (!timestamp || isNaN(timestamp)) {
      return 'Invalid Timestamp'; // Return a fallback message if the timestamp is invalid
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

  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1); // Default playback speed is 1
  const [timestampUnit, setTimestampUnit] = useState<'ROS' | 'TOD'>('ROS');
  const [isPlaying, setIsPlaying] = useState(false);

  // Refs to hold the interval ID
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Update the slider value when selectedTimestamp changes
  useEffect(() => {
    if (selectedTimestamp !== null) {
      const index = timestamps.indexOf(selectedTimestamp);
      setSliderValue(index);
    }
  }, [selectedTimestamp, timestamps]);

  // Handle slider change
  const handleSliderChange = (event: Event, value: number | number[]) => {
    const newValue = Array.isArray(value) ? value[0] : value;
    setSliderValue(newValue);
    onSliderChange(timestamps[newValue]);
  };

  // Handle playback speed change
  const handlePlaybackSpeedChange = (event: SelectChangeEvent<number>) => {
    const newSpeed = event.target.value as number;
    setPlaybackSpeed(newSpeed);

    // If the playback is already running, reset the interval with the new speed
    if (isPlaying) {
      // Clear previous interval
      clearInterval(intervalRef.current!);

      // Set new interval time based on the new speed
      // TODO: automatic playback speed
      const intervalTime = 83.33 / newSpeed;
      intervalRef.current = setInterval(() => {
        setSliderValue(prevSliderValue => {
          const nextIndex = (prevSliderValue + 1) % timestamps.length; // Increment and loop back
          onSliderChange(timestamps[nextIndex]); // Update the parent with new timestamp
          return nextIndex; // Update the slider value
        });
      }, intervalTime);
    }
  };

  // Handle timestamp unit change (ROS or TOD)
  const handleTimestampUnitChange = (event: SelectChangeEvent<string>) => {
    setTimestampUnit(event.target.value as 'ROS' | 'TOD');
  };

  // Function to start and stop playback
  const togglePlayback = () => {
    if (isPlaying) {
      clearInterval(intervalRef.current!); // Clear interval if playback is stopped
      setIsPlaying(false);
    } else {
      // Set the interval time dynamically based on the selected playback speed
      // TODO: automatic playback speed
      const intervalTime = 83.33 / playbackSpeed; // Adjust interval based on playback speed

      // Clear the previous interval before setting a new one
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }

      intervalRef.current = setInterval(() => {
        setSliderValue(prevSliderValue => {
          const nextIndex = (prevSliderValue + 1) % timestamps.length; // Increment and loop back
          onSliderChange(timestamps[nextIndex]); // Update the parent with new timestamp
          return nextIndex; // Update the slider value
        });
      }, intervalTime); // Set interval based on playback speed

      setIsPlaying(true);
    }
  };

  // Cleanup interval when component is unmounted or playback stops
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

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
          sx={{ fontSize: '0.8rem' }} // Apply smaller font size to Select
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
      <FormControl sx={{ m: 1, minWidth: 74 }} size="small">
        <InputLabel id="timestamp-unit-select-label" sx={{ fontSize: '0.8rem' }}></InputLabel>
        <Select
          labelId="timestamp-unit-select-label"
          id="timestamp-unit-select"
          value={timestampUnit}
          onChange={handleTimestampUnitChange}
          sx={{ fontSize: '0.8rem' }} // Apply smaller font size to Select
        >
          <MenuItem value="ROS" sx={{ fontSize: '0.8rem' }}>ROS</MenuItem>
          <MenuItem value="TOD" sx={{ fontSize: '0.8rem' }}>TOD</MenuItem>
        </Select>
      </FormControl>
    </div>
  );
};

export default TimestampSlider;