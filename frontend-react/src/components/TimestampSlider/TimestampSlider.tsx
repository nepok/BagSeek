import React, { useEffect, useState } from 'react';
import './TimestampSlider.css'; // Import the CSS file

interface TimestampSliderProps {
  timestamps: number[];
  selectedTimestamp: number | null;
  onSliderChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
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
    console.log(timestamp)
    const date = new Date(timestamp / 1000000); // Divide by 1.000.000 to convert to seconds
    const berlinTime = new Intl.DateTimeFormat('de-DE', {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false, // 24-hour format
    }).format(date);

    return berlinTime;
  };

  // Ensure selectedTimestamp is valid or set to the first timestamp
  const [sliderValue, setSliderValue] = useState(
    selectedTimestamp ? timestamps.indexOf(selectedTimestamp) : 0
  );

  useEffect(() => {
    // Update the sliderValue whenever selectedTimestamp changes
    if (selectedTimestamp !== null) {
      const index = timestamps.indexOf(selectedTimestamp);
      setSliderValue(index);
    }
  }, [selectedTimestamp, timestamps]);

  return (
    <div className="timestamp-slider-container">
      <input
        type="range"
        min={0}
        max={timestamps.length - 1}
        step={1}
        value={sliderValue} // Use local state for the slider's value
        onChange={onSliderChange}
        className="timestamp-slider"
      />
      <p><b>Timestamp</b>: {selectedTimestamp} ({selectedTimestamp && formatDate(selectedTimestamp)})</p>
    </div>
  );
};

export default TimestampSlider;
