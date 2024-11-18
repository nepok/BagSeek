import React from 'react';
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
    const date = new Date(timestamp/1000000); // Divide by 1.000.000 to convert to seconds
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

  return (
    <div className="timestamp-slider-container">
      <input
        type="range"
        min={0}
        max={timestamps.length - 1}
        step={1}
        value={timestamps.indexOf(selectedTimestamp ?? 0)} // Map the timestamp to its index
        onChange={onSliderChange}
        className="timestamp-slider"
      />
      <p><b>Timestamp</b>: {selectedTimestamp} ({selectedTimestamp && formatDate(selectedTimestamp)})</p>
    </div>
  );
};

export default TimestampSlider;
