// components/TimestampSlider.tsx
import React from 'react';

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
  return (
    <div
      style={{
        position: 'absolute',
        bottom: 0,
        left: '50%',
        transform: 'translateX(-50%)',
        width: '80%',
        padding: '20px',
      }}
    >
      <h2>Select a Timestamp</h2>
      <input
        type="range"
        min={0}
        max={timestamps.length - 1}
        step={1}
        value={timestamps.indexOf(selectedTimestamp ?? 0)} // Map the timestamp to its index
        onChange={onSliderChange}
        style={{ width: '100%' }}
      />
      <p>Timestamp: {selectedTimestamp}</p>
    </div>
  );
};

export default TimestampSlider;
