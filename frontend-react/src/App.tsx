// App.tsx
import React, { useEffect, useState } from 'react';
import CanvasContainer from './components/CanvasContainer/CanvasContainer';
import TimestampSlider from './components/TimestampSlider/TimestampSlider';

function App() {
  const [timestamps, setTimestamps] = useState<number[]>([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);

  useEffect(() => {
    fetch('/api/timestamps')
      .then((response) => response.json())
      .then((data) => {
        setTimestamps(data.timestamps);
        if (data.timestamps.length > 0) {
          setSelectedTimestamp(data.timestamps[0]);
        }
      })
      .catch((error) => {
        console.error('Error fetching timestamps:', error);
      });
  }, []);

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newIndex = Number(event.target.value);
    setSelectedTimestamp(timestamps[newIndex]);
  };

  return (
    <div className="App" style={{ height: '100vh', position: 'relative' }}>
      {/* Render a full-page CanvasContainer */}
      <CanvasContainer selectedTimestamp={selectedTimestamp} />

      {/* Use the extracted TimestampSlider component */}
      <TimestampSlider
        timestamps={timestamps}
        selectedTimestamp={selectedTimestamp}
        onSliderChange={handleSliderChange}
      />
    </div>
  );
}

export default App;

