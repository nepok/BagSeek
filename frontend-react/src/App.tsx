// App.tsx
import { useState, useEffect } from 'react';
import Header from './components/Header/Header';
import TimestampSlider from './components/TimestampSlider/TimestampSlider';
import './App.css'; // Import the CSS file
import SplittableCanvas from './components/SplittableCanvas/SplittableCanvas';
import { ThemeProvider } from '@mui/material/styles';
import darkTheme from './theme';

function App() {
  const [timestamps, setTimestamps] = useState<number[]>([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);
  const [topics, setTopics] = useState<string[]>([]); // State to hold topics

  // Fetch available timestamps and topics when the component mounts
  useEffect(() => {
    // Fetch topics from the Flask API
    fetch('/api/topics')
      .then((response) => response.json())
      .then((data) => {
        setTopics(data.topics); // Set topics state
      })
      .catch((error) => {
        console.error('Error fetching topics:', error);
      });

    // Fetch timestamps from another API
    fetch('/api/timestamps')
      .then((response) => response.json())
      .then((data) => {
        setTimestamps(data.timestamps);
        if (data.timestamps.length > 0) {
          setSelectedTimestamp(data.timestamps[0]); // Set the first timestamp as default
        }
      })
      .catch((error) => {
        console.error('Error fetching timestamps:', error);
      });
  }, []);

  const handleSliderChange = (value: number) => {
    // Handle slider change
    setSelectedTimestamp(value); // Directly use the value instead of accessing timestamps array
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <div className="App" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <Header />
        <SplittableCanvas
          selectedTimestamp={selectedTimestamp}
          topics={topics}
        />
        <TimestampSlider
          timestamps={timestamps}
          selectedTimestamp={selectedTimestamp}
          onSliderChange={handleSliderChange}
        />
      </div>
    </ThemeProvider>
  );
}

export default App;