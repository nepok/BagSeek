// App.tsx
import React, { useState, useEffect } from 'react';
import Header from './components/Header/Header';
import TimestampSlider from './components/TimestampSlider/TimestampSlider';
import './App.css'; // Import the CSS file
import SplittableCanvas from './components/SplittableCanvas/SplittableCanvas';
import { ThemeProvider } from '@mui/material/styles';
import theme from './theme'
import darkTheme from './theme';

interface Container {
  id: number;
  width: number;
  height: number;
}

function App() {
  const [timestamps, setTimestamps] = useState<number[]>([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);
  const [topics, setTopics] = useState<string[]>([]); // State to hold topics
  const [containers, setContainers] = useState<Container[]>([
    { id: 1, width: 100, height: 100 } // Initial full-size container
  ]);

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

  // Handle container splitting logic
  const handleCreateCanvas = (id: number, direction: 'below' | 'right') => {
    setContainers((prevContainers) => {
      const newContainers = prevContainers.map((container) => {
        if (container.id === id) {
          if (direction === 'below') {
            return { ...container, height: container.height / 2 }; // Halve the height of the original container
          } else if (direction === 'right') {
            return { ...container, width: container.width / 2 }; // Halve the width of the original container
          }
        }
        return container;
      });

      // Create a new container with either width or height adjusted based on direction
      const newContainer: Container = {
        id: prevContainers.length + 1,
        width: direction === 'right' ? 50 : 100, // Adjust width based on direction
        height: direction === 'below' ? 50 : 100, // Adjust height based on direction
      };

      return [...newContainers, newContainer];
    });
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <div className="App" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <Header />
        <SplittableCanvas
          selectedTimestamp={selectedTimestamp}
          topics={topics}
        />
        {/*<CanvasContainer
          selectedTimestamp={selectedTimestamp}
          topics={topics}
        />*/}
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