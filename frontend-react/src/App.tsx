import { useState, useEffect } from 'react';
import Header from './components/Header/Header';
import TimestampSlider from './components/TimestampSlider/TimestampSlider';
import './App.css'; // Import the CSS file
import SplittableCanvas from './components/SplittableCanvas/SplittableCanvas';
import FileInput from './components/FileInput/FileInput'; // Import FileInput
import { ThemeProvider } from '@mui/material/styles';
import darkTheme from './theme';

function App() {
  const [timestamps, setTimestamps] = useState<number[]>([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);
  const [topics, setTopics] = useState<string[]>([]);
  const [isFileInputVisible, setIsFileInputVisible] = useState(false);


  // Update selectedTimestamp when timestamps change
  useEffect(() => {
    if (timestamps.length > 0) {
      setSelectedTimestamp(timestamps[0]); // Default to the first timestamp
    } else {
      setSelectedTimestamp(null); // Clear the selection if no timestamps
    }
  }, [timestamps]); // This effect runs whenever `timestamps` changes


  useEffect(() => {
    fetch('/api/topics')
      .then((response) => response.json())
      .then((data) => {
        setTopics(data.topics);
      })
      .catch((error) => {
        console.error('Error fetching topics:', error);
      });

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

  const handleSliderChange = (value: number) => {
    setSelectedTimestamp(value);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <FileInput
        isVisible={isFileInputVisible}
        onClose={() => setIsFileInputVisible(false)}
        onTopicsUpdate={() => {
          fetch('/api/topics')
            .then((response) => response.json())
            .then((data) => {
              setTopics(data.topics);
            })
            .catch((error) => {
              console.error('Error fetching topics:', error);
            });
        }}
        onTimestampsUpdate={() => {
          console.log("Timstamps update");
          fetch('/api/timestamps')
            .then((response) => response.json())
            .then((data) => {
              setTimestamps(data.timestamps);
            })
            .catch((error) => {
              console.error('Error fetching timestamps:', error);
            });
        }}
      />
      <div className="App" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <Header setIsFloatingBoxVisible={setIsFileInputVisible} />
        <SplittableCanvas selectedTimestamp={selectedTimestamp} topics={topics} />
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