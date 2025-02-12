import { useState, useEffect } from 'react';
import Header from './components/Header/Header';
import TimestampSlider from './components/TimestampSlider/TimestampSlider';
import './App.css';
import SplittableCanvas from './components/SplittableCanvas/SplittableCanvas';
import FileInput from './components/FileInput/FileInput';
import { ThemeProvider } from '@mui/material/styles';
import darkTheme from './theme';
import Export from './components/Export/Export';

interface Node {
  id: number;
  direction?: 'horizontal' | 'vertical';
  left?: Node;
  right?: Node;
  size?: number;
}

interface NodeMetadata {
  topic: string | null;
  timestamp: number | null;
}

function App() {
  const [timestamps, setTimestamps] = useState<number[]>([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);
  const [topics, setTopics] = useState<string[]>([]);
  const [selectedRosbag, setSelectedRosbag] = useState<string | null>(null);
  const [isFileInputVisible, setIsFileInputVisible] = useState(false);
  const [isExportDialogVisible, setIsExportDialogVisible] = useState(false);

  const [currentRoot, setCurrentRoot] = useState<Node | null>(null); 
  const [currentMetadata, setCurrentMetadata] = useState<{ [id: number]: NodeMetadata }>({});
  const [canvasList, setCanvasList] = useState<{ [key: string]: { root: Node, metadata: { [id: number]: NodeMetadata } } }>({});

  //console.log("Current Root: ", currentRoot, "Current Metadata: ", currentMetadata);

  const handleCanvasChange = (root: Node, metadata: { [id: number]: NodeMetadata }) => {
    setCurrentRoot(root);
    setCurrentMetadata(metadata);
  };
  
  // Update selectedTimestamp when timestamps change
  useEffect(() => {
    if (timestamps.length > 0) {
      setSelectedTimestamp(timestamps[0]); // Default to the first timestamp
    } else {
      setSelectedTimestamp(null); // Clear the selection if no timestamps
    }
  }, [timestamps]); // This effect runs whenever `timestamps` changes

  // Update topics, timestamps and rosbag name if rosbag changes
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

    fetch('/api/get-selected-rosbag')
      .then((response) => response.json())
      .then((data) => {
        setSelectedRosbag(data.selectedRosbag);
      })
      .catch((error) => {
        console.error('Error fetching selected rosbag:', error);
      });

  }, [selectedRosbag]);

  useEffect(() => {
    handleLoadCanvas(""); // Load all canvases on startup
  }, []);

  const handleSliderChange = (value: number) => {
    setSelectedTimestamp(value);
  };

  const handleAddCanvas = async (name: string) => {
    if (!currentRoot || !selectedRosbag) return;
  
    const newCanvas = {
      root: currentRoot,
      metadata: currentMetadata,
      rosbag: selectedRosbag,
    };
  
    const updatedCanvasList = { ...canvasList, [name]: newCanvas };
    setCanvasList(updatedCanvasList);
  
    try {
      await fetch('/api/save-canvases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedCanvasList),
      });
    } catch (error) {
      console.error('Error saving canvas:', error);
    }
  };
  
  const handleLoadCanvas = async (name: string) => {
    try {
      const response = await fetch('/api/load-canvases');
      const data = await response.json();
      setCanvasList(data);
  
      if (data[name]) {
        setCurrentRoot(data[name].root);
        setCurrentMetadata(data[name].metadata);
        setSelectedRosbag(data[name].rosbag);
      }
    } catch (error) {
      console.error('Error loading canvases:', error);
    }
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
          //console.log("Timstamps update");
          fetch('/api/timestamps')
            .then((response) => response.json())
            .then((data) => {
              setTimestamps(data.timestamps);
            })
            .catch((error) => {
              console.error('Error fetching timestamps:', error);
            });
        }}
        onRosbagUpdate={() => {
          //console.log("Rosbag update");
          fetch('/api/get-selected-rosbag')
            .then((response) => response.json())
            .then((data) => {
              setSelectedRosbag(data.selectedRosbag);
            })
            .catch((error) => {
              console.error('Error fetching selected rosbag:', error);
            });
        }}
      />
      <Export
        timestamps={timestamps}
        topics={topics}
        isVisible={isExportDialogVisible}
        onClose={() => setIsExportDialogVisible(false)}
      />
      <div className="App" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <Header 
          setIsFileInputVisible={setIsFileInputVisible} 
          setIsExportDialogVisible={setIsExportDialogVisible} 
          selectedRosbag={selectedRosbag}
          handleLoadCanvas={handleLoadCanvas}
          handleAddCanvas={handleAddCanvas}
        />
        <SplittableCanvas 
          topics={topics} 
          selectedTimestamp={selectedTimestamp} 
          selectedRosbag={selectedRosbag}
          onCanvasChange={handleCanvasChange}
          currentRoot={currentRoot} // Pass currentRoot here
          currentMetadata={currentMetadata} // Pass currentMetadata here
        />
        <TimestampSlider
          timestamps={timestamps}
          selectedTimestamp={selectedTimestamp}
          onSliderChange={handleSliderChange}
          selectedRosbag={selectedRosbag}
        />
      </div>
    </ThemeProvider>
  );
}

export default App;