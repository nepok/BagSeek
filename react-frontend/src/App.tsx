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
  nodeTopic: string | null;
  nodeTopicType: string | null; // Type of the topic, e.g., "sensor_msgs/Image"
}

function App() {
  const [availableTimestamps, setAvailableTimestamps] = useState<number[]>([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);
  const [mappedTimestamps, setMappedTimestamps] = useState<{ [topic: string]: number }>({});
  const [availableTopics, setAvailableTopics] = useState<string[]>([]);
  const [availableTopicTypes, setAvailableTopicTypes] = useState<{ [topic: string]: string }>({});
  const [selectedRosbag, setSelectedRosbag] = useState<string | null>(null);
  const [isFileInputVisible, setIsFileInputVisible] = useState(false);
  const [isExportDialogVisible, setIsExportDialogVisible] = useState(false);

  const [currentRoot, setCurrentRoot] = useState<Node | null>(null); 
  const [currentMetadata, setCurrentMetadata] = useState<{ [id: number]: NodeMetadata }>({});
  const [canvasList, setCanvasList] = useState<{ [key: string]: { root: Node, metadata: { [id: number]: NodeMetadata } } }>({});
  const [searchMarks, setSearchMarks] = useState<{ value: number; label: string }[]>([]);

  const handleCanvasChange = (root: Node, metadata: { [id: number]: NodeMetadata }) => {
    setCurrentRoot(root);
    setCurrentMetadata(metadata);
  };
  
  // Update selectedTimestamp when availableTimestamps change
  useEffect(() => {
    if (availableTimestamps.length > 0) {
      setSelectedTimestamp(availableTimestamps[0]); // Default to the first timestamp
    } else {
      setSelectedTimestamp(null); // Clear the selection if no timestamps
    }
  }, [availableTimestamps]); // This effect runs whenever `availableTimestamps` changes

  
  useEffect(() => {
    const loadAllCanvases = async () => {
      try {
        const response = await fetch('/api/load-canvases');
        const data = await response.json();
        setCanvasList(data);
      } catch (error) {
        console.error('Error loading all canvases:', error);
      }
    };

    loadAllCanvases();
  }, []);

  const handleSliderChange = async (value: number) => {
    setSelectedTimestamp(value);
    
    try {
      const response = await fetch('/api/set-reference-timestamp', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ referenceTimestamp: value }),
      });

      const data = await response.json(); // Expecting { mappedTimestamps: { "/topic/name": real_ts, ... } }
      setMappedTimestamps(data.mappedTimestamps); // 👈 Store it here
    } catch (error) {
      console.error('Error sending reference timestamp:', error);
    }
  };

  const handleAddCanvas = async (name: string) => {
    if (!currentRoot || !selectedRosbag) return;
  
    const newCanvas = {
      root: currentRoot,
      metadata: currentMetadata,
      rosbag: selectedRosbag,
    };

    try {
      await fetch('/api/save-canvas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name,
          canvas: newCanvas
        }),
      });
      // Update local state
      setCanvasList(prev => ({
        ...prev,
        [name]: newCanvas
      }));
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
        onAvailableTopicsUpdate={() => {
          fetch('/api/get-available-topics')
            .then((response) => response.json())
            .then((data) => {
              setAvailableTopics(data.availableTopics);
            })
            .catch((error) => {
              console.error('Error fetching available topics:', error);
            });
        }}
        onAvailableTopicTypesUpdate={() => {
          fetch('/api/get-available-topic-types')
            .then((response) => response.json())
            .then((data) => {
              setAvailableTopicTypes(data.availableTopicTypes);
            })
            .catch((error) => {
              console.error('Error fetching available topic types:', error);
            });
        }}
        onAvailableTimestampsUpdate={() => {
          fetch('/api/get-available-timestamps')
            .then((response) => response.json())
            .then((data) => {
              setAvailableTimestamps(data.availableTimestamps);
            })
            .catch((error) => {
              console.error('Error fetching available timestamps:', error);
            });
        }}
        onSelectedRosbagUpdate={() => {
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
        timestamps={availableTimestamps}
        topics={availableTopics}
        topicTypes={availableTopicTypes}
        isVisible={isExportDialogVisible}
        onClose={() => setIsExportDialogVisible(false)}
        searchMarks={searchMarks}
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
          availableTopics={availableTopics} 
          availableTopicTypes={availableTopicTypes}
          mappedTimestamps={mappedTimestamps}
          selectedRosbag={selectedRosbag}
          onCanvasChange={handleCanvasChange}
          currentRoot={currentRoot} // Pass currentRoot here
          currentMetadata={currentMetadata} // Pass currentMetadata here
        />
        <TimestampSlider
          availableTimestamps={availableTimestamps}
          selectedTimestamp={selectedTimestamp}
          onSliderChange={handleSliderChange}
          selectedRosbag={selectedRosbag}
          searchMarks={searchMarks}
          setSearchMarks={setSearchMarks}
        />
      </div>
    </ThemeProvider>
  );
}

export default App;