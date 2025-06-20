import { useState, useEffect } from 'react';
import Header from './components/Header/Header';
import TimestampPlayer from './components/TimestampPlayer/TimestampPlayer';
import './App.css';
import SplittableCanvas from './components/SplittableCanvas/SplittableCanvas';
import FileInput from './components/FileInput/FileInput';
import Export from './components/Export/Export';
import { useError } from './components/ErrorContext/ErrorContext';

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

  const { setError } = useError();

  const [availableTimestamps, setAvailableTimestamps] = useState<number[]>([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);
  const [timestampDensity, setTimestampDensity] = useState<number[]>([]);
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

  const fetchAvailableTopics = async () => {
    try {
      const response = await fetch('/api/get-available-topics');
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch topics');
      } 
      setAvailableTopics(data.availableTopics);
    } catch (error) {
      setError('Error fetching available topics');
      console.error('Error fetching available topics:', error);
    }
  };

  const fetchAvailableTopicTypes = async () => {
    try {
      const response = await fetch('/api/get-available-topic-types');
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch topic types');
      }
      setAvailableTopicTypes(data.availableTopicTypes);
    } catch (error) {
      setError('Error fetching available topic types');
      console.error('Error fetching available topic types:', error);
    }
  };

  const fetchAvailableTimestampsAndDensity = async () => {
    try {
      const response = await fetch('/api/get-available-timestamps');
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch timestamps');
      }
      setAvailableTimestamps(data.availableTimestamps);
    } catch (error) {
      setError('Error fetching available timestamps');
      console.error('Error fetching available timestamps:', error);
    }

    try {
      const response = await fetch('/api/get-timestamp-density');
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch timestamp density');
      }
      setTimestampDensity(data.timestampDensity);
    } catch (error) {
      setError('Error fetching timestamp density');
      console.error('Error fetching timestamp density:', error);
    }
  };

  const fetchSelectedRosbag = async () => {
    try {
      const response = await fetch('/api/get-selected-rosbag');
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch selected rosbag');
      }
      setSelectedRosbag(data.selectedRosbag);
    } catch (error) {
      setError('Error fetching selected rosbag');
      console.error('Error fetching selected rosbag:', error);
    }
  };

  const handleCanvasChange = (root: Node, metadata: { [id: number]: NodeMetadata }) => {
    setCurrentRoot(root);
    setCurrentMetadata(metadata);
  };
  
  // Update selectedTimestamp when availableTimestamps change
  useEffect(() => {
    if (availableTimestamps.length > 0) {
      setSelectedTimestamp(availableTimestamps[0]); // Default to the first timestamp
      handleSliderChange(availableTimestamps[0]); // Call handleSliderChange with the new timestamp
    } else {
      setSelectedTimestamp(null); // Clear the selection if no timestamps
    }
  }, [availableTimestamps]); // This effect runs whenever `availableTimestamps` changes

  useEffect(() => {
    const loadAllCanvases = async () => {
      try {
        const response = await fetch('/api/load-canvases');
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'Failed to load canvases');
        }
        setCanvasList(data);
      } catch (error) {
        setError('Error loading canvases');
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

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to set reference timestamp');
      }
      setMappedTimestamps(data.mappedTimestamps);
    } catch (error) {
      setError('Error sending reference timestamp');
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
      const response = await fetch('/api/save-canvas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name,
          canvas: newCanvas
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to save canvas');
      }
      // Update local state
      setCanvasList(prev => ({
        ...prev,
        [name]: newCanvas
      }));
    } catch (error) {
      setError('Error saving canvas');
      console.error('Error saving canvas:', error);
    }
  };
  
  const handleLoadCanvas = async (name: string) => {
    try {
      const response = await fetch('/api/load-canvases');
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to load canvases');
      }
      if (data[name]) {
        setCurrentRoot(data[name].root);
        setCurrentMetadata(data[name].metadata);
      }
    } catch (error) {
      setError('Error loading canvases');
      console.error('Error loading canvases:', error);
    }
  };

  return (
    <>
      <FileInput
        isVisible={isFileInputVisible}
        onClose={() => setIsFileInputVisible(false)}
        onAvailableTopicsUpdate={fetchAvailableTopics}
        onAvailableTopicTypesUpdate={fetchAvailableTopicTypes}
        onAvailableTimestampsUpdate={fetchAvailableTimestampsAndDensity}
        onSelectedRosbagUpdate={fetchSelectedRosbag}
      />
      <Export
        timestamps={availableTimestamps}
        timestampDensity={timestampDensity}
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
        <TimestampPlayer
          availableTimestamps={availableTimestamps}
          timestampDensity={timestampDensity}
          selectedTimestamp={selectedTimestamp}
          onSliderChange={handleSliderChange}
          selectedRosbag={selectedRosbag}
          searchMarks={searchMarks}
          setSearchMarks={setSearchMarks}
        />
      </div>
    </>
  );
}

export default App;