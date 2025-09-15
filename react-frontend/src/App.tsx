import { useEffect, useRef, useState } from 'react';
import { Routes, Route, Navigate, useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import Header from './components/Header/Header';
import TimestampPlayer from './components/TimestampPlayer/TimestampPlayer';
import './App.css';
import SplittableCanvas from './components/SplittableCanvas/SplittableCanvas';
import FileInput from './components/FileInput/FileInput';
import Export from './components/Export/Export';
import { useError } from './components/ErrorContext/ErrorContext';
import GlobalSearch from './components/GlobalSearch/GlobalSearch';

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

  const { setError } = useError(); // Hook to set global error messages for user feedback
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  
  const [availableTimestamps, setAvailableTimestamps] = useState<number[]>([]);  // List of all available timestamps fetched from backend
  const [selectedTimestamp, setSelectedTimestamp] = useState<number | null>(null);  // Currently selected timestamp for playback or display
  const [timestampDensity, setTimestampDensity] = useState<number[]>([]);  // Density information of timestamps, used for UI visualization like heatmaps
  const [mappedTimestamps, setMappedTimestamps] = useState<{ [topic: string]: number }>({});  // Mapping from topic names to their respective timestamps at the selected reference
  const [availableTopics, setAvailableTopics] = useState<string[]>([]);  // List of all available topics fetched from backend
  const [availableTopicTypes, setAvailableTopicTypes] = useState<{ [topic: string]: string }>({});  // Mapping from topic names to their types, e.g., sensor_msgs/Image
  const [selectedRosbag, setSelectedRosbag] = useState<string | null>(null);  // Currently selected rosbag file identifier or name
  const [isFileInputVisible, setIsFileInputVisible] = useState(false);  // Controls visibility of the file input dialog component
  const [isExportDialogVisible, setIsExportDialogVisible] = useState(false);  // Controls visibility of the export dialog component

  const [currentRoot, setCurrentRoot] = useState<Node | null>(null);  // Root node of the current canvas layout representing the splits and content
  const [currentMetadata, setCurrentMetadata] = useState<{ [id: number]: NodeMetadata }>({});  // Metadata associated with nodes in the current canvas, keyed by node id
  const [canvasList, setCanvasList] = useState<{ [key: string]: { root: Node, metadata: { [id: number]: NodeMetadata } } }>({});  // Collection of saved canvases, keyed by canvas name, each with root and metadata
  const [searchMarks, setSearchMarks] = useState<{ value: number; label: string }[]>([]);  // Marks used for search highlighting in timestamp player, each with value and label

  // Refs to apply URL-provided state once data is ready
  const pendingTsRef = useRef<number | null>(null);
  const pendingCanvasRef = useRef<{ root: Node; metadata: { [id: number]: NodeMetadata } } | null>(null);
  const pendingRosbagParamRef = useRef<string | null>(null);

  // Helpers to encode/decode canvas JSON into query param
  const encodeCanvas = (canvas: { root: Node; metadata: { [id: number]: NodeMetadata } }) => {
    try {
      return encodeURIComponent(JSON.stringify(canvas));
    } catch (e) {
      console.error('Failed to encode canvas', e);
      return '';
    }
  };

  const decodeCanvas = (encoded: string) => {
    try {
      return JSON.parse(decodeURIComponent(encoded));
    } catch (e) {
      console.error('Failed to decode canvas', e);
      return null;
    }
  };

  // Update URL search params helper (merges provided keys)
  const updateSearchParams = (updates: Record<string, string | null | undefined>) => {
    const next = new URLSearchParams(searchParams as any);
    Object.entries(updates).forEach(([k, v]) => {
      if (v === null || v === undefined || v === '') next.delete(k);
      else next.set(k, String(v));
    });
    setSearchParams(next, { replace: true });
  };

  // Fetch list of available topics from backend API
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

  // Fetch mapping of topics to their types from backend API
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

  // Fetch both available timestamps and their density information from backend
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

  // Fetch the currently selected rosbag identifier from backend
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

  // Update current canvas root node and metadata when user changes the canvas
  const handleCanvasChange = (root: Node, metadata: { [id: number]: NodeMetadata }) => {
    setCurrentRoot(root);
    setCurrentMetadata(metadata);
    // Persist canvas in URL when on /explore
    if (location.pathname.startsWith('/explore')) {
      const encoded = encodeCanvas({ root, metadata });
      const current = searchParams.get('canvas') || '';
      if (encoded && encoded !== current) updateSearchParams({ canvas: encoded });
    }
  };
  
  // Effect to update selected timestamp when available timestamps change
  useEffect(() => {
    // If timestamps are available, select the first one by default
    if (availableTimestamps.length > 0) {
      // If a timestamp is pending from URL, let that effect apply it
      if (pendingTsRef.current === null) {
        setSelectedTimestamp(availableTimestamps[0]); // Use first timestamp
        handleSliderChange(availableTimestamps[0]); // Sync mapped timestamps on change
      }
    } else {
      setSelectedTimestamp(null); // Clear selection if no timestamps
    }
  }, [availableTimestamps]);

  // Effect to load all saved canvases once on component mount
  useEffect(() => {
    const loadAllCanvases = async () => {
      try {
        const response = await fetch('/api/load-canvases');
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'Failed to load canvases');
        }
        setCanvasList(data); // Populate canvas list with saved canvases
      } catch (error) {
        setError('Error loading canvases');
        console.error('Error loading all canvases:', error);
      }
    };

    loadAllCanvases();
  }, []);

  // Handler for when user changes the timestamp slider
  const handleSliderChange = async (value: number) => {
    setSelectedTimestamp(value); // Update selected timestamp in state
    // Sync to URL if on /explore
    if (location.pathname.startsWith('/explore')) {
      const curr = searchParams.get('ts');
      const next = String(value);
      if (curr !== next) updateSearchParams({ ts: next });
    }
    
    try {
      // Notify backend of new reference timestamp to get mapped timestamps
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
      setMappedTimestamps(data.mappedTimestamps); // Update mapped timestamps for topics
    } catch (error) {
      setError('Error sending reference timestamp');
      console.error('Error sending reference timestamp:', error);
    }
  };

  // Handler to save the current canvas layout under a given name
  const handleAddCanvas = async (name: string) => {
    if (!currentRoot || !selectedRosbag) return; // Require current canvas and rosbag
  
    const newCanvas = {
      root: currentRoot,
      metadata: currentMetadata,
      rosbag: selectedRosbag,
    };

    try {
      // Send new canvas data to backend to save
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
      // Update local canvas list with newly saved canvas
      setCanvasList(prev => ({
        ...prev,
        [name]: newCanvas
      }));
    } catch (error) {
      setError('Error saving canvas');
      console.error('Error saving canvas:', error);
    }
  };
  
  // Handler to load a saved canvas by name and update current canvas state
  const handleLoadCanvas = async (name: string) => {
    try {
      const response = await fetch('/api/load-canvases');
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to load canvases');
      }
      if (data[name]) {
        setCurrentRoot(data[name].root); // Set root node of loaded canvas
        setCurrentMetadata(data[name].metadata); // Set metadata of loaded canvas
        // Also push into URL
        if (location.pathname.startsWith('/explore')) {
          const encoded = encodeCanvas({ root: data[name].root, metadata: data[name].metadata });
          if (encoded) updateSearchParams({ canvas: encoded });
        }
      }
    } catch (error) {
      setError('Error loading canvases');
      console.error('Error loading canvases:', error);
    }
  };

  // Keep selected rosbag in URL on change
  useEffect(() => {
    if (location.pathname.startsWith('/explore')) {
      if (selectedRosbag) {
        const curr = searchParams.get('rosbag');
        if (curr !== selectedRosbag) updateSearchParams({ rosbag: selectedRosbag });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRosbag]);

  // Apply URL query params when entering /explore
  useEffect(() => {
    if (!location.pathname.startsWith('/explore')) return;

    const rosbagParam = searchParams.get('rosbag');
    const tsParam = searchParams.get('ts');
    const canvasParam = searchParams.get('canvas');

    // 1) Ensure selected rosbag matches URL
    const ensureRosbag = async () => {
      if (!rosbagParam) return;
      if (selectedRosbag === rosbagParam) return;
      try {
        const res = await fetch('/api/get-file-paths');
        const data = await res.json();
        const match = (data.paths as string[]).find((p) => p.split('/').pop() === rosbagParam);
        if (match) {
          await fetch('/api/set-file-paths', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: match }),
          });
          // Refresh dependent data
          await Promise.all([
            fetchAvailableTopics(),
            fetchAvailableTopicTypes(),
            fetchAvailableTimestampsAndDensity(),
            fetchSelectedRosbag(),
          ]);
        }
      } catch (e) {
        console.error('Failed to set rosbag from URL', e);
      }
    };

    ensureRosbag();

    // 2) Stash ts/canvas to apply when data is ready
    if (tsParam) {
      const tsNum = Number(tsParam);
      if (!Number.isNaN(tsNum) && tsNum !== selectedTimestamp) pendingTsRef.current = tsNum;
    }
    if (canvasParam) {
      const parsed = decodeCanvas(canvasParam);
      if (parsed && parsed.root && parsed.metadata) {
        const currentEncoded = currentRoot && currentMetadata ? encodeCanvas({ root: currentRoot, metadata: currentMetadata }) : '';
        if (canvasParam !== currentEncoded) {
          // Apply after rosbag is ensured to avoid topic reset flicker
          pendingCanvasRef.current = parsed as { root: Node; metadata: { [id: number]: NodeMetadata } };
          pendingRosbagParamRef.current = rosbagParam;
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname, location.search]);

  // Apply pending canvas once rosbag is ready/matching
  useEffect(() => {
    if (!location.pathname.startsWith('/explore')) return;
    if (!pendingCanvasRef.current) return;
    const needRosbag = pendingRosbagParamRef.current;
    if (needRosbag && selectedRosbag !== needRosbag) return;
    const { root, metadata } = pendingCanvasRef.current;
    pendingCanvasRef.current = null;
    pendingRosbagParamRef.current = null;
    setCurrentRoot(root);
    setCurrentMetadata(metadata);
    // ensure URL reflects exactly what's applied
    const encoded = encodeCanvas({ root, metadata });
    if (encoded) updateSearchParams({ canvas: encoded });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRosbag]);

  // Apply pending timestamp once timestamps are loaded
  useEffect(() => {
    if (!availableTimestamps || availableTimestamps.length === 0) return;
    if (pendingTsRef.current !== null) {
      // If exact match not found, pick nearest
      const target = pendingTsRef.current;
      let chosen = target;
      if (!availableTimestamps.includes(target)) {
        let best = availableTimestamps[0];
        let bestDiff = Math.abs(best - target);
        for (const t of availableTimestamps) {
          const diff = Math.abs(t - target);
          if (diff < bestDiff) {
            best = t;
            bestDiff = diff;
          }
        }
        chosen = best;
      }
      pendingTsRef.current = null;
      handleSliderChange(chosen);
    }
  }, [availableTimestamps]);


  return (
    <>
      {/* File input dialog for uploading rosbag files and fetching initial data */}
      <FileInput
        isVisible={isFileInputVisible}
        onClose={() => setIsFileInputVisible(false)}
        onAvailableTopicsUpdate={fetchAvailableTopics}
        onAvailableTopicTypesUpdate={fetchAvailableTopicTypes}
        onAvailableTimestampsUpdate={fetchAvailableTimestampsAndDensity}
        onSelectedRosbagUpdate={fetchSelectedRosbag}
      />
      {/* Export dialog to export data based on timestamps, topics, and search marks */}
      <Export
        timestamps={availableTimestamps}
        timestampDensity={timestampDensity}
        topics={availableTopics}
        topicTypes={availableTopicTypes}
        isVisible={isExportDialogVisible}
        onClose={() => setIsExportDialogVisible(false)}
        searchMarks={searchMarks}
      />
      {/* Main application container with header, canvas, and timestamp player */}
      <div className="App" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        {/* Header bar with controls to open dialogs and load/save canvases */}
        <Header 
          setIsFileInputVisible={setIsFileInputVisible} 
          setIsExportDialogVisible={setIsExportDialogVisible} 
          selectedRosbag={selectedRosbag}
          handleLoadCanvas={handleLoadCanvas}
          handleAddCanvas={handleAddCanvas}
        />
        <Routes>
          <Route path="/" element={<Navigate to="/explore" replace />} />
          <Route
            path="/explore"
            element={
              <>
                <SplittableCanvas
                  availableTopics={availableTopics}
                  availableTopicTypes={availableTopicTypes}
                  mappedTimestamps={mappedTimestamps}
                  selectedRosbag={selectedRosbag}
                  onCanvasChange={handleCanvasChange}
                  currentRoot={currentRoot}
                  currentMetadata={currentMetadata}
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
              </>
            }
          />
          <Route path="/search" element={<GlobalSearch />} />
          <Route path="*" element={<Navigate to="/explore" replace />} />
        </Routes>
      </div>
    </>
  );
}

export default App;
