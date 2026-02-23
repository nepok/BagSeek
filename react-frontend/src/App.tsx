import { useCallback, useEffect, useRef, useState } from 'react';
import { Routes, Route, Navigate, useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import { Box, Slider, Typography } from '@mui/material';
import Header from './components/Header/Header';
import TimestampPlayer from './components/TimestampPlayer/TimestampPlayer';
import './App.css';
import SplittableCanvas from './components/SplittableCanvas/SplittableCanvas';
import FileInput from './components/FileInput/FileInput';
import Export from './components/Export/Export';
import { ExportPreselectionProvider } from './components/Export/ExportPreselectionContext';
import { useError } from './components/ErrorContext/ErrorContext';
import GlobalSearch from './components/GlobalSearch/GlobalSearch';
import { SearchResultsCacheProvider } from './components/GlobalSearch/SearchCacheContext';
import PositionalOverview from './components/PositionalOverview/PositionalOverview';
import TractorLoader from './components/TractorLoader/TractorLoader';
import { sortTopicsObject } from './utils/topics';
import { extractRosbagName } from './utils/rosbag';
// import Login from './components/Login/Login';
// import PrivateRoute from './components/AuthContext/PrivateRoute';
// import { useAuth } from './components/AuthContext/AuthContext';

function TractorDebugPage() {
  const [progress, setProgress] = useState(0);
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100vh', width: '100%', gap: 3 }}>
      <TractorLoader progress={progress} />
      <Box sx={{ width: 'calc(100% - 100px)', px: '50px' }}>
        <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.5)', mb: 1 }}>
          Progress: {Math.round(progress * 100)}%
        </Typography>
        <Slider
          value={progress}
          onChange={(_, v) => { setProgress(v as number); }}
          min={0}
          max={1}
          step={0.01}
          valueLabelDisplay="auto"
        />
      </Box>
    </Box>
  );
}

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
  // const { isAuthenticated, authDisabled } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  
  const [timestampCount, setTimestampCount] = useState<number>(0);  // Total number of reference timestamps
  const [firstTimestampNs, setFirstTimestampNs] = useState<string | null>(null);  // First reference timestamp (ns string)
  const [lastTimestampNs, setLastTimestampNs] = useState<string | null>(null);  // Last reference timestamp (ns string)
  const [selectedTimestampIndex, setSelectedTimestampIndex] = useState<number | null>(null);  // Index of the currently selected timestamp
  const [selectedTimestamp, setSelectedTimestamp] = useState<string | null>(null);  // Currently selected timestamp (ns string from backend, for display)
  const [mappedTimestamps, setMappedTimestamps] = useState<{ [topic: string]: number }>({});  // Mapping from topic names to their respective timestamps at the selected reference
  const [mcapIdentifier, setMcapIdentifier] = useState<string | null>(null);  // MCAP identifier for the currently selected reference timestamp
  // Unified topics state: { topicName: messageType } - single source of truth for topics and their types
  const [availableTopics, setAvailableTopics] = useState<Record<string, string>>({});
  const [selectedRosbag, setSelectedRosbag] = useState<string | null>(null);  // Currently selected rosbag file identifier or name
  const [isFileInputVisible, setIsFileInputVisible] = useState(false);  // Controls visibility of the file input dialog component
  const [isExportDialogVisible, setIsExportDialogVisible] = useState(false);  // Controls visibility of the export dialog component
  const [mcapBoundaries, setMcapBoundaries] = useState<number[]>([]);  // Start indices of each MCAP file within available timestamps
  const [mcapIdentifiers, setMcapIdentifiers] = useState<string[]>([]);  // Actual MCAP identifiers parallel to mcapBoundaries (e.g. "48","49",... for multipart rosbags)

  const [currentRoot, setCurrentRoot] = useState<Node | null>(null);  // Root node of the current canvas layout representing the splits and content
  const [currentMetadata, setCurrentMetadata] = useState<{ [id: number]: NodeMetadata }>({});  // Metadata associated with nodes in the current canvas, keyed by node id
  const [canvasList, setCanvasList] = useState<{ [key: string]: { root: Node, metadata: { [id: number]: NodeMetadata } } }>({});  // Collection of saved canvases, keyed by canvas name, each with root and metadata
  const [searchMarks, setSearchMarks] = useState<{ value: number; rank?: number }[]>([]);  // Marks used for search heatmap in timestamp player (from search -> explore navigation)
  const [mcapHighlightMask, setMcapHighlightMask] = useState<boolean[]>([]);  // Per-MCAP boolean mask: true = inside polygon (from MAP -> explore navigation)

  // Refs to apply URL-provided state once data is ready
  const pendingTsRef = useRef<number | null>(null);
  const pendingMcapRef = useRef<number | null>(null); // MCAP id to jump to (resolved via mcapBoundaries)
  const pendingCanvasRef = useRef<{ root: Node; metadata: { [id: number]: NodeMetadata } } | null>(null);
  const pendingRosbagParamRef = useRef<string | null>(null);
  const isUpdatingTimestampRef = useRef<boolean>(false); // Track if we're updating timestamp from user action
  const switchingToRosbagRef = useRef<string | null>(null); // Track rosbag currently being switched to (sync guard against double-switch)
  const [isTimestampLoading, setIsTimestampLoading] = useState(false); // True during rosbag-switch until first set-reference-timestamp completes
  const rosbagSwitchPendingRef = useRef(false); // True from rosbag switch start until handleSliderChange settles

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

  // Helper: extract rosbag name, preserving parent for multipart rosbags.
  const getRosbagName = (value: string | null | undefined): string | null | undefined => {
    if (!value) return value as any;
    return extractRosbagName(value);
  };

  // Fetch unified topics (topicName -> messageType) from backend API
  const fetchAvailableTopics = async (rosbag?: string) => {
    try {
      const url = rosbag
        ? `/api/get-available-topics?rosbag=${encodeURIComponent(rosbag)}`
        : '/api/get-available-topics';
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch topics');
      }
      // Sort topics on frontend
      const sortedTopics = sortTopicsObject(data.topics || {});
      setAvailableTopics(sortedTopics);
    } catch (error) {
      setError('Error fetching available topics');
      console.error('Error fetching available topics:', error);
    }
  };

  // Fetch timestamp metadata and MCAP boundaries from backend (single endpoint)
  const fetchAvailableTimestampsAndDensity = async (rosbag?: string) => {
    try {
      const url = rosbag
        ? `/api/get-timestamp-summary?rosbag=${encodeURIComponent(rosbag)}`
        : '/api/get-timestamp-summary';
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch timestamp summary');
      }
      setSelectedTimestamp(null); // clear stale value so it never mismatches new firstTimestampNs/lastTimestampNs
      setTimestampCount(data.count ?? 0);
      setFirstTimestampNs(data.firstTimestampNs ?? null);
      setLastTimestampNs(data.lastTimestampNs ?? null);
      setMcapBoundaries(
        (data.mcapRanges ?? []).map((r: { startIndex: number }) => r.startIndex)
      );
      setMcapIdentifiers(
        (data.mcapRanges ?? []).map((r: { mcapIdentifier: string }) => String(r.mcapIdentifier))
      );
      setMcapHighlightMask([]);
    } catch (error) {
      setError('Error fetching available timestamps');
      console.error('Error fetching available timestamps:', error);
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

  // Reset canvas to empty state (root with id 1, no metadata)
  const handleResetCanvas = () => {
    const emptyRoot: Node = { id: 1 };
    const emptyMetadata: { [id: number]: NodeMetadata } = {};
    setCurrentRoot(emptyRoot);
    setCurrentMetadata(emptyMetadata);
    // Update URL when on /explore
    if (location.pathname.startsWith('/explore')) {
      const encoded = encodeCanvas({ root: emptyRoot, metadata: emptyMetadata });
      updateSearchParams({ canvas: encoded });
    }
  };
  
  // Effect to select initial timestamp index when count changes
  useEffect(() => {
    if (timestampCount > 0) {
      const tsParam = searchParams.get('ts');
      const tsFromUrl = tsParam ? Number(tsParam) : null;

      if (tsFromUrl !== null && !Number.isNaN(tsFromUrl) && pendingTsRef.current === null) {
        pendingTsRef.current = tsFromUrl;
      }

      if (pendingTsRef.current === null && (tsFromUrl === null || Number.isNaN(tsFromUrl))) {
        handleSliderChange(0); // Select first timestamp
      }
    } else {
      setSelectedTimestampIndex(null);
      setSelectedTimestamp(null);
    }
  }, [timestampCount, searchParams]);

  // Function to refresh canvas list from backend
  const refreshCanvasList = useCallback(async () => {
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
  }, [setError]);

  // Effect to load all saved canvases once authenticated
  useEffect(() => {
    // if (isAuthenticated || authDisabled) {
      refreshCanvasList();
    // }
  }, [refreshCanvasList]);

  // Handler for when user changes the timestamp slider (receives index)
  const handleSliderChange = async (index: number) => {
    isUpdatingTimestampRef.current = true;
    setSelectedTimestampIndex(index);
    // Sync index to URL if on /explore
    if (location.pathname.startsWith('/explore')) {
      const curr = searchParams.get('ts');
      const next = String(index);
      if (curr !== next) updateSearchParams({ ts: next });
    }

    try {
      const response = await fetch('/api/set-reference-timestamp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index, rosbag: selectedRosbag }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to set reference timestamp');
      }
      setSelectedTimestamp(data.referenceTimestamp || null); // ns string for display
      setMappedTimestamps(data.mappedTimestamps);
      setMcapIdentifier(data.mcapIdentifier || null);
    } catch (error) {
      setError('Error sending reference timestamp');
      console.error('Error sending reference timestamp:', error);
    } finally {
      // Only clear the loading indicator if this slider change was triggered by a rosbag switch
      if (rosbagSwitchPendingRef.current) {
        rosbagSwitchPendingRef.current = false;
        setIsTimestampLoading(false);
      }
      setTimeout(() => {
        isUpdatingTimestampRef.current = false;
      }, 100);
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
      // Refresh canvas list from backend to get full data structure
      await refreshCanvasList();
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

  // Handler: user picks a rosbag from the Popper selector
  const handleRosbagSelect = async (path: string) => {
    try {
      await fetch('/api/set-file-paths', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      });
      // Set rosbag BEFORE fetching timestamps so when timestampCount changes and
      // triggers handleSliderChange(0), selectedRosbag is already the new value.
      setSelectedRosbag(getRosbagName(path) ?? null);
      rosbagSwitchPendingRef.current = true;
      setIsTimestampLoading(true);
      await Promise.all([
        fetchAvailableTopics(path),
        fetchAvailableTimestampsAndDensity(path),
      ]);
    } catch (e) {
      setError('Error selecting rosbag');
      console.error('Error selecting rosbag:', e);
    }
  };

  // Keep selected rosbag in URL on change
  useEffect(() => {
    if (location.pathname.startsWith('/explore')) {
      if (selectedRosbag) {
        const curr = searchParams.get('rosbag');
        const next = getRosbagName(selectedRosbag);
        if (curr !== next) updateSearchParams({ rosbag: next as string });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRosbag]);

  // Apply URL query params when entering /explore
  useEffect(() => {
    if (!location.pathname.startsWith('/explore')) return;

    const rosbagParam = searchParams.get('rosbag');
    const tsParam = searchParams.get('ts');
    const mcapParam = searchParams.get('mcap');
    const canvasParam = searchParams.get('canvas');
    const highlightParam = searchParams.get('highlight');

    // 1) Ensure selected rosbag matches URL
    const ensureRosbag = async () => {
      if (!rosbagParam) return;
      // Already matches current state or already switching to this rosbag — skip
      if (getRosbagName(selectedRosbag) === rosbagParam) return;
      if (switchingToRosbagRef.current === rosbagParam) return;
      switchingToRosbagRef.current = rosbagParam;
      try {
        const res = await fetch('/api/get-file-paths');
        const data = await res.json();
        const match = (data.paths as string[]).find((p) => getRosbagName(p) === rosbagParam);
        if (match) {
          await fetch('/api/set-file-paths', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: match }),
          });
          // Set rosbag BEFORE fetching timestamps — same reason as handleRosbagSelect.
          setSelectedRosbag(getRosbagName(match) ?? null);
          rosbagSwitchPendingRef.current = true;
          setIsTimestampLoading(true);
          await Promise.all([
            fetchAvailableTopics(match),
            fetchAvailableTimestampsAndDensity(match),
          ]);
        }
      } catch (e) {
        console.error('Failed to set rosbag from URL', e);
      } finally {
        switchingToRosbagRef.current = null;
      }
    };

    ensureRosbag();

    // 2) Load marks from sessionStorage when navigating from search (arrow icon) to seed the heatmap
    if (rosbagParam && canvasParam) {
      setMcapHighlightMask([]);
      try {
        const canvas = decodeCanvas(canvasParam);
        if (canvas?.metadata) {
          const nodeId = Object.keys(canvas.metadata)[0];
          const topic = canvas.metadata[nodeId]?.nodeTopic;
          if (topic) {
            const marksKey = `marks_${rosbagParam}_${topic}`;
            const stored = sessionStorage.getItem(marksKey);
            if (stored) {
              const decoded = JSON.parse(stored);
              if (Array.isArray(decoded)) {
                const normalized = decoded.map((m: any) => {
                  if (m && typeof m === 'object' && typeof m.value === 'number') return { value: m.value, ...(m.rank != null ? { rank: m.rank } : {}) };
                  if (typeof m === 'number') return { value: m };
                  return null;
                }).filter(Boolean) as { value: number; rank?: number }[];
                setSearchMarks(normalized);
              } else {
                setSearchMarks([]);
              }
            } else {
              setSearchMarks([]);
            }
          } else {
            setSearchMarks([]);
          }
        } else {
          setSearchMarks([]);
        }
      } catch (e) {
        setSearchMarks([]);
      }
    } else {
      setSearchMarks([]);
    }

    // 3) Stash ts index to apply when data is ready
    if (tsParam) {
      const tsIndex = Number(tsParam);
      if (!Number.isNaN(tsIndex) && tsIndex !== selectedTimestampIndex) pendingTsRef.current = tsIndex;
    }
    // Stash mcap id (from positional-overview "Open in Explore") — resolved to a ts index once mcapBoundaries load
    if (mcapParam) {
      const mcapId = Number(mcapParam);
      if (!Number.isNaN(mcapId)) {
        pendingMcapRef.current = mcapId;
        // If same rosbag and boundaries are already loaded, resolve immediately (mcapBoundaries effect won't fire)
        if (mcapBoundaries.length > 0 && (!rosbagParam || getRosbagName(selectedRosbag) === rosbagParam)) {
          const mcapIdx = mcapIdentifiers.findIndex(id => id === String(mcapId));
          const tsIndex = mcapIdx >= 0 ? mcapBoundaries[mcapIdx] : (mcapBoundaries[mcapId] ?? mcapBoundaries[0]);
          pendingMcapRef.current = null;
          updateSearchParams({ mcap: null });
          if (timestampCount > 0) {
            handleSliderChange(Math.max(0, Math.min(tsIndex, timestampCount - 1)));
          } else {
            pendingTsRef.current = tsIndex;
          }
        }
      }
    }
    if (canvasParam) {
      const parsed = decodeCanvas(canvasParam);
      if (parsed && parsed.root && parsed.metadata) {
        const currentEncoded = currentRoot && currentMetadata ? encodeCanvas({ root: currentRoot, metadata: currentMetadata }) : '';
        if (canvasParam !== currentEncoded) {
          // Apply after rosbag is ensured to avoid topic reset flicker
          pendingCanvasRef.current = parsed as { root: Node; metadata: { [id: number]: NodeMetadata } };
          pendingRosbagParamRef.current = rosbagParam;
          // If the currently selected rosbag already matches the URL rosbag, apply immediately
          if (!rosbagParam || getRosbagName(selectedRosbag) === rosbagParam) {
            const { root, metadata } = pendingCanvasRef.current;
            pendingCanvasRef.current = null;
            pendingRosbagParamRef.current = null;
            setCurrentRoot(root);
            setCurrentMetadata(metadata);
            const encoded = encodeCanvas({ root, metadata });
            if (encoded) updateSearchParams({ canvas: encoded });
          }
        }
      }
    }
    // If timestamp index param exists and rosbag already matches, apply immediately
    if (tsParam && (!rosbagParam || getRosbagName(selectedRosbag) === rosbagParam) && !isUpdatingTimestampRef.current) {
      const tsIndex = Number(tsParam);
      if (!Number.isNaN(tsIndex) && tsIndex !== selectedTimestampIndex) {
        pendingTsRef.current = tsIndex;
        if (timestampCount > 0) {
          const clamped = Math.max(0, Math.min(pendingTsRef.current, timestampCount - 1));
          pendingTsRef.current = null;
          handleSliderChange(clamped);
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
    // Proceed when the selected rosbag basename matches the needed rosbag from URL
    if (needRosbag && getRosbagName(selectedRosbag) !== needRosbag) return;
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

  // Apply pending timestamp index once count is loaded
  useEffect(() => {
    if (timestampCount === 0) return;
    if (pendingTsRef.current !== null) {
      const clamped = Math.max(0, Math.min(pendingTsRef.current, timestampCount - 1));
      pendingTsRef.current = null;
      handleSliderChange(clamped);
    }
  }, [timestampCount]);

  // Resolve pending mcap navigation + highlight marks once mcapBoundaries arrive
  useEffect(() => {
    if (mcapBoundaries.length === 0) return;

    // Jump to the selected MCAP's first timestamp — look up by actual identifier
    if (pendingMcapRef.current !== null) {
      const identifierStr = String(pendingMcapRef.current);
      const mcapIdx = mcapIdentifiers.findIndex(id => id === identifierStr);
      const tsIndex = mcapIdx >= 0 ? mcapBoundaries[mcapIdx] : (mcapBoundaries[pendingMcapRef.current] ?? mcapBoundaries[0]);
      pendingMcapRef.current = null;
      pendingTsRef.current = tsIndex;
      // Remove mcap from URL immediately so the ts URL update doesn't re-trigger
      // the URL params effect with mcap still present, which would cause a second call.
      updateSearchParams({ mcap: null });
      if (timestampCount > 0) {
        const clamped = Math.max(0, Math.min(tsIndex, timestampCount - 1));
        pendingTsRef.current = null;
        handleSliderChange(clamped);
      }
    }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mcapBoundaries]);

  // Apply MCAP highlight mask from URL ?highlight=48,49,50 param (actual MCAP identifiers from MAP)
  const highlightParam = searchParams.get('highlight');
  useEffect(() => {
    if (!highlightParam || mcapBoundaries.length === 0 || mcapIdentifiers.length === 0) {
      if (!highlightParam) setMcapHighlightMask([]);
      return;
    }
    const ids = new Set(highlightParam.split(',').map(s => s.trim()));
    const mask = mcapIdentifiers.map(id => ids.has(id));
    if (mask.some(Boolean)) setMcapHighlightMask(mask);
  }, [highlightParam, mcapBoundaries, mcapIdentifiers]); // eslint-disable-line react-hooks/exhaustive-deps


  return (
    <Routes>
      {/* <Route path="/login" element={<Login />} /> */}
      <Route path="/*" element={
        // <PrivateRoute>
    <ExportPreselectionProvider onOpenExport={() => setIsExportDialogVisible(true)}>
    <>
      {/* File input dialog for uploading rosbag files and fetching initial data */}
      <FileInput
        isVisible={isFileInputVisible}
        onClose={() => setIsFileInputVisible(false)}
        onAvailableTopicsUpdate={fetchAvailableTopics}
        onAvailableTimestampsUpdate={fetchAvailableTimestampsAndDensity}
        onSelectedRosbagUpdate={(path) => setSelectedRosbag(getRosbagName(path) ?? null)}
      />
      {/* Export dialog to export data based on timestamps, topics, and search marks */}
      <Export
        timestampCount={timestampCount}
        availableTopics={availableTopics}
        isVisible={isExportDialogVisible}
        onClose={() => setIsExportDialogVisible(false)}
        selectedRosbag={selectedRosbag}
        preSelectedRosbag={selectedRosbag}
        openedFromMap={location.pathname.startsWith('/positional-overview')}
      />
      {/* Main application container with header, canvas, and timestamp player */}
      <div className="App" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        {/* Header bar with controls to open dialogs and load/save canvases */}
        <Header
          setIsFileInputVisible={setIsFileInputVisible}
          setIsExportDialogVisible={setIsExportDialogVisible}
          onRosbagSelect={handleRosbagSelect}
          selectedRosbag={selectedRosbag}
          handleLoadCanvas={handleLoadCanvas}
          handleAddCanvas={handleAddCanvas}
          handleResetCanvas={handleResetCanvas}
          availableTopics={availableTopics}
          canvasList={canvasList}
          refreshCanvasList={refreshCanvasList}
          currentMetadata={currentMetadata}
          selectedTimestampIndex={selectedTimestampIndex}
          searchMarks={searchMarks}
        />
        <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          <SearchResultsCacheProvider>
          <Routes>
            <Route path="/" element={<Navigate to="/explore" replace />} />
            <Route
              path="/explore"
              element={
                <>
                  <SplittableCanvas
                    availableTopics={availableTopics}
                    mappedTimestamps={mappedTimestamps}
                    selectedRosbag={selectedRosbag}
                    mcapIdentifier={mcapIdentifier}
                    onCanvasChange={handleCanvasChange}
                    currentRoot={currentRoot}
                    currentMetadata={currentMetadata}
                  />
                  <TimestampPlayer
                    timestampCount={timestampCount}
                    firstTimestampNs={firstTimestampNs}
                    lastTimestampNs={lastTimestampNs}
                    selectedTimestampIndex={selectedTimestampIndex}
                    selectedTimestamp={selectedTimestamp}
                    onSliderChange={handleSliderChange}
                    selectedRosbag={selectedRosbag}
                    searchMarks={searchMarks}
                    setSearchMarks={setSearchMarks}
                    mcapBoundaries={mcapBoundaries}
                    mcapIdentifiers={mcapIdentifiers}
                    mcapHighlightMask={mcapHighlightMask}
                    isTimestampLoading={isTimestampLoading}
                  />
                </>
              }
            />
            <Route path="/search" element={
              <Box sx={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                <GlobalSearch />
              </Box>
            } />
            <Route path="/positional-overview" element={<PositionalOverview />} />
            <Route path="/tractor" element={<TractorDebugPage />} />
            <Route path="*" element={<Navigate to="/explore" replace />} />
          </Routes>
          </SearchResultsCacheProvider>
        </Box>
      </div>
    </>
    </ExportPreselectionProvider>
        // </PrivateRoute>
      } />
    </Routes>
  );
}

export default App;
