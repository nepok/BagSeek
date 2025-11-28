import { Select, MenuItem, Slider, InputLabel, FormControl, Checkbox, ListItemText, OutlinedInput, IconButton, SelectChangeEvent, Box, Typography, Popper, Paper, TextField, LinearProgress, ButtonGroup, Button, Chip, Tabs, Tab, FormControlLabel } from '@mui/material';
import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import RosbagOverview from '../RosbagOverview/RosbagOverview';
import { Center } from '@react-three/drei';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import DownloadIcon from '@mui/icons-material/Download';
import RoomIcon from '@mui/icons-material/Room';

// Hover-still helper: true after delay ms with no meaningful cursor movement
function useHoverStill(delay: number = 500, tolerancePx: number = 3) {
  const [still, setStill] = useState(false);
  const lastPos = useRef<{ x: number; y: number } | null>(null);
  const timer = useRef<number | null>(null);

  const clearTimer = () => {
    if (timer.current) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  };

  const scheduleCheck = (anchor: { x: number; y: number }) => {
    clearTimer();
    timer.current = window.setTimeout(() => {
      const end = lastPos.current || anchor;
      const dx = Math.abs(end.x - anchor.x);
      const dy = Math.abs(end.y - anchor.y);
      if (dx <= tolerancePx && dy <= tolerancePx) {
        setStill(true);
      }
    }, delay) as unknown as number;
  };

  const onPointerEnter = (e: React.PointerEvent) => {
    const pos = { x: e.clientX, y: e.clientY };
    lastPos.current = pos;
    setStill(false);
    scheduleCheck(pos);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const pos = { x: e.clientX, y: e.clientY };
    lastPos.current = pos;
    if (still) setStill(false);
    scheduleCheck(pos);
  };

  const onPointerLeave = () => {
    clearTimer();
    setStill(false);
  };

  useEffect(() => {
    return () => clearTimer();
  }, []);

  return { still, onPointerEnter, onPointerMove, onPointerLeave } as const;
}

type SearchResultItem = {
  rank: number;
  rosbag: string;
  embedding_path: string;
  similarityScore: number;
  topic: string;
  timestamp: string;
  minuteOfDay: string;
  model: string;
};

function ResultImageCard({
  result,
  url,
  onOpenExplore,
  onOpenDownload,
}: {
  result: SearchResultItem;
  url: string;
  onOpenExplore: () => void;
  onOpenDownload: () => void;
}) {
  const { still, onPointerEnter, onPointerMove, onPointerLeave } = useHoverStill(500, 3);

  return (
    <Box
      sx={{ position: 'relative', width: '100%' }}
      onPointerEnter={onPointerEnter}
      onPointerMove={onPointerMove}
      onPointerLeave={onPointerLeave}
    >
      <img
        src={url}
        alt="Result"
        style={{
          width: '100%',
          borderRadius: '4px',
          objectFit: 'cover',
          aspectRatio: '16/9',
          display: 'block',
        }}
      />

      <Chip
        label={`${result.rosbag}`}
        size="small"
        sx={{
          position: 'absolute',
          top: 4,
          left: 4,
          bgcolor: 'rgba(100, 85, 130, 0.6)',
          color: 'white'
        }}
      />
      <Chip
        label={result.minuteOfDay || '—'}
        size="small"
        sx={{
          position: 'absolute',
          top: 4,
          right: 4,
          bgcolor: 'rgba(50,50,50,0.6)',
          color: 'white'
        }}
      />
      <Chip
        label={`${result.topic || ''}`}
        size="small"
        sx={{
          position: 'absolute',
          bottom: 4,
          left: 4,
          bgcolor: 'rgba(204, 180, 159, 0.6)',
          color: 'white',
          maxWidth: '62%',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      />
      <Chip
        label={result.model || ''}
        size="small"
        sx={{
          position: 'absolute',
          bottom: 4,
          right: 4,
          bgcolor: 'rgba(120, 170, 200, 0.6)',
          color: 'white',
          maxWidth: '35%',
        }}
      />
      <IconButton
        color="primary"
        onClick={onOpenExplore}
        sx={{
          position: 'absolute',
          top: '45%',
          right: 6,
          transform: 'translateY(-50%)',
          bgcolor: 'rgba(120, 170, 200, 0.6)',
          color: 'white',
          p: 0.5,
          '& svg': {
            fontSize: 18,
          },
          '&:hover': {
            bgcolor: 'rgba(120, 170, 200, 0.8)',
          },
        }}
      >
        <KeyboardArrowRightIcon />
      </IconButton>
      <IconButton
        size="small"
        color="primary"
        onClick={onOpenDownload}
        sx={{
          position: 'absolute',
          top: '55%',
          right: 6,
          transform: 'translateY(-50%)',
          bgcolor: 'rgba(120, 170, 200, 0.6)',
          color: 'white',
          p: 0.5,
          '& svg': {
            fontSize: 18,
          },
          '&:hover': {
            bgcolor: 'rgba(120, 170, 200, 0.8)',
          },
        }}
      >
        <DownloadIcon />
      </IconButton>
    </Box>
  );
}

const GlobalSearch: React.FC = () => {

    const navigate = useNavigate();
    const [search, setSearch] = useState('');
    const [searchDone, setSearchDone] = useState(false);
    // View mode state: 'images' or 'rosbags'
    const [viewMode, setViewMode] = useState<'images' | 'rosbags'>(() => 'images');
    const [searchResults, setSearchResults] = useState<{ rank: number, rosbag: string, mcap_identifier: string, embedding_path: string, similarityScore: number, topic: string, timestamp: string, minuteOfDay: string, model: string }[]>([]);
    const [marksPerTopic, setMarksPerTopic] = useState<{ [model: string]: { [rosbag: string]: { [topic: string]: { marks: { value: number }[] } } } }>({});
    const [searchStatus, setSearchStatus] = useState<{progress: number, status: string, message: string}>({progress: 0, status: 'idle', message: ''});

    const searchIconRef = useRef<HTMLDivElement | null>(null);
    const searchInputRef = useRef<HTMLInputElement | null>(null);

    const [models, setModels] = useState<string[]>([]);
    const [rosbags, setRosbags] = useState<string[]>([]);
    const DEFAULT_MODEL = 'ViT-B-16-quickgelu__openai';
    const DEFAULT_ROSBAG_PATH = '/mnt/data/rosbags/output_bag';
    const [confirmedModels, setConfirmedModels] = useState<string[]>([]);
    const [confirmedRosbags, setConfirmedRosbags] = useState<string[]>([]);
    const [timeRange, setTimeRange] = useState<number[]>([0, 1439]);
    const [sampling, setSampling] = useState<number>(10); // Default to 1, which is 10^0
    const [enhancePrompt, setEnhancePrompt] = useState<boolean>(true);
    const [enhancedPrompt, setEnhancedPrompt] = useState<string>('');
    const [isEnhancing, setIsEnhancing] = useState<boolean>(false);
    const samplingSteps = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
      20, 30, 40, 50, 60, 70, 80, 90, 100,
      200, 300, 400, 500, 600, 700, 800, 900, 1000
    ];
    const render_limit = 12; // Max images to show in grid

    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [availableRosbags, setAvailableRosbags] = useState<string[]>([]);
    const [positionallyFilteredRosbags, setPositionallyFilteredRosbags] = useState<string[] | null>(null);

    const getBasename = (fullPath: string) => fullPath.split(/[\\/]/).pop() ?? fullPath;
    
    // Compute filtered rosbags based on positional filter
    const filteredAvailableRosbags = positionallyFilteredRosbags
        ? availableRosbags.filter(rosbagPath => {
            const basename = getBasename(rosbagPath);
            return positionallyFilteredRosbags.includes(basename);
          })
        : availableRosbags;

    // In-memory cache shared across route switches in this tab
    type Cache = {
      search?: string;
      models?: string[];
      rosbags?: string[];
      viewMode?: 'images' | 'rosbags';
      searchResults?: { rank: number, rosbag: string, mcap_identifier: string, embedding_path: string, similarityScore: number, topic: string, timestamp: string, minuteOfDay: string, model: string }[];
      marksPerTopic?: { [model: string]: { [rosbag: string]: { [topic: string]: { marks: { value: number }[] } } } };
      searchDone?: boolean;
      confirmedModels?: string[];
      confirmedRosbags?: string[];
      searchStatus?: {progress: number, status: string, message: string};
    };
    const GS_CACHE_KEY = '__BagSeekGlobalSearchCache';
    const cacheRef: Cache = (globalThis as any)[GS_CACHE_KEY] || ((globalThis as any)[GS_CACHE_KEY] = {});
    const GS_SESSION_KEY = '__BagSeekSearchNewCache';

    // Time slider: minutes of the day (0-1439)
    const timestamps: number[] = Array.from({ length: 1440 }, (_, i) => i); // minutes of the day

    // Helper for time slider label: "hh:mm"
    const valueLabelFormat = (value: number) => {
      const hours = Math.floor(value / 60).toString().padStart(2, '0');
      const minutes = (value % 60).toString().padStart(2, '0');
      return `${hours}:${minutes}`;
    };

    useEffect(() => {
        fetch('/api/get-models')
            .then(res => res.json())
            .then(data => setAvailableModels(data.models || []))
            .catch(err => console.error('Failed to fetch models:', err));

        fetch('/api/get-file-paths')
            .then(res => res.json())
            .then(data => setAvailableRosbags(data.paths || []))
            .catch(err => console.error('Failed to fetch rosbags:', err));
    }, []);

    // Restore cached state on mount
    useEffect(() => {
        // Restore from in-memory first
        if (cacheRef.search !== undefined) setSearch(cacheRef.search);
        if (cacheRef.models !== undefined) setModels(cacheRef.models);
        if (cacheRef.rosbags !== undefined) setRosbags(cacheRef.rosbags);
        if (cacheRef.viewMode !== undefined) setViewMode(cacheRef.viewMode);
        if (cacheRef.searchResults !== undefined) setSearchResults(cacheRef.searchResults);
        if (cacheRef.marksPerTopic !== undefined) setMarksPerTopic(cacheRef.marksPerTopic as any);
        if (cacheRef.searchDone !== undefined) setSearchDone(cacheRef.searchDone);
        if (cacheRef.confirmedModels !== undefined) setConfirmedModels(cacheRef.confirmedModels);
        if (cacheRef.confirmedRosbags !== undefined) setConfirmedRosbags(cacheRef.confirmedRosbags);
        if (cacheRef.searchStatus !== undefined) setSearchStatus(cacheRef.searchStatus);

        // If no in-memory results, restore last successful results from session
        const hasMem = Array.isArray(cacheRef.searchResults) && cacheRef.searchResults.length > 0;
        if (!hasMem) {
          try {
            const raw = sessionStorage.getItem(GS_SESSION_KEY);
            if (raw) {
              const saved = JSON.parse(raw);
              if (saved) {
                if (saved.query) setSearch(saved.query);
                if (Array.isArray(saved.results)) setSearchResults(saved.results);
                if (saved.marksPerTopic) setMarksPerTopic(saved.marksPerTopic);
                if (Array.isArray(saved.confirmedModels)) setConfirmedModels(saved.confirmedModels);
                if (Array.isArray(saved.confirmedRosbags)) setConfirmedRosbags(saved.confirmedRosbags);
                if (Array.isArray(saved.timeRange)) setTimeRange(saved.timeRange);
                if (typeof saved.sampling === 'number') setSampling(saved.sampling);
                setSearchDone(Boolean(saved.results));
                setSearchStatus({ progress: 0, status: 'idle', message: '' });
              }
            }
          } catch {}
        }

        // Restore last selected tab from sessionStorage
        try {
          const lastTab = sessionStorage.getItem('lastSearchTab');
          if (lastTab === 'images' || lastTab === 'rosbags') {
            setViewMode(lastTab);
          }
        } catch {}
        
        // Restore positional filter from sessionStorage
        const loadPositionalFilter = () => {
          try {
            const positionalFilterRaw = sessionStorage.getItem('__BagSeekPositionalFilter');
            if (positionalFilterRaw) {
              const filtered = JSON.parse(positionalFilterRaw);
              if (Array.isArray(filtered) && filtered.length > 0) {
                setPositionallyFilteredRosbags(filtered);
              } else {
                setPositionallyFilteredRosbags(null);
              }
            } else {
              setPositionallyFilteredRosbags(null);
            }
          } catch {}
        };
        
        loadPositionalFilter();
        
        // Listen for storage changes to update filter immediately
        const handleStorageChange = (e: StorageEvent) => {
          if (e.key === '__BagSeekPositionalFilter') {
            loadPositionalFilter();
          }
        };
        
        window.addEventListener('storage', handleStorageChange);
        
        // Also listen for custom event for same-tab updates
        const handleCustomStorageChange = () => {
          loadPositionalFilter();
        };
        
        window.addEventListener('__BagSeekPositionalFilterChanged', handleCustomStorageChange);
        
        return () => {
          window.removeEventListener('storage', handleStorageChange);
          window.removeEventListener('__BagSeekPositionalFilterChanged', handleCustomStorageChange);
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Persist to in-memory cache whenever state changes
    useEffect(() => {
        cacheRef.search = search;
        cacheRef.models = models;
        cacheRef.rosbags = rosbags;
        cacheRef.viewMode = viewMode;
        cacheRef.searchResults = searchResults;
        cacheRef.marksPerTopic = marksPerTopic as any;
        cacheRef.searchDone = searchDone;
        cacheRef.confirmedModels = confirmedModels;
        cacheRef.confirmedRosbags = confirmedRosbags;
        cacheRef.searchStatus = searchStatus;
    }, [search, models, rosbags, viewMode, searchResults, marksPerTopic, searchDone, confirmedModels, confirmedRosbags, searchStatus]);

    // Preselect default model once models are loaded
    useEffect(() => {
        if (availableModels.length > 0 && models.length === 0) {
            if (availableModels.includes(DEFAULT_MODEL)) {
                setModels([DEFAULT_MODEL]);
            }
        }
    }, [availableModels]);

    // Preselect default rosbag path once rosbags are loaded
    useEffect(() => {
        if (availableRosbags.length > 0 && rosbags.length === 0) {
            const match = availableRosbags.find(p => p === DEFAULT_ROSBAG_PATH);
            if (match) setRosbags([match]);
        }
    }, [availableRosbags]);
    
    // Filter out selected rosbags that don't match positional filter
    useEffect(() => {
        if (positionallyFilteredRosbags && rosbags.length > 0) {
            const filtered = rosbags.filter(rosbagPath => {
                const basename = getBasename(rosbagPath);
                return positionallyFilteredRosbags.includes(basename);
            });
            if (filtered.length !== rosbags.length) {
                setRosbags(filtered);
            }
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [positionallyFilteredRosbags]);

    // Poll search status periodically when running, stop after 3 failed fetches
    useEffect(() => {
        let retryCount = 0;
        let interval: NodeJS.Timeout;

        if (searchStatus.status === 'running') {
            interval = setInterval(async () => {
                try {
                    const response = await fetch('/api/search-status');
                    const data = await response.json();
                    setSearchStatus(data);
                    retryCount = 0; // reset on success
                } catch (err) {
                    console.error('Failed to fetch search status:', err);
                    retryCount++;
                    if (retryCount >= 3) {
                        console.warn('Stopping search status polling after 3 failed attempts');
                        clearInterval(interval);
                    }
                }
            }, 1000);
        }

        return () => clearInterval(interval);
    }, [searchStatus.status]);


    const handleKeyDown = async (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter' && search.trim()) {
            // Validate that at least one rosbag is selected
            if (rosbags.length === 0) {
                return; // Prevent search if no rosbags selected
            }
            
            // Clear cache and UI before starting a new search
            try { sessionStorage.removeItem(GS_SESSION_KEY); } catch {}
            setSearchResults([]);
            setMarksPerTopic({});
            setConfirmedModels(models);
            setConfirmedRosbags(rosbags);
            setSearchDone(false);
            setSearchStatus({ progress: 0, status: 'running', message: 'Starting search...' });
            try {
                // Wait for enhancement if enhancePrompt is enabled
                let queryToUse = search;
                if (enhancePrompt) {
                    try {
                        const enhanceResponse = await fetch(`/api/enhance-prompt?prompt=${encodeURIComponent(search)}`);
                        const enhanceData = await enhanceResponse.json();
                        if (enhanceData.enhanced) {
                            queryToUse = enhanceData.enhanced;
                            setEnhancedPrompt(enhanceData.enhanced);
                        }
                    } catch (error) {
                        console.error('Failed to enhance prompt:', error);
                        // Continue with original prompt if enhancement fails
                    }
                }
                
                const modelParams = models.join(',');
                const rosbagParams = rosbags.join(',');
                const timeRangeParam = timeRange.join(',');
                const accuracyParam = sampling.toString();
                const queryParams = new URLSearchParams({
                  query: queryToUse,
                  models: modelParams,
                  rosbags: rosbagParams,
                  timeRange: timeRangeParam,
                  accuracy: accuracyParam
                }).toString();
                const response = await fetch(`/api/search?${queryParams}`, { method: 'GET' });
                const data = await response.json();
                setSearchResults(data.results || []);
                setMarksPerTopic(data.marksPerTopic || {});
                setSearchDone(true);
                setConfirmedModels(models);
                setConfirmedRosbags(rosbags);
                // Persist successful response to session so results survive navigation
                try {
                    sessionStorage.setItem(GS_SESSION_KEY, JSON.stringify({
                      query: search,
                      results: data.results || [],
                      marksPerTopic: data.marksPerTopic || {},
                      confirmedModels: models,
                      confirmedRosbags: rosbags,
                      timeRange,
                      sampling,
                    }));
                } catch {}
            } catch (error) {
                console.error('Search failed', error);
                // Leave UI empty on failure since we cleared before starting
                setSearchDone(true);
            }
        }
    };

    // Handler for selecting models (with SELECT ALL logic)
    const handleModelSelection = (event: SelectChangeEvent<string[]>) => {
        const value = event.target.value as string[];
        let newSelection: string[] = [];

        if (value.includes("ALL")) {
            newSelection =
                models.length === availableModels.length ? [] : availableModels;
        } else {
            newSelection = value;
        }

        setModels(newSelection);
    };

    // Handler for selecting rosbags (with SELECT ALL and CURRENTLY SELECTED logic)
    const handleRosbagSelection = (event: SelectChangeEvent<string[]>) => {
        const value = event.target.value as string[];
        let newSelection: string[] = [];
        const rosbagsToUse = positionallyFilteredRosbags ? filteredAvailableRosbags : availableRosbags;

        if (value.includes("ALL")) {
            newSelection =
                rosbags.length === rosbagsToUse.length ? [] : rosbagsToUse;
        } else if (value.includes("currently selected")) {
            const matched = rosbagsToUse.find(
                bag => searchResults[0]?.rosbag && bag.split("/").pop() === searchResults[0].rosbag
            );
            if (matched) {
                newSelection = [matched];
            }
        } else {
            newSelection = value;
        }

        setRosbags(newSelection);
    };

    const openExplorePage = (result: { rosbag: string; topic: string; timestamp: string }) => {
        if (!result || !result.rosbag || !result.topic || !result.timestamp) return;
        // Cache current tab before navigating
        try { sessionStorage.setItem('lastSearchTab', viewMode); } catch {}
        // Build a single-panel canvas JSON containing only the selected topic
        const canvas = {
          root: { id: 1 },
          metadata: {
            1: { nodeTimestamp: result.timestamp, nodeTopic: result.topic, nodeTopicType: "sensor_msgs/msg/CompressedImage" },
          },
        };
        const encodedCanvas = encodeURIComponent(JSON.stringify(canvas));
        // Navigate to explore with parsed params
        const params = new URLSearchParams();
        params.set('rosbag', result.rosbag);
        params.set('ts', String(result.timestamp));
        params.set('canvas', encodedCanvas);
        navigate(`/explore?${params.toString()}`);
    }

    interface TabPanelProps {
      children?: React.ReactNode;
      index: number;
      value: number;
    }

    function a11yProps(index: number) {
      return {
        id: `simple-tab-${index}`,
        'aria-controls': `simple-tabpanel-${index}`,
      };
    }

    return (
        <>
        <Box sx={{ width: '80%', mx: 'auto', mt: 4 }}>
            <Box sx={{ display: 'flex', gap: 1, mb: 1, width: '100%' }}>
                {/* Positional Filter */}
                <Button
                    variant={positionallyFilteredRosbags ? "contained" : "outlined"}
                    color={positionallyFilteredRosbags ? "primary" : "secondary"}
                    onClick={(e) => {
                        if (e.ctrlKey || e.metaKey) {
                            // Clear filter on Ctrl/Cmd+Click
                            setPositionallyFilteredRosbags(null);
                            try {
                                sessionStorage.removeItem('__BagSeekPositionalFilter');
                            } catch {}
                        } else {
                            navigate('/positional-overview');
                        }
                    }}
                    sx={{
                        alignSelf: 'stretch',
                        minWidth: 0,
                        px: 2,
                        whiteSpace: 'nowrap',
                        position: 'relative'
                    }}
                >
                    <RoomIcon />
                    <Typography variant="body2" sx={{ ml: 1 }}>Positional Filter</Typography>
                    {positionallyFilteredRosbags && (
                        <Chip
                            label={positionallyFilteredRosbags.length}
                            size="small"
                            color="primary"
                            sx={{
                                ml: 1,
                                height: 20,
                                fontSize: '0.7rem',
                                '& .MuiChip-label': {
                                    px: 0.75
                                }
                            }}
                        />
                    )}
                </Button>

                {/* Rosbags */}
                <FormControl 
                    size="small" 
                    sx={{ flex: 1, minWidth: 0 }}
                    error={rosbags.length === 0}
                >
                    <InputLabel>Rosbags</InputLabel>
                    <Select
                        multiple
                        value={rosbags}
                        onChange={handleRosbagSelection}
                        input={<OutlinedInput label="Rosbags" />}
                        renderValue={(selected) => (selected as string[]).length > 0 ? (selected as string[]).join(', ') : ''}
                        MenuProps={{
                          anchorOrigin: { vertical: 'bottom', horizontal: 'center' },
                          transformOrigin: { vertical: 'top', horizontal: 'center' },
                          PaperProps: {
                            sx: {
                              minWidth: '380px',
                              maxWidth: '600px',
                              maxHeight: '80vh',
                              mt: 0,
                              px: 2,
                              pt: 1,
                              overflowX: 'auto',
                              '& .MuiMenuItem-root': {
                                minHeight: '24px',
                                fontSize: '0.75rem',
                                py: 0.25,
                                whiteSpace: 'normal',
                                wordBreak: 'break-all',
                              }
                            }
                          }
                        }}
                    >
                        <MenuItem value="ALL">
                            <Checkbox checked={rosbags.length === filteredAvailableRosbags.length && filteredAvailableRosbags.length > 0} />
                            <ListItemText primary="SELECT ALL" />
                        </MenuItem>
                        <MenuItem value="currently selected">
                            <Checkbox checked={(() => {
                                const matched = filteredAvailableRosbags.find(b => b.split('/').pop() === searchResults[0]?.rosbag);
                                return rosbags.includes(matched || '');
                            })()} />
                            <ListItemText primary="CURRENTLY SELECTED" />
                        </MenuItem>
                        {availableRosbags.map((name) => {
                            const basename = getBasename(name);
                            const isFiltered = positionallyFilteredRosbags ? positionallyFilteredRosbags.includes(basename) : true;
                            const isInFilteredList = !positionallyFilteredRosbags || isFiltered;
                            
                            return (
                                <MenuItem 
                                    key={name} 
                                    value={name}
                                    disabled={!isInFilteredList}
                                    sx={{
                                        opacity: isInFilteredList ? 1 : 0.5
                                    }}
                                >
                                    <Checkbox checked={rosbags.includes(name)} disabled={!isInFilteredList} />
                                    <ListItemText primary={getBasename(name)} />
                                </MenuItem>
                            );
                        })}
                    </Select>
                </FormControl>

                {/* Time Range */}
                <FormControl size="small" sx={{ flex: 1, minWidth: 0 }}>
                    <InputLabel id="time-range-label">Time Range</InputLabel>
                    <Select
                        labelId="time-range-label"
                        value="timeRange"
                        displayEmpty
                        renderValue={() => `${valueLabelFormat(timeRange[0])} – ${valueLabelFormat(timeRange[1])}`}
                        input={<OutlinedInput label="Time Range" />}
                        MenuProps={{
                          anchorOrigin: { vertical: 'bottom', horizontal: 'center' },
                          transformOrigin: { vertical: 'top', horizontal: 'center' },
                          PaperProps: {
                            sx: {
                              width: '300px',
                              maxHeight: '80vh',
                              mt: 0,
                              px: 2,
                              pt: 1,
                              '& .MuiMenuItem-root': {
                                minHeight: '24px',
                                fontSize: '0.75rem',
                                py: 0.25,
                              }
                            }
                          }
                        }}
                    >
                        <MenuItem
                          disableRipple
                          sx={{
                            display: 'block',
                            '&:hover': {
                              backgroundColor: 'transparent',
                            },
                          }}
                        >
                          <Box sx={{ position: 'relative', width: '100%', pt: 2, pb: 0 }}>
                            <Slider
                              value={timeRange}
                              onChange={(_, newValue) => setTimeRange(newValue as number[])}
                              valueLabelDisplay="auto"
                              min={0}
                              max={1439}
                              valueLabelFormat={valueLabelFormat}
                              marks={[
                                { value: 0, label: '00:00' },
                                { value: 360, label: '06:00' },
                                { value: 720, label: '12:00' },
                                { value: 1080, label: '18:00' },
                                { value: 1439, label: '23:59' },
                              ]}
                            />
                          </Box>
                        </MenuItem>
                    </Select>
                </FormControl>

                {/* Tradeoff */}
                <FormControl size="small" sx={{ flex: 1, minWidth: 0 }}>
                    <InputLabel id="sampling-label">Sampling</InputLabel>
                    <Select
                        labelId="sampling-label"
                        value="sampling"
                        displayEmpty
                        renderValue={() => `${sampling}`}
                        input={<OutlinedInput label="Sampling" />}
                        MenuProps={{
                          anchorOrigin: { vertical: 'bottom', horizontal: 'right' },
                          transformOrigin: { vertical: 'top', horizontal: 'right' },
                          PaperProps: {
                            sx: {
                              width: '300px',
                              maxHeight: '80vh',
                              mt: 0,
                              px: 2,
                              pt: 1,
                              '& .MuiMenuItem-root': {
                                minHeight: '24px',
                                fontSize: '0.75rem',
                                py: 0.25,
                              }
                            }
                          }
                        }}
                    >
                        <MenuItem
                          disableRipple
                          sx={{
                            display: 'block',
                            '&:hover': {
                              backgroundColor: 'transparent',
                            },
                          }}
                        >
                          <Box sx={{ position: 'relative', width: '100%', pt: 2, pb: 0 }}>
                            <Slider
                              value={samplingSteps.indexOf(sampling)}
                              onChange={(_, newValue) => {
                                const actual = samplingSteps[newValue as number];
                                setSampling(actual);
                              }}
                              min={0}
                              max={samplingSteps.length - 1}
                              step={null} // disables in-between steps
                              marks={samplingSteps.map((val, idx) => ({
                                value: idx,
                                label: [1, 10, 100, 1000].includes(val) ? `${val}` : '',
                              }))}
                              scale={(index) => samplingSteps[index]}
                              valueLabelFormat={() => `${sampling}`}
                              valueLabelDisplay="auto"
                            />
                          </Box>
                        </MenuItem>
                    </Select>
                </FormControl>

                {/* Models */}
                <FormControl 
                  size="small" 
                  sx={{ flex: 1, minWidth: 0 }}
                  error={models.length === 0}
                >
                    <InputLabel>Models</InputLabel>
                    <Select
                        multiple
                        value={models}
                        onChange={handleModelSelection}
                        input={<OutlinedInput label="Models" />}
                        renderValue={(selected) => (selected as string[]).join(', ')}
                        MenuProps={{
                          anchorOrigin: { vertical: 'bottom', horizontal: 'left' },
                          transformOrigin: { vertical: 'top', horizontal: 'left' },
                          PaperProps: {
                            sx: {
                              width: '300px',
                              maxHeight: '80vh',
                              mt: 0,
                              '& .MuiMenuItem-root': {
                                minHeight: '24px',
                                fontSize: '0.75rem',
                                py: 0.25,
                              }
                            }
                          }
                        }}
                    >
                        <MenuItem value="ALL">
                            <Checkbox checked={models.length === availableModels.length && availableModels.length > 0} />
                            <ListItemText primary="SELECT ALL" />
                        </MenuItem>
                        {availableModels.map((name) => (
                            <MenuItem key={name} value={name}>
                                <Checkbox checked={models.includes(name)} />
                                <ListItemText primary={name} />
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
            </Box>
            <Box ref={searchIconRef}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TextField
                        fullWidth
                        label="Search"
                        variant="outlined"
                        value={enhancePrompt && enhancedPrompt && !isEnhancing ? enhancedPrompt : search}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                            setSearch(e.target.value);
                            // Clear enhanced prompt when user starts typing
                            if (enhancedPrompt) {
                                setEnhancedPrompt('');
                            }
                        }}
                        onKeyDown={handleKeyDown}
                        InputProps={{
                            sx: {
                                '& input': {
                                    color: 'white',
                                }
                            }
                        }}
                    />
                    <FormControlLabel
                      control={<Checkbox checked={enhancePrompt} onChange={(event: React.ChangeEvent<HTMLInputElement>) => setEnhancePrompt(event.target.checked)} />}
                      label="ENHANCE"
                      sx={{
                        ml: 0.0,
                        color: 'rgba(255,255,255,0.7)',
                        userSelect: 'none',
                        '& .MuiFormControlLabel-label': {
                          fontSize: '0.875rem',
                        }
                      }}
                    />
                </Box>
            </Box>

            {/* Tabs are now always visible, above search status and viewMode blocks */}
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs 
                variant="fullWidth"
                value={viewMode === 'images' ? 0 : 1}
                onChange={(_, newValue) => setViewMode(newValue === 0 ? 'images' : 'rosbags')} 
                aria-label="basic tabs example">
                <Tab label="Images" {...a11yProps(0)} />
                <Tab label="Rosbags" {...a11yProps(1)} />
              </Tabs>
            </Box>

            {searchStatus.status !== 'idle' && searchResults.length === 0 && (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column', mt: 4, mb: 4 }}>
                <Box sx={{ width: '50%' }}>
                  {searchStatus.status !== 'done' && (
                    <LinearProgress variant="determinate" value={searchStatus.progress * 100} />
                  )}
                </Box>
                {searchStatus.message && (
                  <Typography variant="body1" sx={{ mt: 2, color: 'white', textAlign: 'center', whiteSpace: 'pre-line' }}>
                    {searchStatus.message}
                  </Typography>
                )}
              </Box>
            )}
            {viewMode === 'images' && (
              searchResults.length > 0 && (
                <Box
                  sx={{
                    padding: '8px',
                    background: '#121212',
                    borderRadius: '8px',
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(30%, 1fr))',
                    gap: 2,
                    maxHeight: 'calc(100vh - 200px)',
                    overflowY: 'auto',
                    mt: 2,
                  }}
                >
                  {searchResults.slice(0, render_limit).map((result, index) => {
                    const url = result.topic && result.timestamp && result.rosbag && result.mcap_identifier
                      ? `http://localhost:5000/images/${result.rosbag}/${result.topic.replace(/\//g, '_').replace(/^_/, '')}/${result.mcap_identifier}/${result.timestamp}.png`
                      : undefined;

                    return (
                      <Box
                        key={index}
                        sx={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'flex-start',
                        }}
                      >
                        {url && (
                          <ResultImageCard
                            result={result as any}
                            url={url}
                            onOpenExplore={() => openExplorePage(result)}
                            onOpenDownload={() => window.open(url, '_blank')}
                          />
                        )}
                      </Box>
                    );
                  })}
                </Box>
              )
            )}
            {viewMode === 'rosbags' && (
              <Box sx={{ mt: 4 }}>
                <Box>
                  {searchDone && (
                    <RosbagOverview
                      rosbags={confirmedRosbags}
                      models={confirmedModels}
                      searchDone={searchDone}
                      marksPerTopic={marksPerTopic}
                    />
                  )}
                </Box>
              </Box>
            )}
        </Box>
        </>
    );
};

export default GlobalSearch;
