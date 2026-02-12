import { Slider, Checkbox, ListItemText, IconButton, Box, Typography, TextField, LinearProgress, Button, Chip, Tabs, Tab, FormControlLabel, Collapse } from '@mui/material';
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import RosbagOverview from '../RosbagOverview/RosbagOverview';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import DownloadIcon from '@mui/icons-material/Download';
import RoomIcon from '@mui/icons-material/Room';
import FilterListIcon from '@mui/icons-material/FilterList';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SelectAllIcon from '@mui/icons-material/SelectAll';
import { extractRosbagName } from '../../utils/rosbag';
import { sortTopics } from '../../utils/topics';
import McapRangeFilter, { McapRangeFilterItem, McapFilterState, formatNsToTime } from '../McapRangeFilter/McapRangeFilter';
import TractorLoader from '../TractorLoader/TractorLoader';

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
  mcap_identifier?: string;
};

function ResultImageCard({
  result,
  onOpenExplore,
  onOpenDownload,
}: {
  result: SearchResultItem;
  onOpenExplore: () => void;
  onOpenDownload: () => void;
}) {
  const { still, onPointerEnter, onPointerMove, onPointerLeave } = useHoverStill(500, 3);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!result.topic || !result.timestamp || !result.mcap_identifier) {
      setIsLoading(false);
      return;
    }

    const fetchImage = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(
          `/api/content-mcap?rosbag=${result.rosbag}&topic=${encodeURIComponent(result.topic)}&mcap_identifier=${result.mcap_identifier}&timestamp=${result.timestamp}`
        );
        const data = await response.json();

        if (data.error) {
          console.error("API error:", data.error);
          setIsLoading(false);
          return;
        }

        if (data.type === 'image' && data.image) {
          const format = data.format || 'jpeg';
          setImageUrl(`data:image/${format};base64,${data.image}`);
        }
      } catch (error) {
        console.error("Error fetching image:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchImage();
  }, [result.topic, result.timestamp, result.mcap_identifier]);

  return (
    <Box
      sx={{ position: 'relative', width: '100%' }}
      onPointerEnter={onPointerEnter}
      onPointerMove={onPointerMove}
      onPointerLeave={onPointerLeave}
    >
      {isLoading ? (
        <Box
          sx={{
            width: '100%',
            aspectRatio: '16/9',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: '#1e1e1e',
            borderRadius: '4px',
          }}
        >
          <Typography variant="body2" color="text.secondary">
            Loading...
          </Typography>
        </Box>
      ) : imageUrl ? (
      <img
          src={imageUrl}
        alt="Result"
        style={{
          width: '100%',
          borderRadius: '4px',
          objectFit: 'cover',
          aspectRatio: '16/9',
          display: 'block',
        }}
      />
      ) : (
        <Box
          sx={{
            width: '100%',
            aspectRatio: '16/9',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: '#1e1e1e',
            borderRadius: '4px',
          }}
        >
          <Typography variant="body2" color="text.secondary">
            No image available
          </Typography>
        </Box>
      )}

      <Chip
        label={`${result.rosbag}`}
        size="small"
        sx={{
          position: 'absolute',
          top: 4,
          left: 4,
          bgcolor: (theme) => `${theme.palette.primary.main}99`,
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
          bgcolor: (theme) => `${theme.palette.warning.main}99`,
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
          bgcolor: (theme) => `${theme.palette.secondary.main}99`,
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
        onClick={() => {
          if (imageUrl) {
            // Extract format from data URL (e.g., "data:image/jpeg;base64,..." or "data:image/png;base64,...")
            const formatMatch = imageUrl.match(/data:image\/([^;]+)/);
            const format = formatMatch ? formatMatch[1] : 'jpeg';
            const extension = format === 'png' ? 'png' : 'jpg';
            
            // Convert base64 to blob and download
            const byteCharacters = atob(imageUrl.split(',')[1]);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
              byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], { type: `image/${format}` });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${result.rosbag}_${result.timestamp}.${extension}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
          } else {
            onOpenDownload();
          }
        }}
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
    const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const abortControllerRef = useRef<AbortController | null>(null);
    const hasAttemptedRefreshForPositionalMismatchRef = useRef<string | null>(null);

    const [models, setModels] = useState<string[]>([]);
    const [rosbags, setRosbags] = useState<string[]>([]);
    const DEFAULT_MODEL = 'ViT-B-16-quickgelu__openai';
    const [confirmedModels, setConfirmedModels] = useState<string[]>([]);
    const [confirmedRosbags, setConfirmedRosbags] = useState<string[]>([]);
    const [timeRange, setTimeRange] = useState<number[]>([0, 1439]);
    const [mcapFilters, setMcapFilters] = useState<McapFilterState>({});
    // Pending MCAP IDs from positional filter (map view), keyed by rosbag name
    const [pendingMcapIds, setPendingMcapIds] = useState<Record<string, string[]> | null>(null);
    const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
    const [availableImageTopics, setAvailableImageTopics] = useState<Record<string, Record<string, string[]>>>({});
    const [topicTypes, setTopicTypes] = useState<Record<string, string>>({});
    const [sampling, setSampling] = useState<number>(10); // Default to 1, which is 10^0
    const [enhancePrompt, setEnhancePrompt] = useState<boolean>(true);
    const [enhancedPrompt, setEnhancedPrompt] = useState<string>('');
    const [isEnhancing, setIsEnhancing] = useState<boolean>(false);
    const samplingSteps = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
      20, 30, 40, 50, 60, 70, 80, 90, 100,
      200, 300, 400, 500, 600, 700, 800, 900, 1000
    ];
    const initial_render_limit = 12; // Initial images to show in grid
    const [displayedResultsCount, setDisplayedResultsCount] = useState<number>(initial_render_limit);
    const [filtersOpen, setFiltersOpen] = useState(true);
    const [expandedRosbags, setExpandedRosbags] = useState(true);
    const [expandedTopics, setExpandedTopics] = useState(true);
    const [expandedTimeRange, setExpandedTimeRange] = useState(true);
    const [expandedSampling, setExpandedSampling] = useState(true);
    const [expandedModels, setExpandedModels] = useState(true);

    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [availableRosbags, setAvailableRosbags] = useState<string[]>([]);
    const [positionallyFilteredRosbags, setPositionallyFilteredRosbags] = useState<string[] | null>(null);
    const [isRefreshingFilePaths, setIsRefreshingFilePaths] = useState(false);

    const getBasename = (fullPath: string) => fullPath.split(/[\\/]/).pop() ?? fullPath;
    
    // Compute filtered rosbags based on positional filter
    const filteredAvailableRosbags = positionallyFilteredRosbags
        ? availableRosbags.filter(rosbagPath => {
            const name = extractRosbagName(rosbagPath);
            return positionallyFilteredRosbags.some(f => extractRosbagName(f) === name);
          })
        : availableRosbags;

    // Compute all unique topics from availableImageTopics, sorted by topics.ts priority
    const allTopics = useMemo(() => {
        const topicSet = new Set<string>();
        for (const modelData of Object.values(availableImageTopics)) {
            for (const topics of Object.values(modelData)) {
                for (const t of topics) {
                    topicSet.add(t);
                }
            }
        }
        return sortTopics(Array.from(topicSet), topicTypes);
    }, [availableImageTopics, topicTypes]);

    // Auto-select all topics when they first become available
    const initialTopicsSelected = useRef(false);
    useEffect(() => {
        if (!initialTopicsSelected.current && allTopics.length > 0) {
            initialTopicsSelected.current = true;
            setSelectedTopics(allTopics);
        }
    }, [allTopics]);

    // Compute which rosbags are available for selected topics (checks across ALL models)
    const topicAvailableRosbags = useMemo((): Set<string> | null => {
        if (selectedTopics.length === 0) return null;
        const available = new Set<string>();
        for (const modelData of Object.values(availableImageTopics)) {
            for (const rosbag of availableRosbags) {
                const rosbagName = extractRosbagName(rosbag);
                const topics = modelData[rosbagName];
                if (topics && topics.some(t => selectedTopics.includes(t))) {
                    available.add(rosbag);
                }
            }
        }
        return available;
    }, [selectedTopics, availableRosbags, availableImageTopics]);

    // Compute which models are available for selected topics (checks across ALL rosbags)
    const topicAvailableModels = useMemo((): Set<string> | null => {
        if (selectedTopics.length === 0) return null;
        const available = new Set<string>();
        for (const [model, modelData] of Object.entries(availableImageTopics)) {
            for (const topics of Object.values(modelData)) {
                if (topics.some(t => selectedTopics.includes(t))) {
                    available.add(model);
                    break;
                }
            }
        }
        return available;
    }, [selectedTopics, availableImageTopics]);

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

    // When positional filter is active, compare counts and refresh file paths if they mismatch (stale cache)
    useEffect(() => {
        if (!positionallyFilteredRosbags || positionallyFilteredRosbags.length === 0) {
            hasAttemptedRefreshForPositionalMismatchRef.current = null;
            return;
        }
        const positionalCount = positionallyFilteredRosbags.length;
        const availableCount = availableRosbags.filter(rosbagPath =>
            positionallyFilteredRosbags.some(f => extractRosbagName(f) === extractRosbagName(rosbagPath))
        ).length;
        if (positionalCount === availableCount) return;

        const filterKey = positionallyFilteredRosbags.slice().sort().join(',');
        if (hasAttemptedRefreshForPositionalMismatchRef.current === filterKey) return;
        hasAttemptedRefreshForPositionalMismatchRef.current = filterKey;

        console.log(`[GlobalSearch] Positional filter mismatch: positional=${positionalCount}, available=${availableCount} - refreshing file paths`);
        setIsRefreshingFilePaths(true);
        fetch('/api/refresh-file-paths', { method: 'POST' })
            .then((res) => res.json())
            .then((data) => {
                if (data.paths) {
                    setAvailableRosbags(data.paths);
                }
            })
            .catch((err) => console.error('Failed to refresh file paths:', err))
            .finally(() => {
                // Defer clearing so auto-select effect runs first with new availableRosbags, then pending MCAP effect runs with full rosbags
                setTimeout(() => setIsRefreshingFilePaths(false), 0);
            });
    }, [positionallyFilteredRosbags, availableRosbags]);

    // Restore cached state on mount
    useEffect(() => {
        // If coming from "Apply to Search", clear in-memory cache so we start fresh
        try {
            if (sessionStorage.getItem('__BagSeekApplyToSearchJustNavigated') === '1') {
                Object.keys(cacheRef).forEach(k => delete (cacheRef as any)[k]);
            }
        } catch {}

        // Restore from in-memory cache (empty on page reload or Apply to Search)
        if (cacheRef.search !== undefined) setSearch(cacheRef.search);
        if (cacheRef.models !== undefined) setModels(cacheRef.models);
        if (cacheRef.rosbags !== undefined) {
            console.log(`\t\t[MCAP-DEBUG] cache restore: setting rosbags from cache (${cacheRef.rosbags.length}):`, cacheRef.rosbags);
            setRosbags(cacheRef.rosbags);
        }
        if (cacheRef.viewMode !== undefined) setViewMode(cacheRef.viewMode);
        if (cacheRef.searchResults !== undefined) setSearchResults(cacheRef.searchResults);
        if (cacheRef.marksPerTopic !== undefined) setMarksPerTopic(cacheRef.marksPerTopic as any);
        if (cacheRef.searchDone !== undefined) setSearchDone(cacheRef.searchDone);
        if (cacheRef.confirmedModels !== undefined) setConfirmedModels(cacheRef.confirmedModels);
        if (cacheRef.confirmedRosbags !== undefined) setConfirmedRosbags(cacheRef.confirmedRosbags);
        if (cacheRef.searchStatus !== undefined) setSearchStatus(cacheRef.searchStatus);

        // Restore last selected tab from sessionStorage
        try {
          const lastTab = sessionStorage.getItem('lastSearchTab');
          if (lastTab === 'images' || lastTab === 'rosbags') {
            setViewMode(lastTab);
          }
        } catch {}
        
        // Restore positional filter from sessionStorage
        const loadPositionalFilter = () => {
          console.log('\t\t[MCAP-DEBUG] loadPositionalFilter() called');
          try {
            const positionalFilterRaw = sessionStorage.getItem('__BagSeekPositionalFilter');
            if (positionalFilterRaw) {
              const filtered = JSON.parse(positionalFilterRaw);
              if (Array.isArray(filtered) && filtered.length > 0) {
                console.log(`\t\t[MCAP-DEBUG] positionallyFilteredRosbags set (${filtered.length}):`, filtered);
                setPositionallyFilteredRosbags(filtered);
              } else {
                console.log('\t\t[MCAP-DEBUG] positionallyFilteredRosbags cleared (empty array)');
                setPositionallyFilteredRosbags(null);
              }
            } else {
              console.log('\t\t[MCAP-DEBUG] positionallyFilteredRosbags cleared (no sessionStorage key)');
              setPositionallyFilteredRosbags(null);
            }
          } catch {}
          // Also load per-rosbag MCAP IDs from positional filter
          try {
            const mcapFilterRaw = sessionStorage.getItem('__BagSeekPositionalMcapFilter');
            if (mcapFilterRaw) {
              const parsed = JSON.parse(mcapFilterRaw);
              if (parsed && typeof parsed === 'object' && Object.keys(parsed).length > 0) {
                console.log(`\t\t[MCAP-DEBUG] pendingMcapIds set (${Object.keys(parsed).length} rosbags):`, Object.keys(parsed));
                for (const [k, v] of Object.entries(parsed)) {
                  console.log(`\t\t[MCAP-DEBUG]   "${k}": ${(v as string[]).length} mcap IDs -> [${(v as string[]).join(', ')}]`);
                }
                setPendingMcapIds(parsed);
              } else {
                console.log('\t\t[MCAP-DEBUG] pendingMcapIds cleared (empty object)');
                setPendingMcapIds(null);
              }
            } else {
              console.log('\t\t[MCAP-DEBUG] pendingMcapIds cleared (no sessionStorage key)');
              setPendingMcapIds(null);
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

    // When coming from MAP "Apply to Search", auto-select the filtered rosbags with full paths
    // Only clear the flag when we have a full match, so we re-run when refresh completes with updated availableRosbags
    useEffect(() => {
        try {
            if (sessionStorage.getItem('__BagSeekApplyToSearchJustNavigated') !== '1') {
                console.log('\t\t[MCAP-DEBUG] auto-select effect: __BagSeekApplyToSearchJustNavigated is NOT "1", skipping');
                return;
            }
        } catch { return; }
        console.log(`\t\t[MCAP-DEBUG] auto-select effect: positionallyFilteredRosbags=${positionallyFilteredRosbags?.length ?? 'null'}, availableRosbags=${availableRosbags.length}`);
        if (positionallyFilteredRosbags && positionallyFilteredRosbags.length > 0 && availableRosbags.length > 0) {
            const normalizedFiltered = new Set(positionallyFilteredRosbags.map(f => extractRosbagName(f)));
            const matched = availableRosbags.filter(p => normalizedFiltered.has(extractRosbagName(p)));
            console.log(`\t\t[MCAP-DEBUG] auto-select: normalizedFiltered (${normalizedFiltered.size}):`, Array.from(normalizedFiltered));
            console.log(`\t\t[MCAP-DEBUG] auto-select: matched (${matched.length}):`, matched);
            if (matched.length > 0) {
                setRosbags(matched);
                // Only clear flag when we have full match - otherwise refresh may bring more, and we need to re-run
                if (matched.length === positionallyFilteredRosbags.length) {
                    try { sessionStorage.removeItem('__BagSeekApplyToSearchJustNavigated'); } catch {}
                }
            }
        }
    }, [positionallyFilteredRosbags, availableRosbags]);
    
    // Filter out selected rosbags that don't match positional filter
    useEffect(() => {
        if (positionallyFilteredRosbags && rosbags.length > 0) {
            const normalizedFiltered = new Set(positionallyFilteredRosbags.map(f => extractRosbagName(f)));
            const filtered = rosbags.filter(rosbagPath =>
                normalizedFiltered.has(extractRosbagName(rosbagPath))
            );
            if (filtered.length !== rosbags.length) {
                setRosbags(filtered);
            }
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [positionallyFilteredRosbags]);

    // Fetch available image topics (independent of selection - uses all available models/rosbags)
    useEffect(() => {
        if (availableModels.length === 0 || availableRosbags.length === 0) {
            setAvailableImageTopics({});
            setTopicTypes({});
            return;
        }
        const params = new URLSearchParams();
        availableModels.forEach(m => params.append('models', m));
        availableRosbags.forEach(r => params.append('rosbags', extractRosbagName(r)));
        fetch(`/api/get-available-image-topics?${params.toString()}`)
            .then(res => res.json())
            .then(data => {
                setAvailableImageTopics(data.availableTopics || {});
                setTopicTypes(data.topicTypes || {});
            })
            .catch(err => console.error('Failed to fetch image topics:', err));
    }, [availableModels, availableRosbags]);

    // Helper function to start polling - called directly from search handler
    const startStatusPolling = () => {
        // Clear any existing polling
        if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
        }

        let retryCount = 0;
        pollingIntervalRef.current = setInterval(async () => {
            try {
                const response = await fetch('/api/search-status');
                const data = await response.json();
                setSearchStatus(data);
                retryCount = 0;

                // Stop polling when search is done or errored
                if (data.status === 'done' || data.status === 'error') {
                    if (pollingIntervalRef.current) {
                        clearInterval(pollingIntervalRef.current);
                        pollingIntervalRef.current = null;
                    }
                }
            } catch (err) {
                console.error('Failed to fetch search status:', err);
                retryCount++;
                if (retryCount >= 3) {
                    console.warn('Stopping search status polling after 3 failed attempts');
                    if (pollingIntervalRef.current) {
                        clearInterval(pollingIntervalRef.current);
                        pollingIntervalRef.current = null;
                    }
                }
            }
        }, 500); // Poll every 500ms for more responsive updates
    };

    // Cleanup polling on unmount
    useEffect(() => {
        return () => {
            if (pollingIntervalRef.current) {
                clearInterval(pollingIntervalRef.current);
            }
        };
    }, []);


    const handleKeyDown = async (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter' && search.trim()) {
            // Nothing selected = all selected
            const effectiveRosbagsPre = rosbags.length > 0 ? rosbags : [...rosbagsToUse];
            const effectiveModelsPre = models.length > 0 ? models : [...availableModels];
            if (effectiveRosbagsPre.length === 0 || effectiveModelsPre.length === 0) {
                return; // No rosbags/models available at all
            }
            
            // Cancel any running search (frontend fetch + backend processing)
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
            fetch('/api/cancel-search', { method: 'POST' }).catch(() => {});
            const abortController = new AbortController();
            abortControllerRef.current = abortController;

            // Clear UI before starting a new search
            setSearchResults([]);
            setMarksPerTopic({});
            setConfirmedModels(models);
            setConfirmedRosbags(rosbags);
            setSearchDone(false);
            setDisplayedResultsCount(initial_render_limit); // Reset displayed results count
            setSearchStatus({ progress: 0, status: 'running', message: 'Starting search...' });
            // Start polling immediately - don't rely on useEffect timing
            startStatusPolling();
            try {
                // Wait for enhancement if enhancePrompt is enabled
                let queryToUse = search;
                if (enhancePrompt) {
                    try {
                        const enhanceResponse = await fetch(`/api/enhance-prompt?prompt=${encodeURIComponent(search)}`, { signal: abortController.signal });
                        const enhanceData = await enhanceResponse.json();
                        if (enhanceData.enhanced) {
                            queryToUse = enhanceData.enhanced;
                            // Overwrite search text completely with enhanced version
                            setSearch(enhanceData.enhanced);
                            setEnhancedPrompt(enhanceData.enhanced);
                        }
                    } catch (error) {
                        if (error instanceof DOMException && error.name === 'AbortError') return;
                        console.error('Failed to enhance prompt:', error);
                        // Continue with original prompt if enhancement fails
                    }
                }

                // Compute effective rosbags/models based on topic filter
                let effectiveModels = effectiveModelsPre;
                let effectiveRosbags = effectiveRosbagsPre;
                if (selectedTopics.length > 0) {
                    if (topicAvailableRosbags) effectiveRosbags = effectiveRosbagsPre.filter(r => topicAvailableRosbags.has(r));
                    if (topicAvailableModels) effectiveModels = effectiveModelsPre.filter(m => topicAvailableModels.has(m));
                }
                if (effectiveRosbags.length === 0 || effectiveModels.length === 0) {
                    setSearchDone(true);
                    setSearchStatus({ progress: 0, status: 'done', message: 'No rosbags/models available for selected topics.' });
                    return;
                }

                const modelParams = effectiveModels.join(',');
                const rosbagParams = effectiveRosbags.join(',');
                const timeRangeParam = timeRange.join(',');
                const accuracyParam = sampling.toString();

                // Build MCAP filter: only include rosbags with active windows
                const mcapFilterParam: Record<string, [number, number][]> = {};
                for (const [rosbag, filter] of Object.entries(mcapFilters)) {
                  if (filter.windows.length > 0 && filter.ranges.length > 0) {
                    mcapFilterParam[rosbag] = filter.windows.map(([startIdx, endIdx]) => [
                      parseInt(filter.ranges[startIdx]?.mcapIdentifier ?? '0', 10),
                      parseInt(filter.ranges[endIdx]?.mcapIdentifier ?? '0', 10),
                    ]);
                  }
                }

                const params: Record<string, string> = {
                  query: queryToUse,
                  models: modelParams,
                  rosbags: rosbagParams,
                  timeRange: timeRangeParam,
                  accuracy: accuracyParam,
                };
                if (Object.keys(mcapFilterParam).length > 0) {
                  params.mcapFilter = JSON.stringify(mcapFilterParam);
                }
                if (selectedTopics.length > 0) {
                  params.topics = selectedTopics.join(',');
                }
                const queryParams = new URLSearchParams(params).toString();
                const response = await fetch(`/api/search?${queryParams}`, { method: 'GET', signal: abortController.signal });
                const data = await response.json();
                if (data.cancelled) return; // Search was superseded, ignore results
                setSearchResults(data.results || []);
                setMarksPerTopic(data.marksPerTopic || {});
                setSearchDone(true);
                setConfirmedModels(models);
                setConfirmedRosbags(rosbags);
            } catch (error) {
                if (error instanceof DOMException && error.name === 'AbortError') return; // Cancelled, not an error
                console.error('Search failed', error);
                // Leave UI empty on failure since we cleared before starting
                setSearchDone(true);
            }
        }
    };

    const rosbagsToUse = positionallyFilteredRosbags ? filteredAvailableRosbags : availableRosbags;

    const handleModelToggle = (name: string) => {
        if (name === "ALL") {
            setModels(models.length === availableModels.length ? [] : [...availableModels]);
        } else {
            setModels(models.includes(name) ? models.filter(m => m !== name) : [...models, name]);
        }
    };

    const handleRosbagToggle = (name: string) => {
        if (name === "ALL") {
            setRosbags(rosbags.length === rosbagsToUse.length ? [] : [...rosbagsToUse]);
        } else if (name === "currently selected") {
            const matched = rosbagsToUse.find(b => searchResults[0]?.rosbag && extractRosbagName(b) === searchResults[0].rosbag);
            setRosbags(matched ? [matched] : []);
        } else {
            setRosbags(rosbags.includes(name) ? rosbags.filter(r => r !== name) : [...rosbags, name]);
        }
    };

    const handleTopicToggle = (topic: string) => {
        if (topic === "ALL") {
            setSelectedTopics(selectedTopics.length === allTopics.length ? [] : [...allTopics]);
        } else {
            setSelectedTopics(selectedTopics.includes(topic) ? selectedTopics.filter(t => t !== topic) : [...selectedTopics, topic]);
        }
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

    const DRAWER_WIDTH = 560;
    const DRAWER_COLLAPSED = 56;

    return (
        <>
        <Box sx={{ display: 'flex', height: '100%', overflow: 'hidden', width: '100%' }}>
            {/* MCAP range logic - always mounted so effects run (fetch metadata, apply pending from map) */}
            <McapRangeFilter
                selectedRosbags={rosbags}
                mcapFilters={mcapFilters}
                onMcapFiltersChange={setMcapFilters}
                pendingMcapIds={pendingMcapIds}
                rosbagsToPreload={filteredAvailableRosbags}
                onPendingMcapIdsConsumed={() => {
                    setPendingMcapIds(null);
                    sessionStorage.removeItem('__BagSeekPositionalMcapFilter');
                }}
                logicOnly
                deferPendingUntilAfterRefresh={isRefreshingFilePaths}
            />
            {/* Collapsible filter sidebar - stays below header, part of content area */}
            <Box
                sx={{
                    width: filtersOpen ? DRAWER_WIDTH : DRAWER_COLLAPSED,
                    flexShrink: 0,
                    display: 'flex',
                    flexDirection: 'column',
                    borderRight: '1px solid rgba(255, 255, 255, 0.12)',
                    backgroundColor: 'background.paper',
                    transition: 'width 225ms cubic-bezier(0.4, 0, 0.2, 1)',
                    overflow: 'hidden',
                }}
            >
                {/* Toggle button - always visible */}
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: filtersOpen ? 'flex-end' : 'center',
                        py: 1,
                        px: filtersOpen ? 1 : 0,
                        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
                        minHeight: 48,
                        flexShrink: 0,
                    }}
                >
                    <IconButton
                        onClick={() => setFiltersOpen(!filtersOpen)}
                        sx={{ color: 'rgba(255,255,255,0.7)' }}
                        aria-label={filtersOpen ? 'Collapse filters' : 'Expand filters'}
                    >
                        {filtersOpen ? <ChevronLeftIcon /> : <FilterListIcon />}
                    </IconButton>
                </Box>

                {/* Filter content - shown when expanded */}
                {filtersOpen && (
                    <Box sx={{ overflowY: 'auto', overflowX: 'hidden', flex: 1, minHeight: 0, py: 1.5, px: 1.5, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                        {/* Positional Filter */}
                        <Button
                            variant={positionallyFilteredRosbags ? "contained" : "outlined"}
                            color={"primary"}
                            fullWidth
                            onClick={(e) => {
                                if (e.ctrlKey || e.metaKey) {
                                    setPositionallyFilteredRosbags(null);
                                    setPendingMcapIds(null);
                                    try {
                                        sessionStorage.removeItem('__BagSeekPositionalFilter');
                                        sessionStorage.removeItem('__BagSeekPositionalMcapFilter');
                                    } catch {}
                                } else {
                                    navigate('/positional-overview');
                                }
                            }}
                            sx={{ whiteSpace: 'nowrap' }}
                        >
                            <RoomIcon />
                            <Typography variant="body2" sx={{ ml: 1 }}>Positional Filter</Typography>
                            {positionallyFilteredRosbags && (
                                <Chip
                                    label={positionallyFilteredRosbags.length}
                                    size="small"
                                    color="primary"
                                    sx={{ ml: 1, height: 20, fontSize: '0.7rem', '& .MuiChip-label': { px: 0.75 } }}
                                />
                            )}
                        </Button>

                        {/* Rosbags - collapsible */}
                        <Box sx={{ border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1 }}>
                            <Box
                                sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    px: 1.5,
                                    py: 0.75,
                                    cursor: 'pointer',
                                    '&:hover': { backgroundColor: 'rgba(255,255,255,0.06)' },
                                    borderBottom: expandedRosbags ? '1px solid rgba(255,255,255,0.08)' : 'none',
                                }}
                                onClick={() => setExpandedRosbags(!expandedRosbags)}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>Rosbags</Typography>
                                    {rosbags.length > 0 && rosbags.length < rosbagsToUse.length && (
                                        <Box sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.primary.main}59`, color: 'primary.main' }}>{rosbags.length}</Box>
                                    )}
                                </Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                    <IconButton
                                        size="small"
                                        onClick={(e) => { e.stopPropagation(); handleRosbagToggle('ALL'); }}
                                        title="Toggle all"
                                        sx={{ color: rosbags.length === rosbagsToUse.length && rosbagsToUse.length > 0 ? 'primary.main' : 'rgba(255,255,255,0.4)', p: 0.25 }}
                                    >
                                        <SelectAllIcon sx={{ fontSize: 16 }} />
                                    </IconButton>
                                    <ExpandMoreIcon sx={{ color: 'rgba(255,255,255,0.6)', transform: expandedRosbags ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} />
                                </Box>
                            </Box>
                            <Collapse in={expandedRosbags}>
                                <Box sx={{ py: 0.5, px: 1 }}>
                                    {availableRosbags.map((name) => {
                                        const isPositionallyAvailable = !positionallyFilteredRosbags || positionallyFilteredRosbags.some(f => extractRosbagName(f) === extractRosbagName(name));
                                        const isTopicAvailable = !topicAvailableRosbags || topicAvailableRosbags.has(name);
                                        const isAvailable = isPositionallyAvailable && isTopicAvailable;
                                        const filter = mcapFilters[name];
                                        const mcapCount = filter?.ranges?.length ?? 0;
                                        const firstTime = mcapCount > 0 ? formatNsToTime(filter!.ranges[0]?.firstTimestampNs) : null;
                                        const lastTime = mcapCount > 1 ? formatNsToTime(filter!.ranges[mcapCount - 1]?.lastTimestampNs) : null;
                                        const timeRange = mcapCount === 0 ? null : mcapCount === 1 ? firstTime : lastTime ? `${firstTime}–${lastTime}` : null;
                                        const mcapLabel = mcapCount > 0 ? `${mcapCount} MCAP${mcapCount !== 1 ? 's' : ''}` : null;
                                        const isSelected = rosbags.includes(name);
                                        return (
                                            <Box key={name}>
                                                <Box onClick={() => isAvailable && handleRosbagToggle(name)} sx={{ display: 'flex', alignItems: 'center', py: 0, cursor: isAvailable ? 'pointer' : 'default', opacity: isAvailable ? 1 : 0.5, '&:hover': isAvailable ? { bgcolor: 'rgba(255,255,255,0.04)' } : {}, borderRadius: 0.5, px: 0.5 }}>
                                                    <Checkbox size="small" checked={isSelected} disabled={!isAvailable} sx={{ p: 0.5 }} />
                                                    <ListItemText
                                                        primary={
                                                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, minWidth: 0, fontSize: '0.75rem' }}>
                                                                <Box component="span" sx={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>{name}</Box>
                                                                {mcapCount > 0 && (
                                                                    <Box sx={{ display: 'flex', gap: 0.5, flexShrink: 0 }}>
                                                                        {mcapLabel && <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.secondary.main}59`, color: 'secondary.main' }}>{mcapLabel}</Box>}
                                                                        {timeRange && <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.warning.main}59`, color: 'warning.main' }}>{timeRange}</Box>}
                                                                    </Box>
                                                                )}
                                                            </Box>
                                                        }
                                                    />
                                                </Box>
                                                {isSelected && (
                                                    <Box onClick={(e) => e.stopPropagation()}>
                                                        <McapRangeFilterItem rosbag={name} mcapFilters={mcapFilters} onMcapFiltersChange={setMcapFilters} />
                                                    </Box>
                                                )}
                                            </Box>
                                        );
                                    })}
                                </Box>
                            </Collapse>
                        </Box>

                        {/* Topics - collapsible */}
                        <Box sx={{ border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1 }}>
                            <Box onClick={() => setExpandedTopics(!expandedTopics)} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 1.5, py: 0.75, cursor: 'pointer', '&:hover': { backgroundColor: 'rgba(255,255,255,0.06)' }, borderBottom: expandedTopics ? '1px solid rgba(255,255,255,0.08)' : 'none' }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>Topics</Typography>
                                    {selectedTopics.length > 0 && (
                                        <Box sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.primary.main}59`, color: 'primary.main' }}>{selectedTopics.length}</Box>
                                    )}
                                </Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                    <IconButton
                                        size="small"
                                        onClick={(e) => { e.stopPropagation(); handleTopicToggle('ALL'); }}
                                        title="Toggle all"
                                        sx={{ color: selectedTopics.length === allTopics.length && allTopics.length > 0 ? 'primary.main' : 'rgba(255,255,255,0.4)', p: 0.25 }}
                                    >
                                        <SelectAllIcon sx={{ fontSize: 16 }} />
                                    </IconButton>
                                    <ExpandMoreIcon sx={{ color: 'rgba(255,255,255,0.6)', transform: expandedTopics ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} />
                                </Box>
                            </Box>
                            <Collapse in={expandedTopics}>
                                <Box sx={{ py: 0.5, px: 1 }}>
                                    {allTopics.map((topic) => (
                                        <Box key={topic} onClick={() => handleTopicToggle(topic)} sx={{ display: 'flex', alignItems: 'center', py: 0, cursor: 'pointer', '&:hover': { bgcolor: 'rgba(255,255,255,0.04)' }, borderRadius: 0.5, px: 0.5 }}>
                                            <Checkbox size="small" checked={selectedTopics.includes(topic)} sx={{ p: 0.5 }} />
                                            <ListItemText primary={topic} primaryTypographyProps={{ fontSize: '0.75rem', sx: { wordBreak: 'break-all' } }} />
                                        </Box>
                                    ))}
                                </Box>
                            </Collapse>
                        </Box>

                        {/* Time Range - collapsible */}
                        <Box sx={{ border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1 }}>
                            <Box onClick={() => setExpandedTimeRange(!expandedTimeRange)} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 1.5, py: 0.75, cursor: 'pointer', '&:hover': { backgroundColor: 'rgba(255,255,255,0.06)' }, borderBottom: expandedTimeRange ? '1px solid rgba(255,255,255,0.08)' : 'none' }}>
                                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>
                                    Time Range: {valueLabelFormat(timeRange[0])} – {valueLabelFormat(timeRange[1])}
                                </Typography>
                                <ExpandMoreIcon sx={{ color: 'rgba(255,255,255,0.6)', transform: expandedTimeRange ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} />
                            </Box>
                            <Collapse in={expandedTimeRange}>
                                <Box sx={{ px: 3, py: 1.5 }} onClick={(e) => e.stopPropagation()}>
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
                                        sx={{ mt: 0.5 }}
                                    />
                                </Box>
                            </Collapse>
                        </Box>

                        {/* Sampling - collapsible */}
                        <Box sx={{ border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1 }}>
                            <Box onClick={() => setExpandedSampling(!expandedSampling)} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 1.5, py: 0.75, cursor: 'pointer', '&:hover': { backgroundColor: 'rgba(255,255,255,0.06)' }, borderBottom: expandedSampling ? '1px solid rgba(255,255,255,0.08)' : 'none' }}>
                                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>Sampling: {sampling}</Typography>
                                <ExpandMoreIcon sx={{ color: 'rgba(255,255,255,0.6)', transform: expandedSampling ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} />
                            </Box>
                            <Collapse in={expandedSampling}>
                                <Box sx={{ px: 3, py: 1.5 }} onClick={(e) => e.stopPropagation()}>
                                    <Slider
                                        value={samplingSteps.indexOf(sampling)}
                                        onChange={(_, newValue) => setSampling(samplingSteps[newValue as number])}
                                        min={0}
                                        max={samplingSteps.length - 1}
                                        step={null}
                                        marks={samplingSteps.map((val, idx) => ({ value: idx, label: [1, 10, 100, 1000].includes(val) ? `${val}` : '' }))}
                                        scale={(index) => samplingSteps[index]}
                                        valueLabelFormat={() => `${sampling}`}
                                        valueLabelDisplay="auto"
                                        sx={{ mt: 0.5, '& .MuiSlider-track': { display: 'none' }, '& .MuiSlider-mark': { backgroundColor: 'currentColor' } }}
                                    />
                                </Box>
                            </Collapse>
                        </Box>

                        {/* Models - collapsible */}
                        <Box sx={{ border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1 }}>
                            <Box onClick={() => setExpandedModels(!expandedModels)} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 1.5, py: 0.75, cursor: 'pointer', '&:hover': { backgroundColor: 'rgba(255,255,255,0.06)' }, borderBottom: expandedModels ? '1px solid rgba(255,255,255,0.08)' : 'none' }}>
                                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>Models</Typography>
                                    {models.length > 0 && models.length < availableModels.length && (
                                        <Box sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.primary.main}59`, color: 'primary.main' }}>{models.length}</Box>
                                    )}
                                </Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                    <IconButton
                                        size="small"
                                        onClick={(e) => { e.stopPropagation(); handleModelToggle('ALL'); }}
                                        title="Toggle all"
                                        sx={{ color: models.length === availableModels.length && availableModels.length > 0 ? 'primary.main' : 'rgba(255,255,255,0.4)', p: 0.25 }}
                                    >
                                        <SelectAllIcon sx={{ fontSize: 16 }} />
                                    </IconButton>
                                    <ExpandMoreIcon sx={{ color: 'rgba(255,255,255,0.6)', transform: expandedModels ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} />
                                </Box>
                            </Box>
                            <Collapse in={expandedModels}>
                                <Box sx={{ py: 0.5, px: 1 }}>
                                    {availableModels.map((name) => {
                                        const isTopicAvailable = !topicAvailableModels || topicAvailableModels.has(name);
                                        return (
                                            <Box key={name} onClick={() => isTopicAvailable && handleModelToggle(name)} sx={{ display: 'flex', alignItems: 'center', py: 0, cursor: isTopicAvailable ? 'pointer' : 'default', opacity: isTopicAvailable ? 1 : 0.5, '&:hover': isTopicAvailable ? { bgcolor: 'rgba(255,255,255,0.04)' } : {}, borderRadius: 0.5, px: 0.5 }}>
                                                <Checkbox size="small" checked={models.includes(name)} disabled={!isTopicAvailable} sx={{ p: 0.5 }} />
                                                <ListItemText primary={name} primaryTypographyProps={{ fontSize: '0.75rem' }} />
                                            </Box>
                                        );
                                    })}
                                </Box>
                            </Collapse>
                        </Box>
                    </Box>
                )}
            </Box>

            {/* Right-side content: search bar + tabs + results in a column */}
            <Box sx={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {/* Main content area */}
            <Box
                component="main"
                sx={{
                    flexShrink: 0,
                    display: 'flex',
                    flexDirection: 'column',
                    minWidth: 0,
                    pt: 2,
                    pb: 2,
                    px: 2,
                }}
            >
            <Box ref={searchIconRef}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TextField
                        sx={{ flex: 1, minWidth: 0 }}
                        label="Search"
                        variant="outlined"
                        value={search}
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
            </Box>

            {/* Scrollable results section */}
            <Box sx={{ 
                flex: 1, 
                overflowY: 'auto', 
                overflowX: 'hidden',
                display: 'flex',
                flexDirection: 'column',
            }}>
            {searchStatus.status !== 'idle' && searchResults.length === 0 && (
              <Box
                sx={{
                  flex: 1,
                  padding: 4,
                  background: '#121212',
                  borderRadius: '8px',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  flexDirection: 'column',
                  mt: 2,
                  mb: 2,
                }}
              >
                {searchStatus.status !== 'done' && (
                  <TractorLoader progress={searchStatus.progress} />
                )}
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
                    padding: 0,
                    background: '#121212',
                    borderRadius: '8px',
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(30%, 1fr))',
                    gap: 2,
                    mt: 2,
                    mb: 2,
                  }}
                >
                  {searchResults.slice(0, displayedResultsCount).map((result, index) => {
                    const hasRequiredFields = result.topic && result.timestamp && result.rosbag && result.mcap_identifier;

                    return (
                      <Box
                        key={index}
                        sx={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'flex-start',
                        }}
                      >
                        {hasRequiredFields && (
                          <ResultImageCard
                            result={result as any}
                            onOpenExplore={() => openExplorePage(result)}
                            onOpenDownload={() => {}}
                          />
                        )}
                      </Box>
                    );
                  })}
                  {displayedResultsCount < searchResults.length && (
                    <Button
                      variant="outlined"
                      onClick={() => setDisplayedResultsCount(prev => Math.min(prev + initial_render_limit, searchResults.length))}
                      sx={{
                        gridColumn: '1 / -1', // Span full width of grid
                        mt: 2,
                        py: 1.5,
                        color: 'white',
                        borderColor: 'rgba(255, 255, 255, 0.23)',
                        textTransform: 'none',
                        fontSize: '1rem',
                        '&:hover': {
                          borderColor: 'rgba(255, 255, 255, 0.5)',
                          backgroundColor: 'rgba(255, 255, 255, 0.08)',
                        },
                      }}
                    >
                      LOAD MORE
                    </Button>
                  )}
                </Box>
              )
            )}
            {viewMode === 'rosbags' && (
              <Box sx={{ mt: 2, mb: 2 }}>
                <Box>
                  {searchDone && (
                    <RosbagOverview
                      rosbags={confirmedRosbags}
                      models={confirmedModels}
                      searchDone={searchDone}
                      marksPerTopic={marksPerTopic}
                      selectedTopics={selectedTopics}
                    />
                  )}
                </Box>
              </Box>
            )}
            </Box>
            </Box>
        </Box>
        </>
    );
};

export default GlobalSearch;
