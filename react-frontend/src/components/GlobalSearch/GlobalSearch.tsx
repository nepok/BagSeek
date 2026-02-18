import { Slider, Checkbox, ListItemText, IconButton, Box, Typography, TextField, LinearProgress, Button, Chip, Tabs, Tab, FormControlLabel, Collapse, Select, MenuItem, Menu, Tooltip, Divider } from '@mui/material';
import StorageIcon from '@mui/icons-material/Storage';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import TopicIcon from '@mui/icons-material/Topic';
import ScheduleIcon from '@mui/icons-material/Schedule';
import FilterAltIcon from '@mui/icons-material/FilterAlt';
import SearchIcon from '@mui/icons-material/Search';
import HelpPopover from '../HelpPopover/HelpPopover';
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import RosbagOverview from '../RosbagOverview/RosbagOverview';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import DownloadIcon from '@mui/icons-material/Download';
import ImageSearchIcon from '@mui/icons-material/ImageSearch';
import RoomIcon from '@mui/icons-material/Room';
import FilterListIcon from '@mui/icons-material/FilterList';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SelectAllIcon from '@mui/icons-material/SelectAll';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import { extractRosbagName } from '../../utils/rosbag';
import { sortTopics } from '../../utils/topics';
import McapRangeFilter, { McapRangeFilterItem, McapFilterState, formatNsToTime } from '../McapRangeFilter/McapRangeFilter';
import { useSearchResultsCache } from './SearchCacheContext';
import { searchFilterCache, getFilter, clearFilterCache } from './searchFilterCache';
import TractorLoader from '../TractorLoader/TractorLoader';
import { useError } from '../ErrorContext/ErrorContext';

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

interface PipelineCounts {
  total: number;
  afterMcap: number;
  afterTopic: number;
  afterTime: number;
  afterSample: number;
  afterSearch: number;

}

function PipelineStage({ icon, label, count, prevCount, isLast, showBadge = true }: {
  icon: React.ReactNode; label: string; count: number; prevCount: number; isLast?: boolean; showBadge?: boolean;
}) {
  const reductionPct = prevCount > 0 ? Math.round((1 - count / prevCount) * 100) : 0;
  const isActive = count > 0;
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
      <Box sx={{
        display: 'flex', alignItems: 'center', gap: 0.75,
        px: 1.25, py: 0.6, borderRadius: 1.5,
        bgcolor: isActive ? 'rgba(144,202,249,0.08)' : 'rgba(255,255,255,0.04)',
        border: '1px solid', borderColor: isActive ? 'rgba(144,202,249,0.3)' : 'rgba(255,255,255,0.1)',
        transition: 'all 0.2s',
      }}>
        <Box sx={{ color: isActive ? 'primary.main' : 'rgba(255,255,255,0.3)', display: 'flex', fontSize: 14 }}>
          {icon}
        </Box>
        <Box>
          <Typography sx={{ color: 'rgba(255,255,255,0.45)', fontSize: '0.58rem', textTransform: 'uppercase', letterSpacing: '0.05em', lineHeight: 1, display: 'block' }}>
            {label}
          </Typography>
          <Typography sx={{ color: isActive ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.4)', fontFamily: 'monospace', fontSize: '0.78rem', fontWeight: 600, lineHeight: 1.2 }}>
            {count.toLocaleString()}
          </Typography>
        </Box>
        {showBadge && (
          <Typography sx={{
            color: reductionPct > 0 ? 'rgba(255,100,100,0.85)' : 'rgba(255,255,255,0.25)',
            fontSize: '0.58rem', fontWeight: 600,
          }}>
            -{reductionPct}%
          </Typography>
        )}
      </Box>
      {!isLast && (
        <Typography sx={{ color: 'rgba(255,255,255,0.25)', fontSize: '0.7rem', mx: 0.25 }}>→</Typography>
      )}
    </Box>
  );
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
  shard_id?: string;
  row_in_shard?: number;
};

function ResultImageCard({
  result,
  onOpenExplore,
  onSearchWithImage,
}: {
  result: SearchResultItem;
  onOpenExplore: () => void;
  onSearchWithImage: () => void;
}) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [contextMenu, setContextMenu] = useState<{ mouseX: number; mouseY: number } | null>(null);

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

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    setContextMenu({ mouseX: e.clientX, mouseY: e.clientY });
  };

  const handleDownload = () => {
    setContextMenu(null);
    if (!imageUrl) return;
    const formatMatch = imageUrl.match(/data:image\/([^;]+)/);
    const format = formatMatch ? formatMatch[1] : 'jpeg';
    const extension = format === 'png' ? 'png' : 'jpg';
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
  };

  return (
    <Box
      sx={{ position: 'relative', width: '100%', cursor: 'pointer' }}
      onContextMenu={handleContextMenu}
      onClick={(e) => {
        if (contextMenu !== null) {
          setContextMenu(null);
          e.preventDefault();
          e.stopPropagation();
          return;
        }
        onOpenExplore();
      }}
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
      {/* Right-click context menu */}
      <Menu
        open={contextMenu !== null}
        onClose={() => setContextMenu(null)}
        anchorReference="anchorPosition"
        anchorPosition={contextMenu ? { top: contextMenu.mouseY, left: contextMenu.mouseX } : undefined}
      >
        <MenuItem onClick={handleDownload} disabled={!imageUrl}>
          <DownloadIcon sx={{ fontSize: 18, mr: 1 }} />
          Download image
        </MenuItem>
        <MenuItem onClick={() => { setContextMenu(null); onSearchWithImage(); }}>
          <ImageSearchIcon sx={{ fontSize: 18, mr: 1 }} />
          Search with this image
        </MenuItem>
      </Menu>
    </Box>
  );
}

const GlobalSearch: React.FC = () => {

    const navigate = useNavigate();
    const { setError } = useError();
    const { cache: resultsCache, updateCache: updateResultsCache, clearCache: clearResultsCache } = useSearchResultsCache();
    const applyToSearch = typeof window !== 'undefined' && sessionStorage.getItem('__BagSeekApplyToSearchJustNavigated') === '1';

    const [search, setSearch] = useState(() => getFilter('search', applyToSearch));
    const [searchDone, setSearchDone] = useState(() => resultsCache.searchDone);
    const [pipelineCounts, setPipelineCounts] = useState<PipelineCounts | null>(null);
    const [viewMode, setViewMode] = useState<'images' | 'rosbags'>(() => getFilter('viewMode', applyToSearch));
    const [searchResults, setSearchResults] = useState<{ rank: number, rosbag: string, mcap_identifier: string, embedding_path: string, similarityScore: number, topic: string, timestamp: string, minuteOfDay: string, model: string }[]>(() => resultsCache.searchResults);
    const [marksPerTopic, setMarksPerTopic] = useState<{ [model: string]: { [rosbag: string]: { [topic: string]: { marks: { value: number; rank?: number }[] } } } }>(() => resultsCache.marksPerTopic as any);
    const [searchStatus, setSearchStatus] = useState<{progress: number, status: string, message: string}>(() => resultsCache.searchStatus);

    const searchIconRef = useRef<HTMLDivElement | null>(null);
    const searchInputRef = useRef<HTMLInputElement | null>(null);
    const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const abortControllerRef = useRef<AbortController | null>(null);
    const hasAttemptedRefreshForPositionalMismatchRef = useRef<string | null>(null);

    const [models, setModels] = useState<string[]>(() => getFilter('models', applyToSearch));
    const [rosbags, setRosbags] = useState<string[]>(() => getFilter('rosbags', applyToSearch));
    const DEFAULT_MODEL = 'ViT-B-16-quickgelu__openai';
    const [confirmedModels, setConfirmedModels] = useState<string[]>(() => resultsCache.confirmedModels);
    const [confirmedRosbags, setConfirmedRosbags] = useState<string[]>(() => resultsCache.confirmedRosbags);
    const [timeRange, setTimeRange] = useState<number[]>(() => getFilter('timeRange', applyToSearch));
    const [mcapFilters, setMcapFilters] = useState<McapFilterState>(() => getFilter('mcapFilters', applyToSearch));
    const [loadingMcapRosbags, setLoadingMcapRosbags] = useState<Set<string>>(new Set());
    // Pending MCAP IDs from positional filter (map view), keyed by rosbag name
    const [pendingMcapIds, setPendingMcapIds] = useState<Record<string, string[]> | null>(null);
    const [selectedTopics, setSelectedTopics] = useState<string[]>(() => getFilter('selectedTopics', applyToSearch));
    const [availableImageTopics, setAvailableImageTopics] = useState<Record<string, Record<string, string[]>>>({});
    const [topicTypes, setTopicTypes] = useState<Record<string, string>>({});
    const [sampling, setSampling] = useState<number>(() => getFilter('sampling', applyToSearch));
    const [enhancePrompt, setEnhancePrompt] = useState<boolean>(() => getFilter('enhancePrompt', applyToSearch));
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

    // Restore cached state on mount (and handle Apply to Search)
    useEffect(() => {
        const applyNow = sessionStorage.getItem('__BagSeekApplyToSearchJustNavigated') === '1';

        // If coming from "Apply to Search", clear both caches and reset local state
        try {
            if (applyNow) {
                clearResultsCache();
                clearFilterCache();
                setSearch('');
                setModels([]);
                setRosbags([]);
                setViewMode('images');
                setSearchResults([]);
                setMarksPerTopic({});
                setSearchDone(false);
                setConfirmedModels([]);
                setConfirmedRosbags([]);
                setSearchStatus({ progress: -1, status: 'idle', message: '' });
                setTimeRange([0, 1439]);
                setMcapFilters({});
                setSelectedTopics([]);
                setSampling(10);
                setEnhancePrompt(true);
            }
        } catch {}

        const hasCachedResults = !applyNow && (
          (resultsCache.searchResults?.length ?? 0) > 0
          || (resultsCache.confirmedModels?.length ?? 0) > 0
        );
        const hasCachedFilters = !applyNow && (
          (searchFilterCache.search !== undefined && searchFilterCache.search !== '')
          || (searchFilterCache.models?.length ?? 0) > 0
          || (searchFilterCache.rosbags?.length ?? 0) > 0
        );
        const hasCachedState = hasCachedResults || hasCachedFilters;

        // Restore from caches when returning from EXPLORE/MAP
        if (hasCachedState) {
            if (hasCachedResults) {
                setSearchResults(resultsCache.searchResults);
                setMarksPerTopic(resultsCache.marksPerTopic as any);
                setSearchDone(resultsCache.searchDone);
                setConfirmedModels(resultsCache.confirmedModels);
                setConfirmedRosbags(resultsCache.confirmedRosbags);
                setSearchStatus(resultsCache.searchStatus);
            }
            if (hasCachedFilters) {
                setSearch(searchFilterCache.search);
                setModels(searchFilterCache.models);
                setRosbags(searchFilterCache.rosbags);
                setViewMode(searchFilterCache.viewMode);
                setTimeRange(searchFilterCache.timeRange);
                setMcapFilters(searchFilterCache.mcapFilters);
                setSelectedTopics(searchFilterCache.selectedTopics);
                setSampling(searchFilterCache.sampling);
                setEnhancePrompt(searchFilterCache.enhancePrompt);
            }
            try {
                const lastTab = sessionStorage.getItem('lastSearchTab');
                if (lastTab === 'images' || lastTab === 'rosbags') setViewMode(lastTab);
            } catch {}
        } else if (!hasCachedState) {
            try { sessionStorage.removeItem('lastSearchTab'); } catch {}
        }

        const loadPositionalFilter = () => {
          if (hasCachedState) return;
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

    // Fetch pipeline counts whenever filters change (debounced 400ms)
    useEffect(() => {
        if (rosbags.length === 0) {
            setPipelineCounts(null);
            return;
        }
        const effectiveModels = models.length > 0 ? models : availableModels;
        if (effectiveModels.length === 0) {
            setPipelineCounts(null);
            return;
        }
        const timer = setTimeout(async () => {
            try {
                const rosbagSet = new Set(rosbags);
                const mcapFilterParam: Record<string, [number, number][]> = {};
                for (const [rosbag, filter] of Object.entries(mcapFilters)) {
                    if (rosbagSet.has(rosbag) && filter.windows.length > 0 && filter.ranges.length > 0) {
                        mcapFilterParam[rosbag] = filter.windows.map(([startIdx, endIdx]) => [
                            parseInt(filter.ranges[startIdx]?.mcapIdentifier ?? '0', 10),
                            parseInt(filter.ranges[endIdx]?.mcapIdentifier ?? '0', 10),
                        ]);
                    }
                }
                const params = new URLSearchParams({
                    models: effectiveModels.join(','),
                    rosbags: rosbags.join(','),
                    timeRange: timeRange.join(','),
                    accuracy: sampling.toString(),
                });
                if (Object.keys(mcapFilterParam).length > 0) params.set('mcapFilter', JSON.stringify(mcapFilterParam));
                if (selectedTopics.length > 0) params.set('topics', selectedTopics.join(','));
                const res = await fetch(`/api/pipeline-counts?${params}`);
                if (res.ok) setPipelineCounts(await res.json());
            } catch {
                // silently ignore — pipeline counts are non-critical
            }
        }, 400);
        return () => clearTimeout(timer);
    }, [models, rosbags, availableModels, timeRange, mcapFilters, selectedTopics, sampling]);

    // Persist results to context, filters to module cache (survives tab switch)
    useEffect(() => {
        updateResultsCache({
            searchResults, marksPerTopic: marksPerTopic as any, searchDone,
            confirmedModels, confirmedRosbags, searchStatus,
        });
    }, [searchResults, marksPerTopic, searchDone, confirmedModels, confirmedRosbags, searchStatus, updateResultsCache]);

    useEffect(() => {
        searchFilterCache.search = search;
        searchFilterCache.models = models;
        searchFilterCache.rosbags = rosbags;
        searchFilterCache.viewMode = viewMode;
        searchFilterCache.timeRange = timeRange;
        searchFilterCache.mcapFilters = mcapFilters;
        searchFilterCache.selectedTopics = selectedTopics;
        searchFilterCache.sampling = sampling;
        searchFilterCache.enhancePrompt = enhancePrompt;
    }, [search, models, rosbags, viewMode, timeRange, mcapFilters, selectedTopics, sampling, enhancePrompt]);

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
                    if (data.status === 'error' && data.message) {
                        setError(data.message);
                    }
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
                if (data.error) {
                    setError(data.error);
                    setSearchDone(true);
                    return;
                }
                if (data.warning) {
                    setError(data.warning);
                }
                setSearchResults(data.results || []);
                setMarksPerTopic(data.marksPerTopic || {});
                setSearchDone(true);
                setConfirmedModels(models);
                setConfirmedRosbags(rosbags);
            } catch (error) {
                if (error instanceof DOMException && error.name === 'AbortError') return; // Cancelled, not an error
                console.error('Search failed', error);
                setError('Search failed. Please check the backend logs.');
                setSearchDone(true);
            }
        }
    };

    const rosbagsToUse = positionallyFilteredRosbags ? filteredAvailableRosbags : availableRosbags;

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

    const handleSearchWithImage = async (result: {
        rosbag: string;
        topic: string;
        mcap_identifier?: string;
        model?: string;
        shard_id?: string;
        row_in_shard?: number;
    }) => {
        const hasDirect = result?.shard_id != null && result?.row_in_shard != null && result?.model;
        const hasManifest = result?.rosbag && result?.topic && result?.mcap_identifier;
        if (!hasDirect && !hasManifest) return;

        const effectiveRosbagsPre = rosbags.length > 0 ? rosbags : [...rosbagsToUse];
        const effectiveModelsPre = models.length > 0 ? models : [...availableModels];
        if (effectiveRosbagsPre.length === 0 || effectiveModelsPre.length === 0) return;

        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        fetch('/api/cancel-search', { method: 'POST' }).catch(() => {});
        const abortController = new AbortController();
        abortControllerRef.current = abortController;

        setSearchResults([]);
        setMarksPerTopic({});
        setConfirmedModels(models);
        setConfirmedRosbags(rosbags);
        setSearchDone(false);
        setDisplayedResultsCount(initial_render_limit);
        setSearchStatus({ progress: 0, status: 'running', message: 'Searching by image...' });
        startStatusPolling();

        try {
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
                imageRosbag: result.rosbag,
                models: effectiveModels.join(','),
                rosbags: effectiveRosbags.join(','),
                timeRange: timeRange.join(','),
                accuracy: sampling.toString(),
            };
            if (result.topic) params.imageTopic = result.topic;
            if (result.mcap_identifier) params.imageMcapIdentifier = result.mcap_identifier;
            if (result.model) params.imageModel = result.model;
            if (result.shard_id != null) params.imageShardId = String(result.shard_id);
            if (result.row_in_shard != null) params.imageRowInShard = String(result.row_in_shard);
            if (Object.keys(mcapFilterParam).length > 0) {
                params.mcapFilter = JSON.stringify(mcapFilterParam);
            }
            if (selectedTopics.length > 0) {
                params.topics = selectedTopics.join(',');
            }
            const queryParams = new URLSearchParams(params).toString();
            const response = await fetch(`/api/search-by-image?${queryParams}`, { method: 'GET', signal: abortController.signal });
            const data = await response.json();
            if (data.cancelled) return;
            if (data.error) {
                setError(data.error);
                setSearchDone(true);
                return;
            }
            if (data.warning) {
                setError(data.warning);
            }
            setSearchResults(data.results || []);
            setMarksPerTopic(data.marksPerTopic || {});
            setSearchDone(true);
        } catch (error) {
            if (error instanceof DOMException && error.name === 'AbortError') return;
            console.error('Search by image failed', error);
            setError('Search by image failed. Please check the backend logs.');
            setSearchDone(true);
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
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', width: '100%' }}>
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
                onLoadingChange={setLoadingMcapRosbags}
            />

            {/* Pipeline summary strip - full width, above filters + results */}
            {pipelineCounts && (
              <Box sx={{
                display: 'flex', alignItems: 'center', gap: 0, flexShrink: 0,
                px: 1.5, py: 0.75,
                borderBottom: '1px solid rgba(255,255,255,0.08)',
                overflowX: 'auto',
                '&::-webkit-scrollbar': { height: 3 },
              }}>
                <Typography sx={{ color: 'rgba(255,255,255,0.3)', mr: 1.25, fontWeight: 600, fontSize: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.08em', whiteSpace: 'nowrap' }}>
                  Pipeline
                </Typography>
                {[
                  { icon: <StorageIcon sx={{ fontSize: 14 }} />, label: 'Rosbags', count: pipelineCounts.total, prev: pipelineCounts.total, showBadge: false },
                  { icon: <InsertDriveFileIcon sx={{ fontSize: 14 }} />, label: 'MCAPs', count: pipelineCounts.afterMcap, prev: pipelineCounts.total, showBadge: true },
                  { icon: <TopicIcon sx={{ fontSize: 14 }} />, label: 'Topics', count: pipelineCounts.afterTopic, prev: pipelineCounts.afterMcap, showBadge: true },
                  { icon: <ScheduleIcon sx={{ fontSize: 14 }} />, label: 'Time', count: pipelineCounts.afterTime, prev: pipelineCounts.afterTopic, showBadge: true },
                  { icon: <FilterAltIcon sx={{ fontSize: 14 }} />, label: 'Sample', count: pipelineCounts.afterSample, prev: pipelineCounts.afterTime, showBadge: true },
                  ...(searchDone && searchResults.length > 0 ? [{ icon: <SearchIcon sx={{ fontSize: 14 }} />, label: 'Results', count: searchResults.length, prev: pipelineCounts.afterSample, showBadge: true }] : []),
                ].map((s, i, arr) => (
                  <PipelineStage key={s.label} icon={s.icon} label={s.label} count={s.count} prevCount={s.prev} isLast={i === arr.length - 1} showBadge={s.showBadge} />
                ))}
              </Box>
            )}

            {/* Inner row: sidebar + content */}
            <Box sx={{ flex: 1, minHeight: 0, display: 'flex', overflow: 'hidden' }}>
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
                {/* Top bar: "Filters" label + pill counts + toggle button */}
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: filtersOpen ? 'space-between' : 'center',
                        py: 1,
                        px: filtersOpen ? 1.5 : 0,
                        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
                        minHeight: 48,
                        flexShrink: 0,
                        gap: 1,
                    }}
                >
                    {filtersOpen ? (
                        <>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25, flexShrink: 0 }}>
                                <Typography variant="body1" sx={{ color: 'rgba(255,255,255,0.87)', fontWeight: 500 }}>
                                    Filters
                                </Typography>
                                <HelpPopover
                                    title="How filters work"
                                    content={
                                        <Box component="ul" sx={{ m: 0, pl: 2 }}>
                                            <Box component="li">Use <strong>Positional Filter</strong> to restrict results to parts of rosbags recorded within a geographic area you draw on the map.</Box>
                                            <Box component="li" sx={{ mb: 0.5 }}>Select which <strong>rosbags</strong>, <strong>topics</strong>, and <strong>time range</strong> to include before running a search.</Box>
                                            <Box component="li" sx={{ mb: 0.5 }}><strong>Sampling</strong> — only every <em>n</em>th frame per topic is searched, reducing compute at the cost of granularity.</Box>
                                            <Box component="li" sx={{ mb: 0.5 }}>Filters narrow the amount of data that is searched — a smaller selection means faster and more focused results.</Box>
                                            <Box component="li" sx={{ mb: 0.5 }}>The search is done by the Vision Language Model(s) you select in the searchbar.</Box>
                                        </Box>
                                    }
                                />
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end', flex: 1, minWidth: 0 }}>
                                <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.primary.main}59`, color: 'primary.main', whiteSpace: 'nowrap' }}>
                                    {rosbags.length}{availableRosbags.length > 0 ? `/${availableRosbags.length}` : ''} rosbags
                                </Box>

                                <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.primary.main}59`, color: 'primary.main', whiteSpace: 'nowrap' }}>
                                    {selectedTopics.length}{allTopics.length > 0 ? `/${allTopics.length}` : ''} topics
                                </Box>
                                <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.warning.main}59`, color: 'warning.main', whiteSpace: 'nowrap' }}>
                                    {valueLabelFormat(timeRange[0])} – {valueLabelFormat(timeRange[1])}
                                </Box>
                                <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.info.main}59`, color: 'info.main', whiteSpace: 'nowrap' }}>
                                    {sampling}
                                </Box>
                                <Tooltip title="Collapse filters" arrow>
                                  <IconButton
                                      onClick={() => setFiltersOpen(!filtersOpen)}
                                      size="small"
                                      sx={{ color: 'rgba(255,255,255,0.7)', p: 0.25 }}
                                      aria-label="Collapse filters"
                                  >
                                      <ChevronLeftIcon sx={{ fontSize: 20 }} />
                                  </IconButton>
                                </Tooltip>
                            </Box>
                        </>
                    ) : (
                        <Tooltip title="Expand filters" arrow>
                          <IconButton
                              onClick={() => setFiltersOpen(!filtersOpen)}
                              sx={{ color: 'rgba(255,255,255,0.7)' }}
                              aria-label="Expand filters"
                          >
                              <FilterListIcon />
                          </IconButton>
                        </Tooltip>
                    )}
                </Box>

                {/* Filter content - shown when expanded */}
                {filtersOpen && (
                    <Box sx={{ overflowY: 'auto', overflowX: 'hidden', flex: 1, minHeight: 0, py: 1.5, px: 1.5, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                        {/* Positional Filter + CLEAR */}
                        <Box sx={{ display: 'flex', gap: 0.5 }}>
                            <Tooltip title="Filter search by geographic area(s)" arrow>
                            <Button
                                variant={positionallyFilteredRosbags ? "contained" : "outlined"}
                                color={"primary"}
                                fullWidth
                                onClick={(e) => {
                                    if (e.ctrlKey || e.metaKey) {
                                        setPositionallyFilteredRosbags(null);
                                        setPendingMcapIds(null);
                                        setRosbags([]);
                                        try {
                                            sessionStorage.removeItem('__BagSeekPositionalFilter');
                                            sessionStorage.removeItem('__BagSeekPositionalMcapFilter');
                                            sessionStorage.removeItem('__BagSeekMapMcapFilter');
                                            window.dispatchEvent(new CustomEvent('__BagSeekPositionalFilterChanged'));
                                        } catch {}
                                    } else {
                                        navigate('/positional-overview');
                                    }
                                }}
                                sx={{ whiteSpace: 'nowrap', flex: 1 }}
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
                            </Tooltip>
                            {positionallyFilteredRosbags && (
                                <Tooltip title="Clear positional filter" arrow>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    onClick={() => {
                                        setPositionallyFilteredRosbags(null);
                                        setPendingMcapIds(null);
                                        setRosbags([]);
                                        try {
                                            sessionStorage.removeItem('__BagSeekPositionalFilter');
                                            sessionStorage.removeItem('__BagSeekPositionalMcapFilter');
                                            sessionStorage.removeItem('__BagSeekMapMcapFilter');
                                            window.dispatchEvent(new CustomEvent('__BagSeekPositionalFilterChanged'));
                                        } catch (e) {
                                            console.error('Failed to clear positional filter:', e);
                                        }
                                    }}
                                    sx={{ whiteSpace: 'nowrap', flexShrink: 0 }}
                                >
                                    CLEAR
                                </Button>
                                </Tooltip>
                            )}
                        </Box>

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
                                    {availableRosbags.length > 0 && (
                                        <Box sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.primary.main}59`, color: 'primary.main' }}>{rosbags.length}/{availableRosbags.length}</Box>
                                    )}
                                </Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                    <Tooltip title="Toggle all rosbags" arrow>
                                      <IconButton
                                          size="small"
                                          onClick={(e) => { e.stopPropagation(); handleRosbagToggle('ALL'); }}
                                          sx={{ color: rosbags.length === rosbagsToUse.length && rosbagsToUse.length > 0 ? 'primary.main' : 'rgba(255,255,255,0.4)', p: 0.25 }}
                                      >
                                          <SelectAllIcon sx={{ fontSize: 16 }} />
                                      </IconButton>
                                    </Tooltip>
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
                                                        <McapRangeFilterItem rosbag={name} mcapFilters={mcapFilters} onMcapFiltersChange={setMcapFilters} isLoading={loadingMcapRosbags.has(name)} />
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
                                    {allTopics.length > 0 && (
                                        <Box sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.primary.main}59`, color: 'primary.main' }}>{selectedTopics.length}/{allTopics.length}</Box>
                                    )}
                                </Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                    <Tooltip title="Toggle all topics" arrow>
                                      <IconButton
                                          size="small"
                                          onClick={(e) => { e.stopPropagation(); handleTopicToggle('ALL'); }}
                                          sx={{ color: selectedTopics.length === allTopics.length && allTopics.length > 0 ? 'primary.main' : 'rgba(255,255,255,0.4)', p: 0.25 }}
                                      >
                                          <SelectAllIcon sx={{ fontSize: 16 }} />
                                      </IconButton>
                                    </Tooltip>
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
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>Time Range</Typography>
                                    <Box sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.warning.main}59`, color: 'warning.main' }}>
                                        {valueLabelFormat(timeRange[0])} – {valueLabelFormat(timeRange[1])}
                                    </Box>
                                </Box>
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
                                            ...Array.from({ length: 24 }, (_, i) => ({
                                                value: i * 60,
                                                label: i % 3 === 0 ? valueLabelFormat(i * 60) : undefined,
                                            })),
                                            { value: 1439, label: '23:59' },
                                        ]}
                                        sx={{ mt: 0.5, fontSize: '0.65rem', '& .MuiSlider-valueLabel': { fontSize: '0.65rem' }, '& .MuiSlider-markLabel': { fontSize: '0.65rem' } }}
                                    />
                                </Box>
                            </Collapse>
                        </Box>

                        {/* Sampling - collapsible */}
                        <Box sx={{ border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1 }}>
                            <Box onClick={() => setExpandedSampling(!expandedSampling)} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 1.5, py: 0.75, cursor: 'pointer', '&:hover': { backgroundColor: 'rgba(255,255,255,0.06)' }, borderBottom: expandedSampling ? '1px solid rgba(255,255,255,0.08)' : 'none' }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>Sampling</Typography>
                                    <Box sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.info.main}59`, color: 'info.main' }}>
                                        {sampling}
                                    </Box>
                                </Box>
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
                                        sx={{ mt: 0.5, fontSize: '0.65rem', '& .MuiSlider-track': { display: 'none' }, '& .MuiSlider-mark': { backgroundColor: 'currentColor' }, '& .MuiSlider-valueLabel': { fontSize: '0.65rem' }, '& .MuiSlider-markLabel': { fontSize: '0.65rem' } }}
                                    />
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
            <Box
              ref={searchIconRef}
              sx={{
                display: 'flex',
                alignItems: 'center',
                border: '1px solid rgba(255,255,255,0.23)',
                borderRadius: '999px',
                px: 0.5,
                '&:focus-within': { borderColor: 'primary.main' },
              }}
            >
                {/* Model select - left side (fixed width so divider stays put) */}
                <Box sx={{ width: 180, flexShrink: 0, display: 'flex', alignItems: 'center' }}>
                  <Select
                    multiple
                    value={models}
                    onChange={(e) => {
                      const val = e.target.value;
                      setModels(typeof val === 'string' ? val.split(',') : val);
                    }}
                    displayEmpty
                    renderValue={(selected) => {
                      if (selected.length === 0) return 'All models';
                      if (selected.length === 1) return selected[0];
                      if (selected.length === availableModels.length) return 'All models';
                      return `${selected.length} models`;
                    }}
                    variant="standard"
                    disableUnderline
                    sx={{
                      width: '100%',
                      pl: 1.5,
                      color: 'rgba(255,255,255,0.7)',
                      fontSize: '0.8rem',
                      '.MuiSvgIcon-root': { color: 'rgba(255,255,255,0.4)' },
                      '&:before, &:after': { display: 'none' },
                    }}
                  >
                    {availableModels.map((name) => (
                      <MenuItem key={name} value={name} sx={{ py: 0.25 }}>
                        <Checkbox size="small" checked={models.includes(name)} sx={{ p: 0.5 }} />
                        <ListItemText primary={name} primaryTypographyProps={{ fontSize: '0.75rem' }} />
                      </MenuItem>
                    ))}
                  </Select>
                </Box>

                {/* Divider */}
                <Box sx={{ width: '1px', height: 24, bgcolor: 'rgba(255,255,255,0.15)', flexShrink: 0 }} />

                {/* Search input - center, takes remaining space */}
                <input
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value);
                    if (enhancedPrompt) setEnhancedPrompt('');
                  }}
                  onKeyDown={handleKeyDown}
                  placeholder="Search..."
                  style={{
                    flex: 1,
                    minWidth: 0,
                    border: 'none',
                    outline: 'none',
                    background: 'transparent',
                    color: 'white',
                    fontSize: '0.95rem',
                    padding: '14px 12px',
                  }}
                />

                {/* Divider */}
                <Box sx={{ width: '1px', height: 24, bgcolor: 'rgba(255,255,255,0.15)', flexShrink: 0 }} />

                {/* Enhance toggle - right side (fixed width so divider stays put) */}
                <Box sx={{ width: 130, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        size="small"
                        checked={enhancePrompt}
                        onChange={(event: React.ChangeEvent<HTMLInputElement>) => setEnhancePrompt(event.target.checked)}
                        icon={<AutoFixHighIcon sx={{ fontSize: 18, color: 'rgba(255,255,255,0.35)', transition: 'all 0.2s' }} />}
                        checkedIcon={<AutoFixHighIcon sx={{ fontSize: 24, color: 'primary.main', filter: 'drop-shadow(0 0 5px rgba(144,202,249,0.6))', transition: 'all 0.2s' }} />}
                        sx={{ width: 34, height: 34, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                      />
                    }
                    label={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
                        <span>ENHANCE</span>
                        <HelpPopover
                          title="Enhance prompt"
                          content={
                            <Box component="ul" sx={{ m: 0, pl: 2 }}>
                              <Box component="li" sx={{ mb: 0.5 }}>Your query is rewritten by a Large Language Model before being used for the search.</Box>
                              <Box component="li" sx={{ mb: 0.5 }}>Useful for short or abstract queries — e.g. <em>"tractor"</em> becomes <em>"A photo of a tractor, a type of agricultural equipment"</em>.</Box>
                              <Box component="li">Adds ~6–7 s of latency to the search.</Box>
                            </Box>
                          }
                        />
                      </Box>
                    }
                    sx={{
                      ml: 0,
                      mr: 0,
                      color: enhancePrompt ? 'primary.main' : 'rgba(255,255,255,0.4)',
                      userSelect: 'none',
                      whiteSpace: 'nowrap',
                      transition: 'color 0.2s',
                      '& .MuiFormControlLabel-label': { fontSize: '0.75rem' },
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
                  minHeight: 0,
                  padding: 4,
                  background: '#121212',
                  borderRadius: '8px',
                  display: 'flex',
                  flexDirection: 'column',
                  mt: 2,
                  mb: 2,
                }}
              >
                {/* TractorLoader fixed at 1/3 of space */}
                <Box
                  sx={{
                    flex: '0 0 50%',
                    minHeight: 120,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {searchStatus.status !== 'done' && (
                    <TractorLoader progress={searchStatus.progress} />
                  )}
                </Box>
                {/* Message fixed below */}
                {searchStatus.message && (
                  <Typography
                    variant="body1"
                    sx={{
                      flex: 1,
                      minHeight: 0,
                      overflowY: 'auto',
                      mt: 2,
                      color: 'white',
                      textAlign: 'center',
                      whiteSpace: 'pre-line',
                      fontSize: '0.875rem',
                    }}
                  >
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
                            onSearchWithImage={() => handleSearchWithImage(result)}
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
            </Box> {/* closes inner row: sidebar + content */}
        </Box>
        </>
    );
};

export default GlobalSearch;
