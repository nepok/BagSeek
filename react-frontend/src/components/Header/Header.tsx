import React, { useMemo, useRef, useState, useEffect } from 'react';
import { IconButton, Typography, Box, Tooltip, Popper, Paper, MenuItem, TextField, Button, Divider, CircularProgress } from '@mui/material';
import './Header.css';
import FolderIcon from '@mui/icons-material/Folder';
import IosShareIcon from '@mui/icons-material/IosShare';
import ViewQuiltRoundedIcon from '@mui/icons-material/ViewQuiltRounded';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import LogoutIcon from '@mui/icons-material/Logout';
import { useLocation, useNavigate } from 'react-router-dom';
import { extractRosbagName } from '../../utils/rosbag';
import { searchFilterCache } from '../GlobalSearch/searchFilterCache';
import { useExportPreselection } from '../Export/ExportPreselectionContext';
import { useAuth } from '../AuthContext/AuthContext';

import MapIcon from '@mui/icons-material/Map';
import SearchIcon from '@mui/icons-material/Search';
import ExploreIcon from '@mui/icons-material/Explore';


interface RosbagMeta {
  mcapCount: number;
  firstTimestampNs: string | null;
  lastTimestampNs: string | null;
}

interface HeaderProps {
  setIsFileInputVisible: (visible: boolean | ((prev: boolean) => boolean)) => void; // Controls visibility of file input dialog
  setIsExportDialogVisible: (visible: boolean | ((prev: boolean) => boolean)) => void; // Controls visibility of export dialog
  onRosbagSelect: (path: string) => Promise<void>; // Callback when user selects a rosbag in the Popper
  selectedRosbag: string | null; // Currently selected ROS bag name
  handleLoadCanvas: (name: string) => void; // Callback to load a canvas by name
  handleAddCanvas: (name: string) => void; // Callback to add a new canvas by name
  handleResetCanvas: () => void; // Callback to reset canvas to empty state
  availableTopics: Record<string, string>; // Unified: { topicName: messageType }
  canvasList: { [key: string]: { root: any, metadata: { [id: number]: any }, rosbag?: string } }; // Collection of saved canvases from App
  refreshCanvasList: () => Promise<void>; // Function to refresh canvas list from backend
  currentMetadata: { [id: number]: { nodeTopic: string | null; nodeTopicType: string | null } }; // Current canvas metadata
  selectedTimestampIndex: number | null; // Current Explore position
  searchMarks: { value: number; rank?: number }[]; // Heatmap marks from search
}

// Generates a consistent color based on rosbag name hash for UI elements
const generateColor = (rosbagName: string) => {
  const hash = rosbagName.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const colors = [
    '#ff5733', '#33ff57', '#3357ff', '#ff33a6', '#ffd633', '#33fff5', // Original colors
    '#ff9966', '#66ff99', '#6699ff', '#ff6699', '#ffee66', '#66ffee', // Softer tints
    '#cc4422', '#22cc44', '#2244cc', '#cc2266', '#ccaa22', '#22ccaa', // Darker shades
    '#ff7744', '#44ff77', '#4477ff', '#ff4477', '#ffdd44', '#44ffdd', // Intermediate tones
    '#ffb366', '#66ffb3', '#b366ff', '#ff66b3', '#ffff66', '#66ffff', // More variety
    '#aa3311', '#11aa33', '#1133aa', '#aa1133', '#aa8811', '#11aa88', // Deeper hues
  ];
  return colors[hash % colors.length];
};

// Normalize topic name: replace / with _ and remove leading _
const normalizeTopic = (topic: string): string => {
  return topic.replace(/\//g, '_').replace(/^_/, '');
};

// Extract all topics from a canvas metadata
const extractTopicsFromCanvas = (canvas: { metadata?: { [id: number]: { nodeTopic: string | null } } }): string[] => {
  const topics: string[] = [];
  if (canvas?.metadata) {
    Object.values(canvas.metadata).forEach((meta) => {
      if (meta.nodeTopic) {
        topics.push(meta.nodeTopic);
      }
    });
  }
  return topics;
};

// Check if a canvas is compatible with the currently selected rosbag based on topics
const isCanvasCompatible = (canvas: any, availableTopics: Record<string, string>): boolean => {
  const requiredTopics = extractTopicsFromCanvas(canvas);
  if (requiredTopics.length === 0) return false; // Canvas with no topics is not compatible

  // Normalize both canvas topics and available topics for comparison
  const normalizedRequiredTopics = requiredTopics.map(normalizeTopic);
  const normalizedAvailableTopics = Object.keys(availableTopics).map(normalizeTopic);

  // Check if all required topics exist in available topics
  return normalizedRequiredTopics.every(topic => normalizedAvailableTopics.includes(topic));
};

// Format nanosecond timestamp to date string only (e.g. "24.07.2025")
const formatNsToDate = (ns: string | null): string => {
  if (!ns) return '';
  try {
    const ms = Number(BigInt(ns) / BigInt(1_000_000));
    return new Date(ms).toLocaleDateString('de-DE', { year: 'numeric', month: '2-digit', day: '2-digit' });
  } catch { return ''; }
};

// Format nanosecond timestamp to time string only (e.g. "16:22:34")
const formatNsToTime = (ns: string | null): string => {
  if (!ns) return '';
  try {
    const ms = Number(BigInt(ns) / BigInt(1_000_000));
    return new Date(ms).toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit', hour12: false });
  } catch { return ''; }
};

// Header component displays app title and controls for file input, canvas management, and export
const Header: React.FC<HeaderProps> = ({ setIsFileInputVisible, setIsExportDialogVisible, onRosbagSelect, selectedRosbag, handleLoadCanvas, handleAddCanvas, handleResetCanvas, availableTopics, canvasList, refreshCanvasList, currentMetadata, selectedTimestampIndex, searchMarks }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { openExportWithPreselection } = useExportPreselection();
  const { logout, authDisabled } = useAuth();
  const viewMode: 'explore' | 'search' | 'positional' = 
    location.pathname.startsWith('/search') ? 'search' : 
    location.pathname.startsWith('/positional-overview') ? 'positional' : 
    'explore';
  // State to show/hide the canvas selection popper menu
  const [showCanvasPopper, setShowCanvasPopper] = useState(false);
  // Controls visibility of TextField for adding a new canvas
  const [showTextField, setShowTextField] = useState(false);
  // Holds the user input for new canvas name
  const [newCanvasName, setNewCanvasName] = useState('');
  // Ref for the canvas icon button to anchor the popper menu
  const canvasIconRef = useRef<HTMLButtonElement | null>(null);

  // Rosbag selector Popper state
  const [showRosbagPopper, setShowRosbagPopper] = useState(false);
  const [rosbagList, setRosbagList] = useState<string[]>([]);
  const [rosbagMeta, setRosbagMeta] = useState<Record<string, RosbagMeta>>({});
  const [loadingRosbagList, setLoadingRosbagList] = useState(false);
  const rosbagIconRef = useRef<HTMLButtonElement | null>(null);

  // Map canvasList prop to the format needed for display (with colors)
  const mappedCanvasList = useMemo(() => {
    return Object.keys(canvasList).map((name) => ({
      name,
      canvas: canvasList[name],
      rosbag: canvasList[name].rosbag || '',
      color: generateColor(canvasList[name].rosbag || ''),
    }));
  }, [canvasList]);

  // Fetch all rosbag summaries in one call and populate rosbagMeta for the given paths
  const fetchAllSummaries = async (paths: string[]) => {
    try {
      const r = await fetch('/api/get-all-summaries');
      const data = await r.json();
      const allSummaries: Record<string, any> = data.rosbags ?? {};
      const newMeta: Record<string, RosbagMeta> = {};
      for (const path of paths) {
        const name = extractRosbagName(path);
        const s = allSummaries[name];
        newMeta[path] = s
          ? { mcapCount: (s.mcapRanges ?? []).length, firstTimestampNs: s.firstTimestampNs ?? null, lastTimestampNs: s.lastTimestampNs ?? null }
          : { mcapCount: 0, firstTimestampNs: null, lastTimestampNs: null };
      }
      setRosbagMeta(newMeta);
    } catch { /* silently ignore */ }
  };

  // Stale-while-revalidate: show cached list immediately, then silently refresh against disk
  useEffect(() => {
    if (!showRosbagPopper) return;
    let cancelled = false;

    const load = async () => {
      // 1. Show cached list right away
      setLoadingRosbagList(true);
      let initialPaths: string[] = [];
      try {
        const r = await fetch('/api/get-file-paths');
        const data = await r.json();
        initialPaths = data.paths ?? [];
        if (!cancelled) {
          setRosbagList(initialPaths);
          setLoadingRosbagList(false);
        }
      } catch {
        if (!cancelled) setLoadingRosbagList(false);
      }

      // 2. Fetch all summaries in one request
      if (!cancelled) await fetchAllSummaries(initialPaths);

      // 3. Background rescan against disk — silently update list if changed
      try {
        const r = await fetch('/api/refresh-file-paths', { method: 'POST' });
        const data = await r.json();
        const freshPaths: string[] = data.paths ?? [];
        if (!cancelled) {
          const prevSorted = [...initialPaths].sort().join('\n');
          const freshSorted = [...freshPaths].sort().join('\n');
          if (prevSorted !== freshSorted) {
            setRosbagList(freshPaths);
            await fetchAllSummaries(freshPaths);
          }
        }
      } catch { /* silently ignore background refresh errors */ }
    };

    load();
    return () => { cancelled = true; };
  }, [showRosbagPopper]);

  // Toggle the visibility of the canvas selection popper
  const toggleCanvasOptions = () => {
    setShowCanvasPopper(!showCanvasPopper);
  };

  // Add a new canvas using the entered name, update list and reset input field
  const onAddCanvas = async () => {
    if (!newCanvasName.trim()) return; // Ignore empty names
    await handleAddCanvas(newCanvasName);
    // Refresh canvas list to get the full canvas data
    await refreshCanvasList();
    setNewCanvasName('');
    setShowTextField(false);
  };
  
  // Load a canvas by name via callback
  const onLoadCanvas = (name: string) => {
    handleLoadCanvas(name);
  };

  // Delete a canvas by name, update state and notify backend
  const onDeleteCanvas = async (name: string) => {
    try {
      await fetch('/api/delete-canvas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      // Refresh canvas list to get updated data
      await refreshCanvasList();
    } catch (error) {
      console.error('Error deleting canvas:', error);
    }
  };
  
  return (
    <Box className="header-container" display="flex" justifyContent="space-between" alignItems="center" padding="8px 16px">
      {/* Left side: Logo + Navigation buttons */}
      <Box display="flex" alignItems="center">
          <Typography
            variant="h4"
            className="header-title"
            component="a"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              navigate('/search');
            }}
            sx={{
              cursor: 'pointer',
              textDecoration: 'none',
              color: 'inherit',
              '&:hover': {
                opacity: 0.8,
              },
            }}
          >
            BagSeek
          </Typography>
        {/* Mode selection buttons */}
        <Box sx={{ flexGrow: 0, display: 'flex', marginLeft: 4, gap: 1 }}>
          <Tooltip title="Explore rosbag locations on a map and filter by area" arrow>
            <Button
              variant="text"
              onClick={() => {
                navigate('/positional-overview');
              }}
              sx={{
                color: 'white',
                textTransform: 'none',
                fontWeight: viewMode === 'positional' ? 700 : 400,
                opacity: viewMode === 'positional' ? 1 : 0.7,
                '&:hover': {
                  opacity: 1,
                  backgroundColor: 'transparent',
                },
              }}
            >
              <MapIcon sx={{ fontSize: 18, mr: 0.5 }} />
              MAP
            </Button>
          </Tooltip>
          <Tooltip title="Search and filter content in rosbags" arrow>
            <Button
              variant="text"
              onClick={() => {
                // Save current explore query so we can restore it later
                if (location.pathname.startsWith('/explore')) {
                  try { sessionStorage.setItem('lastExploreSearch', location.search || ''); } catch {}
                }
                navigate('/search');
              }}
              sx={{
                color: 'white',
                textTransform: 'none',
                fontWeight: viewMode === 'search' ? 700 : 400,
                opacity: viewMode === 'search' ? 1 : 0.7,
                '&:hover': {
                  opacity: 1,
                  backgroundColor: 'transparent',
                },
              }}
            >
              <SearchIcon sx={{ fontSize: 18, mr: 0.5 }} />
              SEARCH
            </Button>
          </Tooltip>
          <Tooltip title="Visualise rosbag sensor topics in resizable panels" arrow>
            <Button
              variant="text"
              onClick={() => {
                // Cache current search tab before navigating to explore
                if (location.pathname.startsWith('/search')) {
                  try {
                    if (searchFilterCache.viewMode) {
                      sessionStorage.setItem('lastSearchTab', searchFilterCache.viewMode);
                    }
                  } catch {}
                }
                let qs = '';
                try { qs = sessionStorage.getItem('lastExploreSearch') || ''; } catch {}
                navigate(`/explore${qs}`);
              }}
              sx={{
                color: 'white',
                textTransform: 'none',
                fontWeight: viewMode === 'explore' ? 700 : 400,
                opacity: viewMode === 'explore' ? 1 : 0.7,
                '&:hover': {
                  opacity: 1,
                  backgroundColor: 'transparent',
                },
              }}
            >
              <ExploreIcon sx={{ fontSize: 18, mr: 0.5 }} />
              EXPLORE
            </Button>
          </Tooltip>
        </Box>
      </Box>

      {/* Right side: Icons */}
      <Box display="flex" alignItems="center">
        {/* Popper menu anchored to canvas icon, lists canvases and add option */}
        <Popper open={showCanvasPopper} anchorEl={canvasIconRef.current} placement="bottom-end" sx={{ zIndex: 10000, width: '300px' }}>
          <Paper sx={{ padding: '8px', background: '#202020', borderRadius: '8px' }}>
            {/* RESET canvas button */}
            <Tooltip title="Clear all panels and reset the canvas layout" arrow componentsProps={{ popper: { sx: { zIndex: 10001 } } }}>
              <Button
                onClick={() => {
                  handleResetCanvas();
                  setShowCanvasPopper(false);
                }}
                variant="contained"
                color="primary"
                fullWidth
                sx={{
                  fontSize: '0.8rem',
                  marginBottom: '8px',
                  textTransform: 'none'
                }}
              >
                RESET
              </Button>
            </Tooltip>
            
            {mappedCanvasList.map((canvas, index) => {
              const isExactMatch = selectedRosbag === canvas.rosbag;
              const isCompatible = isCanvasCompatible(canvas.canvas, availableTopics);
              const canLoad = isExactMatch || isCompatible;
              
              return (
                <MenuItem 
                  key={index} 
                  style={{ 
                    fontSize: '0.8rem', 
                    padding: '0px 8px', 
                    display: 'flex', 
                    alignItems: 'center',
                    color: canLoad ? 'white' : 'gray' // Gray out if not compatible
                  }}
                  onClick={() => canLoad && onLoadCanvas(canvas.name)} // Load if compatible
                  disabled={!canLoad} // Disable if not compatible
                >
                  {/* Colored circle representing canvas */}
                  <Box sx={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: canvas.color, marginRight: '8px', marginBottom: '3.5px' }} />
                  {canvas.name}
                  {/* Delete button for canvas */}
                  <IconButton
                    onClick={(e) => { e.stopPropagation(); onDeleteCanvas(canvas.name); }}
                    sx={{
                      marginLeft: 'auto',
                      color: 'rgba(255,255,255,0.5)',
                      transition: 'color 0.15s',
                      '&:hover': { color: 'error.main' },
                    }}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </MenuItem>
              );
            })}

            {/* TextField shown when adding a new canvas */}
            {showTextField ? (
              <TextField
                autoFocus
                fullWidth
                size="small"
                variant="outlined"
                value={newCanvasName}
                onChange={(e) => setNewCanvasName(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && onAddCanvas()} // Submit on Enter key
                onBlur={() => setShowTextField(false)} // Hide TextField on blur
                sx={{
                  backgroundColor: '#303030',
                  borderRadius: '4px',
                  marginTop: '4px',
                  input: { color: 'white' },
                  '& .MuiOutlinedInput-root': {
                    '& fieldset': { borderColor: '#555' },
                    '&:hover fieldset': { borderColor: '#777' },
                    '&.Mui-focused fieldset': { borderColor: '#aaa' },
                  },
                }}
              />
            ) : (
              // Add button to show TextField for new canvas name
              <Tooltip title="Save canvas layout with a name" arrow componentsProps={{ popper: { sx: { zIndex: 10001 } } }}>
                <MenuItem onClick={() => setShowTextField(true)} sx={{ fontSize: '0.8rem', padding: '0px 8px', display: 'flex', justifyContent: 'center' }}>
                  <AddIcon fontSize="small" />
                </MenuItem>
              </Tooltip>
            )}
          </Paper>
        </Popper>

        {viewMode === 'explore' && (
          <>
            {/* Rosbag selector Popper */}
            <Popper open={showRosbagPopper} anchorEl={rosbagIconRef.current} placement="bottom-end" sx={{ zIndex: 10000, width: '630px' }}>
              <Paper sx={{ padding: '8px', background: '#202020', borderRadius: '8px', maxHeight: '50vh', overflowY: 'auto' }}>
                {loadingRosbagList ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                    <CircularProgress size={24} sx={{ color: 'white' }} />
                  </Box>
                ) : rosbagList.length === 0 ? (
                  <Typography sx={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.5)', px: 1, py: 0.5 }}>
                    No rosbags found
                  </Typography>
                ) : rosbagList.map((path) => {
                  const displayName = extractRosbagName(path);
                  const meta = rosbagMeta[path];
                  const isSelected = selectedRosbag ? extractRosbagName(path) === extractRosbagName(selectedRosbag) : false;
                  const color = generateColor(displayName);
                  const date = meta?.firstTimestampNs ? formatNsToDate(meta.firstTimestampNs) : null;
                  const firstTime = meta?.firstTimestampNs ? formatNsToTime(meta.firstTimestampNs) : null;
                  const lastTime = meta?.lastTimestampNs ? formatNsToTime(meta.lastTimestampNs) : null;
                  const timeRange = firstTime && lastTime && firstTime !== lastTime ? `${firstTime} – ${lastTime}` : firstTime;
                  return (
                    <MenuItem
                      key={path}
                      onClick={async () => {
                        setShowRosbagPopper(false);
                        await onRosbagSelect(path);
                      }}
                      sx={{
                        fontSize: '0.8rem',
                        padding: '6px 8px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        backgroundColor: isSelected ? 'rgba(255,255,255,0.08)' : 'transparent',
                        borderRadius: '4px',
                        '&:hover': { backgroundColor: 'rgba(255,255,255,0.12)' },
                      }}
                    >
                      {/* Colored dot */}
                      <Box sx={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: color, flexShrink: 0 }} />
                      {/* Rosbag name */}
                      <Typography sx={{ fontSize: '0.8rem', flexGrow: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', direction: 'rtl', textAlign: 'left' }}>
                        {displayName}
                      </Typography>
                      {/* Pills */}
                      {meta && meta.mcapCount > 0 && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexShrink: 0 }}>
                          <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.secondary.main}59`, color: 'secondary.main', whiteSpace: 'nowrap' }}>
                            {meta.mcapCount} MCAP{meta.mcapCount !== 1 ? 's' : ''}
                          </Box>
                          {date && (
                            <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.warning.main}59`, color: 'warning.main', whiteSpace: 'nowrap' }}>
                              {date}
                            </Box>
                          )}
                          {timeRange && (
                            <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', backgroundColor: (theme) => `${theme.palette.warning.main}59`, color: 'warning.main', whiteSpace: 'nowrap' }}>
                              {timeRange}
                            </Box>
                          )}
                        </Box>
                      )}
                    </MenuItem>
                  );
                })}
              </Paper>
            </Popper>

            {/* Button to open rosbag selector Popper */}
            <Tooltip title="Select Rosbag" arrow>
              <IconButton
                className="header-icon"
                ref={rosbagIconRef}
                onClick={() => {
                  setShowCanvasPopper(false);
                  setShowRosbagPopper(prev => !prev);
                }}
              >
                <FolderIcon />
              </IconButton>
            </Tooltip>

            {/* Button to toggle canvas popper menu */}
            <Tooltip title="Load/Save Canvas" arrow>
              <IconButton className="header-icon" ref={canvasIconRef} onClick={toggleCanvasOptions}>
                <ViewQuiltRoundedIcon />
              </IconButton>
            </Tooltip>
          </>
        )}

        {/* Export button - shown on all pages */}
        <Tooltip title="Open Export Menu" arrow>
          <IconButton className="header-icon" onClick={() => {
            if (viewMode === 'explore' && selectedRosbag) {
              const activeTopics = extractTopicsFromCanvas({ metadata: currentMetadata });
              openExportWithPreselection({
                rosbagPath: selectedRosbag,
                ...(activeTopics.length > 0 ? { topics: activeTopics } : {}),
                ...(selectedTimestampIndex != null ? { timestampIndex: selectedTimestampIndex } : {}),
                ...(searchMarks.length > 0 ? { searchMarks } : {}),
              });
            } else {
              setIsExportDialogVisible((prev: boolean) => !prev);
            }
          }}>
            <IosShareIcon />
          </IconButton>
        </Tooltip>
        
        <Divider orientation="vertical" sx={{margin: '8px'}} flexItem />

        {/* Smart Farming Lab logo - always visible, links to Smart Farming Lab */}
        <Tooltip title="Smart Farming Lab, University of Leipzig" arrow componentsProps={{ popper: { sx: { zIndex: 10001 } } }}>
          <Button
            component="a"
            href="https://research.uni-leipzig.de/smart-farming/#/"
            target="_blank"
            rel="noopener noreferrer"
            sx={{
              minWidth: 0,
              p: 0.5,
              marginLeft: 1,
              color: 'inherit',
              '&:hover': { backgroundColor: 'rgba(255,255,255,0.08)' },
              '& .MuiTouchRipple-root': { borderRadius: 4 },
            }}
          >
            <img
              src="/smart-farming-lab-logo.png"
              alt="Smart Farming Lab"
              style={{ height: 32, width: 'auto', display: 'block', filter: 'grayscale(100%)' }}
            />
          </Button>
        </Tooltip>

        <Divider orientation="vertical" sx={{margin: '8px'}} flexItem />

        {/* Logout button - only shown when auth is enabled */}
        {!authDisabled && (
          <Tooltip title="Logout" arrow>
            <IconButton className="header-icon" onClick={logout}>
              <LogoutIcon />
            </IconButton>
          </Tooltip>
        )}
      </Box>
    </Box>
  );
};

export default Header;
