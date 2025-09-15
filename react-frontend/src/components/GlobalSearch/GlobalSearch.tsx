import { Select, MenuItem, Slider, InputLabel, FormControl, Checkbox, ListItemText, OutlinedInput, IconButton, SelectChangeEvent, Box, Typography, Popper, Paper, TextField, LinearProgress, ButtonGroup, Button, Chip, Tabs, Tab } from '@mui/material';
import React, { useState, useRef, useEffect } from 'react';
import RosbagOverview from '../RosbagOverview/RosbagOverview';
import { Center } from '@react-three/drei';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import DownloadIcon from '@mui/icons-material/Download';

const GlobalSearch: React.FC = () => {

    const [search, setSearch] = useState('');
    const [searchDone, setSearchDone] = useState(false);
    // View mode state: 'images' or 'rosbags'
    const [viewMode, setViewMode] = useState<'images' | 'rosbags'>(() => 'images');
    const [searchResults, setSearchResults] = useState<{ rank: number, rosbag: string, embedding_path: string, similarityScore: number, topic: string, timestamp: string, minuteOfDay: string, model: string }[]>([]);
    const [categorizedSearchResults, setCategorizedSearchResults] = useState<{ [model: string]: { [rosbag: string]: { [topic: string]: { marks: { value: number }[], query: string, results: { minuteOfDay: string, rank: number, similarityScore: number }[] } } } }>({});
    const [searchStatus, setSearchStatus] = useState<{progress: number, status: string, message: string}>({progress: 0, status: 'idle', message: ''});

    const searchIconRef = useRef<HTMLDivElement | null>(null);

    const [models, setModels] = useState<string[]>([]);
    const [rosbags, setRosbags] = useState<string[]>([]);
    const [confirmedModels, setConfirmedModels] = useState<string[]>([]);
    const [confirmedRosbags, setConfirmedRosbags] = useState<string[]>([]);
    const [timeRange, setTimeRange] = useState<number[]>([0, 1439]);
    const [sampling, setSampling] = useState<number>(10); // Default to 1, which is 10^0
    const samplingSteps = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
      20, 30, 40, 50, 60, 70, 80, 90, 100,
      200, 300, 400, 500, 600, 700, 800, 900, 1000
    ];

    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [availableRosbags, setAvailableRosbags] = useState<string[]>([]);

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
            setSearchDone(false);
            setSearchStatus({ progress: 0, status: 'running', message: 'Starting search...' });
            setSearchResults([]); // Clear previous results
            try {
                const modelParams = models.join(',');
                const rosbagParams = rosbags.join(',');
                const timeRangeParam = timeRange.join(',');
                const accuracyParam = sampling.toString();
                const queryParams = new URLSearchParams({
                  query: search,
                  models: modelParams,
                  rosbags: rosbagParams,
                  timeRange: timeRangeParam,
                  accuracy: accuracyParam
                }).toString();
                const response = await fetch(`/api/search-new?${queryParams}`, { method: 'GET' });
                const data = await response.json();
                setSearchResults(data.results || []);
                setCategorizedSearchResults(data.categorizedSearchResults || {});
                setSearchDone(true);
                setConfirmedModels(models);
                setConfirmedRosbags(rosbags);
            } catch (error) {
                console.error('Search failed', error);
                setSearchResults([]);
                setCategorizedSearchResults({});
                setSearchDone(true);
                setConfirmedModels(models);
                setConfirmedRosbags(rosbags);
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

        if (value.includes("ALL")) {
            newSelection =
                rosbags.length === availableRosbags.length ? [] : availableRosbags;
        } else if (value.includes("currently selected")) {
            const matched = availableRosbags.find(
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

    const openExplorePage = () => {
        
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
        <Box sx={{ width: '80%', mx: 'auto', mt: 4 }}>
            <Box sx={{ display: 'flex', gap: 1, mb: 1, width: '100%' }}>
                {/* Models */}
                <FormControl size="small" sx={{ flex: 1, minWidth: 0 }}>
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

                {/* Rosbags */}
                <FormControl size="small" sx={{ flex: 1, minWidth: 0 }}>
                    <InputLabel>Rosbags</InputLabel>
                    <Select
                        multiple
                        value={rosbags}
                        onChange={handleRosbagSelection}
                        input={<OutlinedInput label="Rosbags" />}
                        renderValue={(selected) => (selected as string[]).join(', ')}
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
                            <Checkbox checked={rosbags.length === availableRosbags.length && availableRosbags.length > 0} />
                            <ListItemText primary="SELECT ALL" />
                        </MenuItem>
                        <MenuItem value="currently selected">
                            <Checkbox checked={(() => {
                                const matched = availableRosbags.find(b => b.split('/').pop() === searchResults[0]?.rosbag);
                                return rosbags.includes(matched || '');
                            })()} />
                            <ListItemText primary="CURRENTLY SELECTED" />
                        </MenuItem>
                        {availableRosbags.map((name) => (
                            <MenuItem key={name} value={name}>
                                <Checkbox checked={rosbags.includes(name)} />
                                <ListItemText primary={name} />
                            </MenuItem>
                        ))}
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
            </Box>
            <Box ref={searchIconRef}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <TextField
                        fullWidth
                        label="Search"
                        variant="outlined"
                        value={search}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
                        onKeyDown={handleKeyDown}
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
                  <LinearProgress variant="determinate" value={searchStatus.progress * 100} />
                  {/*<LinearProgress />*/}
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
                    background: '#202020',
                    borderRadius: '8px',
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(30%, 1fr))',
                    gap: 2,
                    maxHeight: 'calc(100vh - 200px)',
                    overflowY: 'auto',
                    mt: 2,
                  }}
                >
                  {searchResults.map((result, index) => {
                    const url = result.topic && result.timestamp && result.rosbag
                      ? `http://localhost:5000/images/${result.rosbag}/${result.topic.replaceAll("/", "__")}-${result.timestamp}.webp`
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
                          <Box sx={{ position: 'relative', width: '100%' }}>
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
                                //overflow: 'hidden',
                                //textOverflow: 'ellipsis',
                              }}
                            />
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={openExplorePage}
                              sx={{
                                position: 'absolute',
                                top: '45%',
                                right: 6,
                                transform: 'translateY(-50%)',
                                bgcolor: 'rgba(120, 170, 200, 0.6)',
                                color: 'white',
                                p: 0.5, // less padding -> smaller button
                                '& svg': {
                                  fontSize: 18, // smaller arrow icon
                                },
                                '&:hover': {
                                  bgcolor: 'rgba(120, 170, 200, 0.8)',
                                },
                              }}
                            >
                              <ArrowForwardIosIcon />
                            </IconButton>
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => window.open(url, '_blank')}
                              sx={{
                                position: 'absolute',
                                top: '55%',
                                right: 6,
                                transform: 'translateY(-50%)',
                                bgcolor: 'rgba(120, 170, 200, 0.6)',
                                color: 'white',
                                p: 0.5, // less padding -> smaller button
                                '& svg': {
                                  fontSize: 18, // smaller arrow icon
                                },
                                '&:hover': {
                                  bgcolor: 'rgba(120, 170, 200, 0.8)',
                                },
                              }}
                            >
                              <DownloadIcon />
                            </IconButton>
                          </Box>
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
                      categorizedSearchResults={categorizedSearchResults}
                    />
                  )}
                </Box>
              </Box>
            )}
        </Box>
    );
};

export default GlobalSearch;