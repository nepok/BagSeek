import Typography from '@mui/material/Typography/Typography';
import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { HeatBar } from '../HeatBar/HeatBar';
// Preview is rendered inline above the bar, fixed to the mark position.
import IconButton from '@mui/material/IconButton/IconButton';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, Chip, Box, Divider, Collapse } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { sortTopics } from '../../utils/topics';
import { extractRosbagName } from '../../utils/rosbag';

interface RosbagOverviewProps {
    rosbags: string[];
    models: string[];
    searchDone: boolean;
    marksPerTopic: {
        [model: string]: {
            [rosbag: string]: {
                [topic: string]: {
                    marks: { value: number }[];
                };
            };
        };
    };
    selectedTopics?: string[];
}

const RosbagOverview: React.FC<RosbagOverviewProps> = ({ rosbags, models, marksPerTopic, selectedTopics }) => {
    const PREVIEW_W = 240; // fixed preview width in px
    const PREVIEW_HALF = PREVIEW_W / 2;
    const navigate = useNavigate();
    const [topics, setTopics] = useState<{ [model: string]: { [rosbag: string]: string[] } }>({});
    const [timestampLengths, setTimestampLengths] = useState<{ [rosbag: string]: number }>({});
    const [mcapMappings, setMcapMappings] = useState<{ [key: string]: { startIndex: number; mcap_identifier: string | null }[] }>({});
    const [preview, setPreview] = useState<{ url: string; fraction: number; rowKey: string; leftPx?: number } | null>(null);
    // Track per-row image topic preview image heights so hover preview scales with them
    const imgTopicPreviewRefs = useRef<{ [rowKey: string]: HTMLImageElement | null }>({});
    const [imgTopicPreviewHeights, setImgTopicPreviewHeights] = useState<{ [rowKey: string]: number }>({});
    const lastKeyRef = useRef<string>('');
    // Debounce + cancellation for hover previews
    const hoverTimerRef = useRef<number | null>(null);
    const abortRef = useRef<AbortController | null>(null);
    const latestSeqRef = useRef<number>(0);
    const latestHoverRef = useRef<{ rowKey: string; rosbagName: string; topic: string; idx: number; fraction: number; leftPx: number } | null>(null);
    const fetchedMappingsRef = useRef<Set<string>>(new Set());
    const [expandedTopics, setExpandedTopics] = useState<{ [key: string]: boolean }>({});
    const [expandedRosbags, setExpandedRosbags] = useState<{ [key: string]: boolean }>({});
    const [expandedModels, setExpandedModels] = useState<{ [key: string]: boolean }>({});

    useEffect(() => {
        const fetchTopics = async () => {
            try {
                const response = await axios.get('/api/get-available-image-topics', {
                    params: {
                        models: models,
                        rosbags: rosbags
                    },
                    paramsSerializer: params => {
                        const searchParams = new URLSearchParams();
                        params.models.forEach((m: string) => searchParams.append('models', m));
                        params.rosbags.forEach((r: string) => searchParams.append('rosbags', r));
                        return searchParams.toString();
                    }
                });

                const data = response.data.availableTopics || {};
                // Sort topics for each model/rosbag combination
                const sortedData: typeof data = {};
                for (const model in data) {
                    sortedData[model] = {};
                    for (const rosbag in data[model]) {
                        sortedData[model][rosbag] = sortTopics(data[model][rosbag] || []);
                    }
                }
                setTopics(sortedData);
            } catch (error) {
                console.error('Failed to fetch topics:', error);
            }
        };
        fetchTopics();
    }, [rosbags, models]);

    // Clear fetched mappings when rosbags/models change (separate effect to avoid double-fetch)
    useEffect(() => {
        fetchedMappingsRef.current.clear();
        setMcapMappings({});
    }, [rosbags, models]);

    // Fetch timestamp count per rosbag via get-timestamp-summary (one call per rosbag)
    useEffect(() => {
        if (rosbags.length === 0) return;

        const fetchTimestampLengths = async () => {
            try {
                const results = await Promise.all(
                    rosbags.map(async (rosbag) => {
                        try {
                            const response = await axios.get('/api/get-timestamp-summary', {
                                params: { rosbag },
                            });
                            return { rosbag, count: response.data?.count ?? 0 };
                        } catch (err) {
                            console.error(`Failed to fetch timestamp summary for ${rosbag}:`, err);
                            return { rosbag, count: 0 };
                        }
                    })
                );
                const lengths: { [rosbag: string]: number } = {};
                results.forEach(({ rosbag, count }) => {
                    lengths[rosbag] = count;
                });
                setTimestampLengths(lengths);
            } catch (error) {
                console.error('Failed to fetch timestamp lengths:', error);
            }
        };
        fetchTimestampLengths();
    }, [rosbags]);

    // Fetch mcap mappings for all rosbags (ranges are the same for all topics in a rosbag)
    useEffect(() => {
        // Don't fetch if we don't have the required data yet
        if (models.length === 0 || rosbags.length === 0 || Object.keys(topics).length === 0) {
            return;
        }

        const fetchMcapMappings = async () => {
            // Check which rosbags we've already fetched
            const rosbagsToFetch = rosbags.filter(rosbag => !fetchedMappingsRef.current.has(rosbag));
            
            // If we already have all mappings, skip fetching
            if (rosbagsToFetch.length === 0) {
                return;
            }
            
            const newMappings: { [key: string]: { startIndex: number; mcap_identifier: string | null }[] } = { ...mcapMappings };
            
            // Fetch ranges for each rosbag (one call per rosbag, returns ranges for all topics)
            for (const rosbag of rosbagsToFetch) {
                // Mark as fetching immediately to prevent duplicate calls
                fetchedMappingsRef.current.add(rosbag);
                
                try {
                    const response = await axios.get('/api/get-timestamp-summary', {
                        params: {
                            rosbag: rosbag
                        }
                    });
                    
                    if (response.data?.mcapRanges) {
                        // Map mcapRanges to expected format: { startIndex, mcap_identifier }
                        const ranges = response.data.mcapRanges.map((r: { startIndex: number; mcapIdentifier: string }) => ({
                            startIndex: r.startIndex,
                            mcap_identifier: r.mcapIdentifier,
                        }));
                        // Get all topics for this rosbag from all models
                        const allTopics = new Set<string>();
                        for (const model of models) {
                            const rosbagName = extractRosbagName(rosbag);
                            const topicList = topics[model]?.[rosbagName] || [];
                            topicList.forEach(topic => allTopics.add(topic));
                        }
                        
                        // Store the same ranges for all topics (they're identical)
                        Array.from(allTopics).forEach(topic => {
                            const key = `${rosbag}|${topic}`;
                            newMappings[key] = ranges;
                        });
                    } else {
                        // If fetch failed, remove from ref so it can be retried
                        fetchedMappingsRef.current.delete(rosbag);
                    }
                } catch (error) {
                    console.error(`Failed to fetch mcap mapping for ${rosbag}:`, error);
                    // Remove from ref so it can be retried
                    fetchedMappingsRef.current.delete(rosbag);
                }
            }
            
            // Only update state if we actually fetched something new
            if (rosbagsToFetch.length > 0) {
                setMcapMappings(newMappings);
            }
        };
        
        fetchMcapMappings();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [rosbags, models, topics]);

    const toggleTopic = (rowKey: string) => {
        setExpandedTopics(prev => ({ ...prev, [rowKey]: !prev[rowKey] }));
    };

    const toggleRosbag = (rosbagKey: string, topics: string[]) => {
        // Check current state - if all are expanded (or default expanded), collapse them; otherwise expand them
        const allExpanded = topics.length === 0 || topics.every(topic => {
            const rowKey = `${rosbagKey}|${topic}`;
            // Default to expanded (true) if not explicitly set to false
            return expandedTopics[rowKey] !== false;
        });
        
        const newExpandedTopics = { ...expandedTopics };
        
        // Toggle all topics for this rosbag
        topics.forEach(topic => {
            const rowKey = `${rosbagKey}|${topic}`;
            // Explicitly set to true or false (not undefined)
            newExpandedTopics[rowKey] = !allExpanded;
        });
        
        setExpandedTopics(newExpandedTopics);
    };

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, pb: 2 }}>
            {models.map((model) => {
                const modelRosbags = rosbags.filter(rosbag => {
                    const rosbagName = extractRosbagName(rosbag);
                    return topics[model]?.[rosbagName] !== undefined;
                });
                
                // Check if model is expanded (default to expanded = true)
                const isModelExpanded = expandedModels[model] !== false;
                
                // Check if all rosbags are expanded for this model (for button icon state)
                const allRosbagsExpanded = modelRosbags.length === 0 || modelRosbags.every(rosbag => {
                    const rosbagName = extractRosbagName(rosbag);
                    const rosbagKey = `${model}|${rosbagName}`;
                    // Default to expanded (true) if not explicitly set
                    return expandedRosbags[rosbagKey] !== false;
                });
                
                return (
                    <Card 
                        key={model}
                        sx={{ 
                            backgroundColor: '#252525',
                            border: '1px solid rgba(255, 255, 255, 0.15)',
                            borderRadius: 2,
                            '&:hover': {
                                borderColor: 'rgba(255, 255, 255, 0.25)',
                            }
                        }}
                    >
                        <CardHeader
                            onClick={() => {
                                if (modelRosbags.length > 0) {
                                    // Only toggle model visibility, preserve rosbag states
                                    setExpandedModels(prev => ({ ...prev, [model]: !isModelExpanded }));
                                }
                            }}
                            title={
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', flex: 1 }}>
                                    <Typography variant="h5" sx={{ color: 'white', fontWeight: 700 }}>
                                        {model}
                                    </Typography>
                                    {modelRosbags.length > 0 && (
                                        <Chip 
                                            label={`${modelRosbags.length} rosbag${modelRosbags.length !== 1 ? 's' : ''}`}
                                            size="small"
                                            sx={{
                                                backgroundColor: 'rgba(144, 202, 249, 0.16)',
                                                color: '#90caf9',
                                                fontWeight: 500,
                                                fontSize: '0.75rem',
                                            }}
                                        />
                                    )}
                                </Box>
                            }
                            action={
                                modelRosbags.length > 0 ? (
                                    <IconButton
                                        onClick={(e) => {
                                            e.stopPropagation(); // Prevent triggering the CardHeader onClick
                                            // Only toggle model visibility, preserve rosbag states
                                            setExpandedModels(prev => ({ ...prev, [model]: !isModelExpanded }));
                                        }}
                                        sx={{
                                            color: 'rgba(255, 255, 255, 0.7)',
                                            transform: isModelExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                                            transition: 'transform 0.2s',
                                        }}
                                        aria-label={isModelExpanded ? 'collapse model' : 'expand model'}
                                    >
                                        <ExpandMoreIcon />
                                    </IconButton>
                                ) : null
                            }
                            sx={{ 
                                pb: 1,
                                cursor: modelRosbags.length > 0 ? 'pointer' : 'default',
                                '&:hover': modelRosbags.length > 0 ? {
                                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                                } : {},
                                transition: 'background-color 0.2s',
                                '& .MuiCardHeader-content': {
                                    overflow: 'hidden',
                                }
                            }}
                        />
                        <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.12)' }} />
                        <CardContent sx={{ pt: 2 }}>
                            <Collapse in={isModelExpanded || modelRosbags.length === 0}>
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                    {modelRosbags.map((rosbag) => {
                                        const rosbagName = extractRosbagName(rosbag);
                                        const rosbagTopicsAll = topics[model]?.[rosbagName] || [];
                                        const rosbagTopics = selectedTopics && selectedTopics.length > 0
                                            ? rosbagTopicsAll.filter(t => selectedTopics.includes(t))
                                            : rosbagTopicsAll;
                                        const rosbagKey = `${model}|${rosbagName}`;
                                        // Default to expanded (true) if not explicitly set to false
                                        const isRosbagExpanded = expandedRosbags[rosbagKey] !== false;
                                        
                                        // Check if all topics are expanded (default to expanded)
                                        const allTopicsExpanded = rosbagTopics.length === 0 || rosbagTopics.every(topic => {
                                            const rowKey = `${rosbagName}|${topic}`;
                                            // Default to expanded (true) if not explicitly set to false
                                            return expandedTopics[rowKey] !== false;
                                        });
                                        
                                        return (
                                            <Card 
                                                key={rosbag} 
                                                sx={{ 
                                                    backgroundColor: '#1e1e1e',
                                                    border: '1px solid rgba(255, 255, 255, 0.12)',
                                                    borderRadius: 2,
                                                    '&:hover': {
                                                        borderColor: 'rgba(255, 255, 255, 0.23)',
                                                    }
                                                }}
                                            >
                                                <CardHeader
                                                    onClick={() => {
                                                        // Only toggle rosbag visibility, preserve individual topic states
                                                        setExpandedRosbags(prev => ({ ...prev, [rosbagKey]: !isRosbagExpanded }));
                                                    }}
                                                    title={
                                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', flex: 1 }}>
                                                            <Typography variant="h6" sx={{ color: 'white', fontWeight: 600 }}>
                                                                {rosbagName}
                                                            </Typography>
                                                            {rosbagTopics.length > 0 && (
                                                                <Chip 
                                                                    label={`${rosbagTopics.length} topic${rosbagTopics.length !== 1 ? 's' : ''}`}
                                                                    size="small"
                                                                    sx={{
                                                                        backgroundColor: 'rgba(255, 255, 255, 0.08)',
                                                                        color: 'rgba(255, 255, 255, 0.7)',
                                                                        fontSize: '0.7rem',
                                                                    }}
                                                                />
                                                            )}
                                                        </Box>
                                                    }
                                                    action={
                                                        <IconButton
                                                            onClick={(e) => {
                                                                e.stopPropagation(); // Prevent triggering the CardHeader onClick
                                                                // Only toggle rosbag visibility, preserve individual topic states
                                                                setExpandedRosbags(prev => ({ ...prev, [rosbagKey]: !isRosbagExpanded }));
                                                            }}
                                                            sx={{
                                                                color: 'rgba(255, 255, 255, 0.7)',
                                                                transform: isRosbagExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                                                                transition: 'transform 0.2s',
                                                            }}
                                                            aria-label={isRosbagExpanded ? 'collapse rosbag' : 'expand rosbag'}
                                                        >
                                                            <ExpandMoreIcon />
                                                        </IconButton>
                                                    }
                                                    sx={{ 
                                                        pb: 1,
                                                        cursor: 'pointer',
                                                        '&:hover': {
                                                            backgroundColor: 'rgba(255, 255, 255, 0.05)',
                                                        },
                                                        transition: 'background-color 0.2s',
                                                        '& .MuiCardHeader-content': {
                                                            overflow: 'hidden',
                                                        }
                                                    }}
                                                />
                                                <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.12)' }} />
                                                <CardContent sx={{ pt: 2 }}>
                                                    <Collapse in={isRosbagExpanded || rosbagTopics.length === 0}>
                                                        {rosbagTopics.length === 0 ? (
                                                            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.5)', fontStyle: 'italic' }}>
                                                                No topics available
                                                            </Typography>
                                                        ) : (
                                                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                                                {rosbagTopics.map((topic) => {
                                                const topicSafe = topic.replace(/\//g, '__');
                                                const rowKey = `${rosbagName}|${topic}`;
                                                // Default to expanded (true) if not explicitly set to false
                                                const isExpanded = expandedTopics[rowKey] !== false;
                                                
                                                return (
                                                    <Box 
                                                        key={topic}
                                                        sx={{
                                                            border: '1px solid rgba(255, 255, 255, 0.08)',
                                                            borderRadius: 1,
                                                            overflow: 'hidden',
                                                            backgroundColor: 'rgba(255, 255, 255, 0.03)',
                                                        }}
                                                    >
                                                        <Box
                                                            onClick={() => toggleTopic(rowKey)}
                                                            sx={{
                                                                display: 'flex',
                                                                alignItems: 'center',
                                                                justifyContent: 'space-between',
                                                                p: 1.5,
                                                                cursor: 'pointer',
                                                                '&:hover': {
                                                                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                                                                },
                                                            }}
                                                        >
                                                            <Typography 
                                                                variant="body2" 
                                                                sx={{ 
                                                                    color: 'white', 
                                                                    fontWeight: 500,
                                                                    flex: 1,
                                                                    overflow: 'hidden',
                                                                    textOverflow: 'ellipsis',
                                                                    whiteSpace: 'nowrap',
                                                                }}
                                                            >
                                                                {topic}
                                                            </Typography>
                                                            <IconButton
                                                                size="small"
                                                                sx={{
                                                                    color: 'rgba(255, 255, 255, 0.7)',
                                                                    transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                                                                    transition: 'transform 0.2s',
                                                                }}
                                                            >
                                                                <ExpandMoreIcon />
                                                            </IconButton>
                                                        </Box>
                                                        <Collapse in={isExpanded}>
                                                            <Box sx={{ p: 1.5, pt: 0 }}>
                                                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, position: 'relative' }}>
                                                    {/* Image Topic Preview */}
                                                    <img
                                                        src={`/image-topic-preview/${rosbagName}/${topicSafe}.jpg`}
                                                        alt={`Image topic preview for ${topic}`}
                                                        style={{ width: 'auto', maxWidth: '100%', height: 'auto', display: 'block' }}
                                                        ref={(el) => {
                                                            imgTopicPreviewRefs.current[rowKey] = el;
                                                            if (el) {
                                                                const h = el.clientHeight;
                                                                if (h && h !== imgTopicPreviewHeights[rowKey]) {
                                                                    setImgTopicPreviewHeights((prev) => ({ ...prev, [rowKey]: h }));
                                                                }
                                                            }
                                                        }}
                                                        onLoad={(e) => {
                                                            const img = e.currentTarget;
                                                            const h = img.clientHeight;
                                                            if (h && h !== imgTopicPreviewHeights[rowKey]) {
                                                                setImgTopicPreviewHeights((prev) => ({ ...prev, [rowKey]: h }));
                                                            }
                                                        }}
                                                    />
                                                    
                                                    {/* Adjacency Image */}
                                                    <img
                                                        src={`/adjacency-image/${model}/${rosbagName}/${topicSafe}/${topicSafe}.jpg`}
                                                        alt={`Adjacency for ${model} - ${topic}`}
                                                        style={{
                                                            width: 'auto',
                                                            maxWidth: '100%',
                                                            height: 'auto',
                                                            maxHeight: '20px',
                                                            display: 'block'
                                                        }}
                                                    />
                                                    
                                                    {/* HeatBar */}
                                                    <Box sx={{ position: 'relative', zIndex: 1 }}>
                                                        <HeatBar
                                                            timestampCount={timestampLengths[rosbag] || 0}
                                                            searchMarks={marksPerTopic[model]?.[rosbagName]?.[topic]?.marks || []}
                                                            bins={1000}
                                                            windowSize={50}
                                                            height={20}
                                                            onHover={(fraction, e) => {
                                                                const count = timestampLengths[rosbag] || 0;
                                                                if (!count) return;
                                                                const idx = Math.max(0, Math.min(count - 1, Math.round(fraction * (count - 1))));
                                                                // Get the parent container (the div with position: relative) for width calculation
                                                                const trackEl = e.currentTarget as HTMLDivElement;
                                                                const containerEl = trackEl?.parentElement as HTMLDivElement;
                                                                if (!containerEl) return;
                                                                const rect = containerEl.getBoundingClientRect();
                                                                const width = rect.width;
                                                                if (!width) return;
                                                                let leftPx = fraction * width;
                                                                leftPx = Math.max(PREVIEW_HALF, Math.min(width - PREVIEW_HALF, leftPx));

                                                                // Remember the latest hover intent
                                                                latestHoverRef.current = { rowKey, rosbagName, topic, idx, fraction, leftPx };

                                                                // Debounce: wait for 500ms of still cursor before fetching
                                                                if (hoverTimerRef.current) {
                                                                    window.clearTimeout(hoverTimerRef.current);
                                                                    hoverTimerRef.current = null;
                                                                }
                                                                hoverTimerRef.current = window.setTimeout(async () => {
                                                                    const intent = latestHoverRef.current;
                                                                    if (!intent || intent.rowKey !== rowKey || intent.idx !== idx) return;

                                                                    // Cancel any in-flight request
                                                                    if (abortRef.current) {
                                                                        abortRef.current.abort();
                                                                        abortRef.current = null;
                                                                    }
                                                                    const seq = ++latestSeqRef.current;
                                                                    const controller = new AbortController();
                                                                    abortRef.current = controller;
                                                                    try {
                                                                        // Get mcap_identifier from cached ranges
                                                                        const mappingKey = `${rosbag}|${intent.topic}`;
                                                                        const ranges = mcapMappings[mappingKey];
                                                                        if (!ranges || ranges.length === 0) return;
                                                                        
                                                                        // Find the range that contains this index
                                                                        // endIndex = next range's startIndex - 1 (or total - 1 for last range)
                                                                        let range = null;
                                                                        for (let i = 0; i < ranges.length; i++) {
                                                                            const r = ranges[i];
                                                                            const nextStart = i < ranges.length - 1 ? ranges[i + 1].startIndex : timestampLengths[rosbag] || Infinity;
                                                                            if (intent.idx >= r.startIndex && intent.idx < nextStart) {
                                                                                range = r;
                                                                                break;
                                                                            }
                                                                        }
                                                                        
                                                                        if (!range || !range.mcap_identifier) return;
                                                                        
                                                                        // Get topic timestamp for this index (lightweight call)
                                                                        const tsParams = new URLSearchParams({
                                                                            relative_rosbag_path: rosbag,
                                                                            topic: intent.topic,
                                                                            index: String(intent.idx)
                                                                        }).toString();
                                                                        const tsRes = await fetch(`/api/get-topic-timestamp-at-index?${tsParams}`, { signal: controller.signal });
                                                                        if (!tsRes.ok) return;
                                                                        const tsData = await tsRes.json();
                                                                        if (!tsData?.topicTimestamp) return;
                                                                        
                                                                        const mcapInfo = {
                                                                            mcap_identifier: range.mcap_identifier,
                                                                            topicTimestamp: tsData.topicTimestamp
                                                                        };

                                                                        // Get the image from content-mcap
                                                                        const contentParams = new URLSearchParams({
                                                                            relative_rosbag_path: rosbag,
                                                                            topic: intent.topic,
                                                                            mcap_identifier: mcapInfo.mcap_identifier,
                                                                            timestamp: String(mcapInfo.topicTimestamp)
                                                                        }).toString();
                                                                        const contentRes = await fetch(`/api/content-mcap?rosbag=${rosbag}&${contentParams}`, { signal: controller.signal });
                                                                        if (!contentRes.ok) return;
                                                                        const contentData = await contentRes.json();
                                                                        if (contentData?.type !== 'image' || !contentData?.image) return;

                                                                        // Convert base64 to data URL
                                                                        const format = contentData.format || 'jpeg';
                                                                        const imageUrl = `data:image/${format};base64,${contentData.image}`;

                                                                        // Preload and only apply if still latest
                                                                        const img = new Image();
                                                                        img.onload = () => {
                                                                            if (seq !== latestSeqRef.current) return;
                                                                            setPreview({ url: imageUrl, fraction: intent.fraction, rowKey: intent.rowKey, leftPx: intent.leftPx });
                                                                        };
                                                                        img.src = imageUrl;
                                                                    } catch (err) {
                                                                        // ignore aborts/errors (e.g. AbortError when moving quickly)
                                                                    }
                                                                }, 100) as unknown as number;
                                                            }}
                                                            onLeave={() => {
                                                                // Clear preview and cancel timers/requests
                                                                setPreview(null);
                                                                lastKeyRef.current = '';
                                                                latestHoverRef.current = null;
                                                                if (hoverTimerRef.current) {
                                                                    window.clearTimeout(hoverTimerRef.current);
                                                                    hoverTimerRef.current = null;
                                                                }
                                                                if (abortRef.current) {
                                                                    abortRef.current.abort();
                                                                    abortRef.current = null;
                                                                }
                                                            }}
                                                    />
                                                        {preview && preview.rowKey === rowKey && (
                                                            <Box
                                                                sx={{
                                                                    position: 'absolute',
                                                                    bottom: '100%',
                                                                    left: preview.leftPx !== undefined ? `${preview.leftPx}px` : `${preview.fraction * 100}%`,
                                                                    transform: 'translate(-50%, -6px)',
                                                                    background: '#202020',
                                                                    padding: 0.5,
                                                                    borderRadius: 1,
                                                                    pointerEvents: 'none',
                                                                    boxShadow: '0 2px 8px rgba(0,0,0,0.5)',
                                                                    zIndex: 10
                                                                }}
                                                            >
                                                                {(() => {
                                                                    const h = imgTopicPreviewHeights[rowKey];
                                                                    const targetH = h ? Math.max(40, Math.min(400, h)) : undefined;
                                                                    return (
                                                                        <img
                                                                            src={preview.url}
                                                                            alt="preview"
                                                                            style={{
                                                                                height: targetH ? `${targetH}px` : 'auto',
                                                                                width: 'auto',
                                                                                maxHeight: targetH ? `${targetH}px` : '180px',
                                                                                display: 'block',
                                                                                borderRadius: 4
                                                                            }}
                                                                        />
                                                                    );
                                                                })()}
                                                            </Box>
                                                        )}
                                                    </Box>
                                                    
                                                                    {/* Arrow button positioned absolutely */}
                                                                    <Box sx={{ position: 'absolute', top: 0, right: 0 }}>
                                                                        <IconButton
                                                                            aria-label="open"
                                                                            size="small"
                                                                            sx={{ 
                                                                                color: 'white',
                                                                                backgroundColor: 'rgba(255, 255, 255, 0.08)',
                                                                                '&:hover': {
                                                                                    backgroundColor: 'rgba(255, 255, 255, 0.15)',
                                                                                }
                                                                            }}
                                                                            onClick={async () => {
                                                                                // Build a single-panel canvas for this topic
                                                                                const canvas = {
                                                                                    root: { id: 1 },
                                                                                    metadata: {
                                                                                        1: { nodeTimestamp: null, nodeTopic: topic, nodeTopicType: "sensor_msgs/msg/CompressedImage" }
                                                                                    }
                                                                                } as any;
                                                                                const encodedCanvas = encodeURIComponent(JSON.stringify(canvas));
                                                                                // Use existing marks for this topic to seed the explore page heatmap
                                                                                const marks = (marksPerTopic[model]?.[rosbagName]?.[topic]?.marks || []);
                                                                                
                                                                                // Store marks in sessionStorage (not in URL for cleaner, shareable links)
                                                                                if (marks && marks.length > 0) {
                                                                                    const marksKey = `marks_${rosbagName}_${topic}`;
                                                                                    try {
                                                                                        sessionStorage.setItem(marksKey, JSON.stringify(marks));
                                                                                    } catch (e) {
                                                                                        console.warn('Failed to store marks in sessionStorage', e);
                                                                                    }
                                                                                }
                                                                                
                                                                                const params = new URLSearchParams();
                                                                                params.set('rosbag', rosbagName);
                                                                                params.set('canvas', encodedCanvas);
                                                                                params.set('ts', '0');
                                                                                // Marks are now stored in sessionStorage, not in URL
                                                                                navigate(`/explore?${params.toString()}`);
                                                                            }}
                                                                        >
                                                                            <KeyboardArrowRightIcon />
                                                                        </IconButton>
                                                                    </Box>
                                                                </Box>
                                                            </Box>
                                                        </Collapse>
                                                    </Box>
                                                );
                                            })}
                                                            </Box>
                                                        )}
                                                    </Collapse>
                                                </CardContent>
                                            </Card>
                                        );
                                    })}
                                </Box>
                            </Collapse>
                        </CardContent>
                    </Card>
                );
            })}
        </Box>
    );
};

export default RosbagOverview;
