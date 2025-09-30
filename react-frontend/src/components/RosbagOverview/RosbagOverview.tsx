import Typography from '@mui/material/Typography/Typography';
import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { HeatBar } from '../HeatBar/HeatBar';
// Preview is rendered inline above the bar, fixed to the mark position.
import IconButton from '@mui/material/IconButton/IconButton';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import { useNavigate } from 'react-router-dom';

interface RosbagOverviewProps {
    rosbags: string[];
    models: string[];
    searchDone: boolean;
    categorizedSearchResults: {
        [model: string]: {
            [rosbag: string]: {
                [topic: string]: {
                    marks: { value: number }[];
                    results: {
                        minuteOfDay: string;
                        rank: number;
                        similarityScore: number;
                    }[];
                };
            };
        };
    };
}

const RosbagOverview: React.FC<RosbagOverviewProps> = ({ rosbags, models, categorizedSearchResults }) => {
    const PREVIEW_W = 240; // fixed preview width in px
    const PREVIEW_HALF = PREVIEW_W / 2;
    const navigate = useNavigate();
    const [topics, setTopics] = useState<{ [model: string]: { [rosbag: string]: string[] } }>({});
    const [timestampLengths, setTimestampLengths] = useState<{ [rosbag: string]: number }>({});
    const [preview, setPreview] = useState<{ url: string; fraction: number; rowKey: string; leftPx?: number } | null>(null);
    // Track per-row representative image heights so hover preview scales with them
    const repImgRefs = useRef<{ [rowKey: string]: HTMLImageElement | null }>({});
    const [repImgHeights, setRepImgHeights] = useState<{ [rowKey: string]: number }>({});
    const lastKeyRef = useRef<string>('');
    // Debounce + cancellation for hover previews
    const hoverTimerRef = useRef<number | null>(null);
    const abortRef = useRef<AbortController | null>(null);
    const latestSeqRef = useRef<number>(0);
    const latestHoverRef = useRef<{ rowKey: string; rosbagName: string; topic: string; idx: number; fraction: number; leftPx: number } | null>(null);

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
                setTopics(data);
            } catch (error) {
                console.error('Failed to fetch topics:', error);
            }
        };
        fetchTopics();

        const fetchTimestampLengths = async () => {
            try {
                const response = await axios.get('/api/get-timestamp-lengths', {
                    params: {
                        rosbags: rosbags
                    },
                    paramsSerializer: params => {
                        const searchParams = new URLSearchParams();
                        params.rosbags.forEach((r: string) => searchParams.append('rosbags', r));
                        return searchParams.toString();
                    }
                });

                const data = response.data.timestampLengths || {};
                setTimestampLengths(data);
            } catch (error) {
                console.error('Failed to fetch timestamp lengths:', error);
            }
        };
        fetchTimestampLengths();

    }, [rosbags, models]);

    return (
        <div>
            {models.map((model) => (
                <div key={model}>
                    <Typography variant="h6" sx={{ color: 'white', mb: 1 }}>
                        Model: {model}
                    </Typography>
                    {rosbags.map((rosbag) => {
                        const rosbagName = rosbag.split('/').pop() || rosbag;
                        return (
                            <div key={rosbag} style={{ paddingLeft: '1rem', marginBottom: '2rem' }}>
                                <Typography variant="subtitle1" sx={{ color: 'white', mb: 1 }}>
                                    Rosbag: {rosbagName}
                                </Typography>
                                {(topics[model]?.[rosbagName] || []).map((topic) => {
                                    const topicSafe = topic.replace(/\//g, '__');
                                    return (
                                        <div key={topic} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem', borderRadius: '6px', overflow: 'hidden' }}>
                                            <div style={{ display: 'flex', flexGrow: 1 }}>
                                                <div style={{
                                                    width: '20%',
                                                    backgroundColor: '#3a3a3a',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    padding: '1rem'
                                                }}>
                                                    <Typography variant="body2" sx={{ color: 'white', textAlign: 'center' }}>{topic}</Typography>
                                                </div>
                                                <div style={{
                                                    width: '80%',
                                                    backgroundColor: '#1e1e1e',
                                                    display: 'flex',
                                                    flexDirection: 'column',
                                                }}>
                                                    <img
                                                        src={`/representative-image/${rosbagName}/${topicSafe}_collage.webp`}
                                                        alt={`Representative for ${topic}`}
                                                        style={{ width: 'auto', maxWidth: '100%', height: 'auto' }}
                                                        ref={(el) => {
                                                            const rowKey = `${rosbagName}|${topic}`;
                                                            repImgRefs.current[rowKey] = el;
                                                            if (el) {
                                                                const h = el.clientHeight;
                                                                if (h && h !== repImgHeights[rowKey]) {
                                                                    setRepImgHeights((prev) => ({ ...prev, [rowKey]: h }));
                                                                }
                                                            }
                                                        }}
                                                        onLoad={(e) => {
                                                            const img = e.currentTarget;
                                                            const rowKey = `${rosbagName}|${topic}`;
                                                            const h = img.clientHeight;
                                                            if (h && h !== repImgHeights[rowKey]) {
                                                                setRepImgHeights((prev) => ({ ...prev, [rowKey]: h }));
                                                            }
                                                        }}
                                                    />
                                                    <img
                                                        src={`/adjacency-image/${model}/${rosbagName}/${topicSafe}/${topicSafe}.png`}
                                                        alt={`Adjacency for ${model} - ${topic}`}
                                                        style={{
                                                            width: 'auto',
                                                            maxWidth: '100%',
                                                            height: 'auto',
                                                            maxHeight: '20px' // ✅ this line reduces height
                                                        }}
                                                    />
                                                    <div style={{ position: 'relative', zIndex: 1 }}>
                                                        <HeatBar
                                                            timestampCount={timestampLengths[rosbag] || 0}
                                                            searchMarks={categorizedSearchResults[model]?.[rosbagName]?.[topic]?.marks || []}
                                                            bins={1000}
                                                            windowSize={50}
                                                            height={20}
                                                            onHover={(fraction, e) => {
                                                                const count = timestampLengths[rosbag] || 0;
                                                                if (!count) return;
                                                                const idx = Math.max(0, Math.min(count - 1, Math.round(fraction * (count - 1))));
                                                                const rowKey = `${rosbagName}|${topic}`;
                                                                const targetEl = e.currentTarget as HTMLDivElement;
                                                                const width = targetEl?.clientWidth || 0;
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
                                                                        const params = new URLSearchParams({
                                                                            rosbag: intent.rosbagName,
                                                                            topic: intent.topic,
                                                                            index: String(intent.idx)
                                                                        }).toString();
                                                                        const res = await fetch(`/api/get-topic-image-preview?${params}`, { signal: controller.signal });
                                                                        if (!res.ok) return;
                                                                        const data = await res.json();
                                                                        if (!data?.imageUrl) return;

                                                                        // Preload and only apply if still latest
                                                                        const img = new Image();
                                                                        img.onload = () => {
                                                                            if (seq !== latestSeqRef.current) return;
                                                                            setPreview({ url: data.imageUrl, fraction: intent.fraction, rowKey: intent.rowKey, leftPx: intent.leftPx });
                                                                        };
                                                                        img.src = data.imageUrl;
                                                                    } catch (err) {
                                                                        // ignore aborts/errors
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
                                                        {preview && preview.rowKey === `${rosbagName}|${topic}` && (
                                                            <div
                                                                style={{
                                                                    position: 'absolute',
                                                                    bottom: '100%',
                                                                    left: preview.leftPx !== undefined ? `${preview.leftPx}px` : `${preview.fraction * 100}%`,
                                                                    transform: 'translate(-50%, -6px)',
                                                                    background: '#202020',
                                                                    padding: 4,
                                                                    borderRadius: 4,
                                                                    pointerEvents: 'none',
                                                                    boxShadow: '0 2px 8px rgba(0,0,0,0.5)',
                                                                    zIndex: 10
                                                                }}
                                                            >
                                                                {(() => {
                                                                    const rowKey = `${rosbagName}|${topic}`;
                                                                    const h = repImgHeights[rowKey];
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
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                            <div style={{ display: 'flex', alignItems: 'center' }}>
                                                <IconButton
                                                    aria-label="open"
                                                    size="small"
                                                    sx={{ color: 'white' }}
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
                                                        const marks = (categorizedSearchResults[model]?.[rosbagName]?.[topic]?.marks || []);
                                                        const encodedMarks = encodeURIComponent(JSON.stringify(marks));
                                                        
                                                        // Try to get the first timestamp for this topic/rosbag
                                                        let firstTimestamp = null;
                                                        try {
                                                            const response = await fetch(`/api/get-first-timestamp-for-topic?rosbag=${encodeURIComponent(rosbagName)}&topic=${encodeURIComponent(topic)}`);
                                                            if (response.ok) {
                                                                const data = await response.json();
                                                                firstTimestamp = data.timestamp;
                                                            }
                                                        } catch (err) {
                                                            // Fallback: try to get first timestamp from available timestamps
                                                            try {
                                                                const timestampResponse = await fetch('/api/get-available-timestamps');
                                                                if (timestampResponse.ok) {
                                                                    const timestampData = await timestampResponse.json();
                                                                    if (timestampData.availableTimestamps && timestampData.availableTimestamps.length > 0) {
                                                                        firstTimestamp = timestampData.availableTimestamps[0];
                                                                    }
                                                                }
                                                            } catch (e) {
                                                                // ignore
                                                            }
                                                        }
                                                        
                                                        const params = new URLSearchParams();
                                                        params.set('rosbag', rosbagName);
                                                        params.set('canvas', encodedCanvas);
                                                        if (firstTimestamp) params.set('ts', String(firstTimestamp));
                                                        if (marks && marks.length > 0) params.set('marks', encodedMarks);
                                                        navigate(`/explore?${params.toString()}`);
                                                    }}
                                                >
                                                    <KeyboardArrowRightIcon />
                                                </IconButton>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        );
                    })}
                </div>
            ))}
            {/* Inline poppers are rendered per-row above */}
        </div>
    );
};

export default RosbagOverview;
