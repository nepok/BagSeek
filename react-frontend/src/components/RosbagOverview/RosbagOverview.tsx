import Typography from '@mui/material/Typography/Typography';
import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { HeatBar } from '../HeatBar/HeatBar';
// Preview is rendered inline above the bar, fixed to the mark position.
import IconButton from '@mui/material/IconButton/IconButton';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';

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
    const [topics, setTopics] = useState<{ [model: string]: { [rosbag: string]: string[] } }>({});
    const [timestampLengths, setTimestampLengths] = useState<{ [rosbag: string]: number }>({});
    const [preview, setPreview] = useState<{ url: string; fraction: number; rowKey: string; leftPx?: number } | null>(null);
    const lastKeyRef = useRef<string>('');

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
                                                            onHover={async (fraction, e) => {
                                                                const count = timestampLengths[rosbag] || 0;
                                                                if (!count) return;
                                                                const idx = Math.max(0, Math.min(count - 1, Math.round(fraction * (count - 1))));
                                                                const cacheKey = `${rosbagName}|${topic}|${idx}`;
                                                                const rowKey = `${rosbagName}|${topic}`;
                                                                // Compute clamped left position in pixels so the image
                                                                // is fully visible and stops moving within the last PREVIEW_HALF px
                                                                const targetEl = e.currentTarget as HTMLDivElement;
                                                                const width = targetEl?.clientWidth || 0;
                                                                let leftPx = fraction * width;
                                                                leftPx = Math.max(PREVIEW_HALF, Math.min(width - PREVIEW_HALF, leftPx));
                                                                if (lastKeyRef.current === cacheKey && preview?.url && preview.rowKey === rowKey) {
                                                                    setPreview({ url: preview.url, fraction, rowKey });
                                                                    return;
                                                                }
                                                                lastKeyRef.current = cacheKey;
                                                            try {
                                                                const params = new URLSearchParams({
                                                                    rosbag: rosbagName,
                                                                    topic: topic,
                                                                    index: String(idx)
                                                                }).toString();
                                                                const res = await fetch(`/api/get-topic-image-preview?${params}`);
                                                                const data = await res.json();
                                                                    if (res.ok && data.imageUrl) {
                                                                        // Preload image to avoid showing tiny placeholder/dot before it has size
                                                                        const img = new Image();
                                                                        img.onload = () => {
                                                                            // Only set if still relevant for this row
                                                                            setPreview({ url: data.imageUrl, fraction, rowKey, leftPx });
                                                                        };
                                                                        img.src = data.imageUrl;
                                                                    }
                                                                } catch (err) {
                                                                    // ignore errors for hover
                                                                }
                                                            }}
                                                        onLeave={() => { setPreview(null); lastKeyRef.current = ''; }}
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
                                                                <img src={preview.url} alt="preview" style={{ width: PREVIEW_W, height: 'auto', maxHeight: 180, display: 'block', borderRadius: 4 }} />
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
                                                    onClick={() => console.log(`Explore → model: ${model}, rosbag: ${rosbagName}, topic: ${topic}`)}
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
