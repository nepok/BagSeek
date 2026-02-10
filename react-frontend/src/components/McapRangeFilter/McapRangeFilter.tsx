import React, { useEffect, useRef, useState } from 'react';
import {
  Box,
  Collapse,
  IconButton,
  Slider,
  TextField,
  Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { extractRosbagName } from '../../utils/rosbag';

// Per-MCAP metadata from the backend
export interface McapRangeMeta {
  mcapIdentifier: string;
  count: number;
  startIndex: number;
  firstTimestampNs?: string;
  lastTimestampNs?: string;
}

// Per-rosbag filter state
export interface RosbagMcapFilter {
  ranges: McapRangeMeta[];
  windows: Array<[number, number]>; // [startIdx, endIdx] into ranges array
}

// Full filter state keyed by rosbag path
export type McapFilterState = Record<string, RosbagMcapFilter>;

interface McapRangeFilterProps {
  selectedRosbags: string[];
  mcapFilters: McapFilterState;
  onMcapFiltersChange: (filters: McapFilterState) => void;
  /** Per-rosbag MCAP IDs from the positional filter (map view). */
  pendingMcapIds?: Record<string, string[]> | null;
  /** Called after pending MCAP IDs have been consumed (converted to windows). */
  onPendingMcapIdsConsumed?: () => void;
}

/** Convert a set of MCAP IDs to contiguous [startIdx, endIdx] windows into a ranges array. */
function mcapIdsToWindows(
  ranges: McapRangeMeta[],
  mcapIds: string[]
): [number, number][] {
  const idSet = new Set(mcapIds);
  const indices = ranges
    .map((r, i) => (idSet.has(r.mcapIdentifier) ? i : -1))
    .filter((i) => i >= 0)
    .sort((a, b) => a - b);

  if (indices.length === 0) return [];

  const windows: [number, number][] = [];
  let start = indices[0];
  let end = indices[0];
  for (let i = 1; i < indices.length; i++) {
    if (indices[i] === end + 1) {
      end = indices[i];
    } else {
      windows.push([start, end]);
      start = indices[i];
      end = indices[i];
    }
  }
  windows.push([start, end]);
  return windows;
}

/** Convert a nanosecond timestamp string to epoch milliseconds, or null on failure. */
function nsToMs(nsStr: string | undefined): number | null {
  if (!nsStr) return null;
  try {
    return Number(BigInt(nsStr) / BigInt(1_000_000));
  } catch {
    return null;
  }
}

/** Format a nanosecond timestamp string to HH:MM time-of-day. */
function formatNsToTime(nsStr: string | undefined): string {
  const ms = nsToMs(nsStr);
  if (ms === null) return '??:??';
  const d = new Date(ms);
  return d.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
}

/** Build slider marks: tick at every minute, label (MCAP ID + time) every 5min (10min if >5h). */
function buildMarks(
  ranges: McapRangeMeta[]
): { value: number; label?: React.ReactNode }[] {
  if (ranges.length < 2) return [];

  const maxIdx = ranges.length - 1;
  const firstMs = nsToMs(ranges[0]?.firstTimestampNs);
  const lastMs = nsToMs(ranges[maxIdx]?.lastTimestampNs);
  if (firstMs === null || lastMs === null) return [];

  const durationH = (lastMs - firstMs) / 3_600_000;
  const labelIntervalMin = durationH > 5 ? 10 : 5;
  const ONE_MIN_MS = 60_000;

  const msValues = ranges.map((r) => nsToMs(r.firstTimestampNs));
  const startMinute = Math.ceil(firstMs / ONE_MIN_MS) * ONE_MIN_MS;

  const marks: { value: number; label?: React.ReactNode }[] = [];
  const usedIndices = new Set<number>();
  const labelledIndices = new Set<number>();
  let searchFrom = 0;

  const makeLabel = (idx: number, useLastTs?: boolean) => {
    const r = ranges[idx];
    if (!r) return undefined;
    const time = useLastTs
      ? formatNsToTime(r.lastTimestampNs)
      : formatNsToTime(r.firstTimestampNs);
    return (
      <span style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', lineHeight: 1.2 }}>
        <span>{r.mcapIdentifier}</span>
        <span style={{ color: 'rgba(255,255,255,0.35)' }}>{time}</span>
      </span>
    );
  };

  for (let t = startMinute; t <= lastMs; t += ONE_MIN_MS) {
    let bestIdx = searchFrom;
    let bestDist = Infinity;
    for (let i = searchFrom; i < msValues.length; i++) {
      const ms = msValues[i];
      if (ms === null) continue;
      const dist = Math.abs(ms - t);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
      if (ms > t + ONE_MIN_MS) break;
    }

    if (usedIndices.has(bestIdx)) continue;
    usedIndices.add(bestIdx);
    searchFrom = Math.max(0, bestIdx - 1);

    const minuteOfDay = Math.round(t / ONE_MIN_MS);
    const isLabelled = minuteOfDay % labelIntervalMin === 0;

    if (isLabelled) {
      labelledIndices.add(bestIdx);
      marks.push({ value: bestIdx, label: makeLabel(bestIdx) });
    } else {
      marks.push({ value: bestIdx });
    }
  }

  // Always include first and last with labels
  if (!usedIndices.has(0)) marks.unshift({ value: 0 });
  if (!labelledIndices.has(0)) {
    const existing = marks.find((m) => m.value === 0);
    if (existing) existing.label = makeLabel(0);
    else marks.unshift({ value: 0, label: makeLabel(0) });
  }
  if (!usedIndices.has(maxIdx)) marks.push({ value: maxIdx });
  if (!labelledIndices.has(maxIdx)) {
    const existing = marks.find((m) => m.value === maxIdx);
    if (existing) existing.label = makeLabel(maxIdx, true);
    else marks.push({ value: maxIdx, label: makeLabel(maxIdx, true) });
  }

  return marks;
}

const McapRangeFilter: React.FC<McapRangeFilterProps> = ({
  selectedRosbags,
  mcapFilters,
  onMcapFiltersChange,
  pendingMcapIds,
  onPendingMcapIdsConsumed,
}) => {
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState<Set<string>>(new Set());
  // Cache fetched MCAP metadata so we don't refetch on every render
  const metaCache = useRef<Record<string, McapRangeMeta[]>>({});
  // Ref to latest mcapFilters to avoid stale closures in async callbacks
  const filtersRef = useRef(mcapFilters);
  filtersRef.current = mcapFilters;

  // Fetch MCAP metadata for selected rosbags that aren't cached yet
  useEffect(() => {
    const toFetch = selectedRosbags.filter(
      (r) => !metaCache.current[r] && !loading.has(r)
    );
    if (toFetch.length === 0) return;

    setLoading((prev) => {
      const next = new Set(prev);
      toFetch.forEach((r) => next.add(r));
      return next;
    });

    toFetch.forEach(async (rosbag) => {
      try {
        const res = await fetch(
          `/api/get-timestamp-summary?rosbag=${encodeURIComponent(rosbag)}`
        );
        const data = await res.json();
        const ranges: McapRangeMeta[] = (data.mcapRanges || []).map(
          (r: any) => ({
            mcapIdentifier: r.mcapIdentifier,
            count: r.count,
            startIndex: r.startIndex,
            firstTimestampNs: r.firstTimestampNs,
            lastTimestampNs: r.lastTimestampNs,
          })
        );
        metaCache.current[rosbag] = ranges;

        // Initialize filter state for this rosbag if not present (no windows = full range)
        if (!filtersRef.current[rosbag]) {
          onMcapFiltersChange({ ...filtersRef.current, [rosbag]: { ranges, windows: [] } });
        }
      } catch (e) {
        console.error(`Failed to fetch MCAP metadata for ${rosbag}:`, e);
      } finally {
        setLoading((prev) => {
          const next = new Set(prev);
          next.delete(rosbag);
          return next;
        });
      }
    });
  }, [selectedRosbags]); // eslint-disable-line react-hooks/exhaustive-deps

  // Clean up filters for deselected rosbags
  useEffect(() => {
    const rosbagSet = new Set(selectedRosbags);
    const toRemove = Object.keys(mcapFilters).filter((r) => !rosbagSet.has(r));
    if (toRemove.length > 0) {
      const next = { ...mcapFilters };
      toRemove.forEach((r) => delete next[r]);
      onMcapFiltersChange(next);
    }
  }, [selectedRosbags]); // eslint-disable-line react-hooks/exhaustive-deps

  // Apply pending MCAP IDs from the positional filter once ranges are loaded
  useEffect(() => {
    if (!pendingMcapIds) return;

    // Check if all rosbags in pendingMcapIds have their ranges loaded
    const allLoaded = Object.keys(pendingMcapIds).every(
      (rosbag) => mcapFilters[rosbag]?.ranges?.length
    );
    if (!allLoaded) return;

    const next = { ...mcapFilters };
    let applied = false;
    for (const [rosbag, mcapIds] of Object.entries(pendingMcapIds)) {
      const filter = next[rosbag];
      if (!filter || filter.ranges.length === 0) continue;
      const windows = mcapIdsToWindows(filter.ranges, mcapIds);
      if (windows.length > 0) {
        next[rosbag] = { ...filter, windows };
        applied = true;
      }
    }
    if (applied) {
      onMcapFiltersChange(next);
      setExpanded(true);
    }
    onPendingMcapIdsConsumed?.();
  }, [pendingMcapIds, mcapFilters]); // eslint-disable-line react-hooks/exhaustive-deps

  const addWindow = (rosbag: string) => {
    const filter = mcapFilters[rosbag];
    if (!filter) return;
    const maxIdx = Math.max(0, filter.ranges.length - 1);
    const next = {
      ...mcapFilters,
      [rosbag]: {
        ...filter,
        windows: [...filter.windows, [0, maxIdx] as [number, number]],
      },
    };
    onMcapFiltersChange(next);
    if (!expanded) setExpanded(true);
  };

  const removeWindow = (rosbag: string, windowIdx: number) => {
    const filter = mcapFilters[rosbag];
    if (!filter) return;
    const next = {
      ...mcapFilters,
      [rosbag]: {
        ...filter,
        windows: filter.windows.filter((_, i) => i !== windowIdx),
      },
    };
    onMcapFiltersChange(next);
  };

  const updateWindow = (
    rosbag: string,
    windowIdx: number,
    value: [number, number]
  ) => {
    const filter = mcapFilters[rosbag];
    if (!filter) return;
    const next = {
      ...mcapFilters,
      [rosbag]: {
        ...filter,
        windows: filter.windows.map((w, i) => (i === windowIdx ? value : w)),
      },
    };
    onMcapFiltersChange(next);
  };

  // Count total active windows across all rosbags
  const totalWindows = Object.values(mcapFilters).reduce(
    (sum, f) => sum + f.windows.length,
    0
  );

  // Only show rosbags that have MCAP metadata loaded and more than 1 MCAP
  const rosbagEntries = selectedRosbags
    .map((r) => ({ rosbag: r, filter: mcapFilters[r] }))
    .filter((e) => e.filter && e.filter.ranges.length > 1);

  if (rosbagEntries.length === 0) return null;

  return (
    <Box
      sx={{
        border: '1px solid rgba(255, 255, 255, 0.12)',
        borderRadius: 1,
        mb: 1,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        onClick={() => setExpanded(!expanded)}
        sx={{
          display: 'flex',
          alignItems: 'center',
          px: 1.5,
          py: 0.5,
          cursor: 'pointer',
          '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.04)' },
        }}
      >
        <ExpandMoreIcon
          sx={{
            transform: expanded ? 'rotate(0deg)' : 'rotate(-90deg)',
            transition: 'transform 0.2s',
            fontSize: '1.2rem',
            mr: 0.5,
            color: 'rgba(255,255,255,0.5)',
          }}
        />
        <Typography
          variant="caption"
          sx={{ color: 'rgba(255,255,255,0.6)', flexGrow: 1, userSelect: 'none' }}
        >
          MCAP Ranges
          {totalWindows > 0 && (
            <span style={{ color: '#B49FCC', marginLeft: 6 }}>
              ({totalWindows} active)
            </span>
          )}
        </Typography>
      </Box>

      {/* Body */}
      <Collapse in={expanded}>
        <Box sx={{ px: 1.5, pb: 1.5 }}>
          {rosbagEntries.map(({ rosbag, filter }) => {
            const { ranges, windows } = filter!;
            const maxIdx = ranges.length - 1;
            const displayName = extractRosbagName(rosbag);
            const globalFirstTime = formatNsToTime(ranges[0]?.firstTimestampNs);
            const globalLastTime = formatNsToTime(
              ranges[maxIdx]?.lastTimestampNs
            );

            const marks = buildMarks(ranges);

            return (
              <Box key={rosbag} sx={{ mt: 1 }}>
                {/* Rosbag header */}
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    mb: 0.5,
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      color: 'rgba(255,255,255,0.8)',
                      fontWeight: 500,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      flexGrow: 1,
                    }}
                  >
                    {displayName}
                    <span
                      style={{
                        color: 'rgba(255,255,255,0.4)',
                        fontWeight: 400,
                        marginLeft: 6,
                      }}
                    >
                      {ranges.length} MCAPs &middot; {globalFirstTime} &ndash;{' '}
                      {globalLastTime}
                    </span>
                  </Typography>
                  <IconButton
                    size="small"
                    onClick={() => addWindow(rosbag)}
                    title="Add MCAP range"
                    sx={{ color: 'rgba(255,255,255,0.5)', ml: 1 }}
                  >
                    <AddIcon fontSize="small" />
                  </IconButton>
                </Box>

                {/* Windows */}
                {windows.length === 0 && (
                  <Typography
                    variant="caption"
                    sx={{ color: 'rgba(255,255,255,0.3)', ml: 1 }}
                  >
                    Full range (no filter)
                  </Typography>
                )}
                {windows.map((window, wIdx) => {
                  const [wStart, wEnd] = window;
                  const startMcap = ranges[wStart];
                  const endMcap = ranges[wEnd];

                  return (
                    <Box
                      key={wIdx}
                      sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}
                    >
                      {/* Range slider */}
                      <Box sx={{ flexGrow: 1, px: 1 }}>
                        <Slider
                          value={[wStart, wEnd]}
                          onChange={(_, newValue) =>
                            updateWindow(
                              rosbag,
                              wIdx,
                              newValue as [number, number]
                            )
                          }
                          min={0}
                          max={maxIdx}
                          valueLabelDisplay="auto"
                          valueLabelFormat={(idx) => {
                            const r = ranges[idx];
                            if (!r) return String(idx);
                            return `MCAP ${r.mcapIdentifier} (${formatNsToTime(r.firstTimestampNs)})`;
                          }}
                          marks={marks}
                          sx={{
                            mb: 3,
                            '& .MuiSlider-mark': {
                              height: 4,
                              width: 1.5,
                              backgroundColor: 'rgba(255,255,255,0.2)',
                            },
                            '& .MuiSlider-markLabel': {
                              fontSize: '0.55rem',
                              color: 'rgba(255,255,255,0.5)',
                              whiteSpace: 'nowrap',
                            },
                          }}
                        />
                      </Box>

                      {/* Remove button */}
                      <IconButton
                        size="small"
                        onClick={() => removeWindow(rosbag, wIdx)}
                        sx={{ color: 'rgba(255,255,255,0.4)' }}
                      >
                        <CloseIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  );
                })}
              </Box>
            );
          })}
        </Box>
      </Collapse>
    </Box>
  );
};

export default McapRangeFilter;
