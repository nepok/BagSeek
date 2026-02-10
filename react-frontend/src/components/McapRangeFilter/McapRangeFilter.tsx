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

/** Format a nanosecond timestamp string to HH:MM time-of-day. */
function formatNsToTime(nsStr: string | undefined): string {
  if (!nsStr) return '??:??';
  try {
    const ms = Number(BigInt(nsStr) / BigInt(1_000_000));
    const d = new Date(ms);
    return d.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '??:??';
  }
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

            // Build slider marks at 0%, 25%, 50%, 75%, 100% positions
            const markPositions = [0, 0.25, 0.5, 0.75, 1].map((frac) => {
              const idx = Math.round(frac * maxIdx);
              return {
                value: idx,
                label: formatNsToTime(ranges[idx]?.firstTimestampNs),
              };
            });
            // Deduplicate marks at same position
            const marks = markPositions.filter(
              (m, i, arr) => arr.findIndex((x) => x.value === m.value) === i
            );

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
                      {/* Start MCAP ID input */}
                      <TextField
                        size="small"
                        type="number"
                        value={startMcap?.mcapIdentifier ?? wStart}
                        onChange={(e) => {
                          const id = parseInt(e.target.value, 10);
                          if (isNaN(id)) return;
                          const idx = ranges.findIndex(
                            (r) => parseInt(r.mcapIdentifier, 10) >= id
                          );
                          if (idx >= 0) {
                            updateWindow(rosbag, wIdx, [
                              Math.min(idx, wEnd),
                              wEnd,
                            ]);
                          }
                        }}
                        slotProps={{
                          input: {
                            sx: {
                              width: 64,
                              '& input': {
                                textAlign: 'center',
                                py: 0.5,
                                fontSize: '0.75rem',
                              },
                            },
                          },
                        }}
                        title={`Start: MCAP ${startMcap?.mcapIdentifier} (${formatNsToTime(startMcap?.firstTimestampNs)})`}
                      />

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
                          sx={{ py: 1 }}
                        />
                      </Box>

                      {/* End MCAP ID input */}
                      <TextField
                        size="small"
                        type="number"
                        value={endMcap?.mcapIdentifier ?? wEnd}
                        onChange={(e) => {
                          const id = parseInt(e.target.value, 10);
                          if (isNaN(id)) return;
                          // Find last range with mcapIdentifier <= id
                          let idx = -1;
                          for (let i = ranges.length - 1; i >= 0; i--) {
                            if (parseInt(ranges[i].mcapIdentifier, 10) <= id) {
                              idx = i;
                              break;
                            }
                          }
                          if (idx >= 0) {
                            updateWindow(rosbag, wIdx, [
                              wStart,
                              Math.max(idx, wStart),
                            ]);
                          }
                        }}
                        slotProps={{
                          input: {
                            sx: {
                              width: 64,
                              '& input': {
                                textAlign: 'center',
                                py: 0.5,
                                fontSize: '0.75rem',
                              },
                            },
                          },
                        }}
                        title={`End: MCAP ${endMcap?.mcapIdentifier} (${formatNsToTime(endMcap?.lastTimestampNs)})`}
                      />

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
