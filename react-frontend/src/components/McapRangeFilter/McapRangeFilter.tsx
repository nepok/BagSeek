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
import './McapRangeFilter.css';

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
  /** Stable IDs for React keys; parallel to windows */
  windowIds?: string[];
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
  /** Additional rosbags to preload metadata for (e.g. dropdown display). */
  rosbagsToPreload?: string[];
  /** When true, only run effects (fetch, etc.) and render nothing. Use when embedding MCAP UI inline elsewhere. */
  logicOnly?: boolean;
  /** When true, skip applying pending MCAP IDs until refresh completes (avoids consuming before rosbags are complete). */
  deferPendingUntilAfterRefresh?: boolean;
}

export interface McapRangeFilterItemProps {
  rosbag: string;
  mcapFilters: McapFilterState;
  onMcapFiltersChange: (filters: McapFilterState) => void;
}

/** Format nanosecond timestamp to HH:MM time-of-day. Exported for use in Rosbag select. */
export function formatNsToTime(nsStr: string | undefined): string {
  const ms = nsToMs(nsStr);
  if (ms === null) return '??:??';
  const d = new Date(ms);
  return d.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
}

/** Generate a stable id for a window (for React keys). */
function genWindowId() {
  return `w-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/** Ensure windowIds array exists and matches windows length. */
function ensureWindowIds(filter: { windows: [number, number][]; windowIds?: string[] }) {
  const { windows, windowIds = [] } = filter;
  if (windowIds.length === windows.length) return windowIds;
  const result = [...windowIds];
  while (result.length < windows.length) result.push(genWindowId());
  return result.slice(0, windows.length);
}

/** Convert a set of MCAP IDs to contiguous [startIdx, endIdx] windows into a ranges array. */
function mcapIdsToWindows(
  ranges: McapRangeMeta[],
  mcapIds: string[]
): [number, number][] {
  const idSet = new Set(mcapIds.map((id) => String(id)));
  const indices = ranges
    .map((r, i) => (idSet.has(String(r.mcapIdentifier)) ? i : -1))
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

/** Format a nanosecond timestamp string to HH:MM:SS time-of-day. */
function formatNsToTimeWithSeconds(nsStr: string | undefined): string {
  const ms = nsToMs(nsStr);
  if (ms === null) return '??:??:??';
  const d = new Date(ms);
  return d.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

const ONE_MIN_MS = 60_000;
const labelStepFraction = 14;

/** Custom track: top = whole MCAP IDs (by index), bottom = whole minutes (by time). Not aligned. */
interface McapRangeTrackProps extends React.HTMLAttributes<HTMLSpanElement> {
  ranges: McapRangeMeta[];
  maxIdx: number;
  value: [number, number];
}

function McapRangeTrack({
  ranges,
  maxIdx,
  value,
  style,
  className,
  ...other
}: McapRangeTrackProps) {
  const [wStart, wEnd] = value;
  const firstMs = nsToMs(ranges[0]?.firstTimestampNs);
  const lastMs = nsToMs(ranges[maxIdx]?.lastTimestampNs);

  // Top: whole MCAP IDs only (0, 1, 2...) - positioned by index fraction
  const idLabelStep = Math.max(1, Math.ceil(ranges.length / labelStepFraction));
  const shouldShowId = (i: number) => i % idLabelStep === 0;

  // Bottom: whole minutes only - positioned by time fraction
  const minuteMarks: { minuteMs: number; frac: number }[] = [];
  if (firstMs !== null && lastMs !== null && lastMs > firstMs) {
    const startMinute = Math.ceil(firstMs / ONE_MIN_MS) * ONE_MIN_MS;
    const endMinute = Math.floor(lastMs / ONE_MIN_MS) * ONE_MIN_MS;
    const durationMs = lastMs - firstMs;
    for (let t = startMinute; t <= endMinute; t += ONE_MIN_MS) {
      const frac = (t - firstMs) / durationMs;
      minuteMarks.push({ minuteMs: t, frac });
    }
  }
  const todLabelStep = Math.max(1, Math.ceil(minuteMarks.length / labelStepFraction));
  const shouldShowTod = (i: number) => i % todLabelStep === 0;

  const labelStyle = {
    position: 'absolute' as const,
    transform: 'translateX(-50%)',
    fontSize: 9,
    pointerEvents: 'none' as const,
    whiteSpace: 'nowrap' as const,
  };

  const tickStyle = {
    position: 'absolute' as const,
    left: '50%',
    transform: 'translateX(-50%)',
    width: 1,
    backgroundColor: 'currentColor',
    pointerEvents: 'none' as const,
  };

  return (
    <span
      style={{
        display: 'block',
        position: 'absolute',
        left: 0,
        right: 0,
        top: '50%',
        transform: 'translateY(-50%)',
        pointerEvents: 'none',
      }}
    >
      {/* "ID" label on left, same height as ID row */}
      <div
        style={{
          position: 'absolute',
          top: -23,
          left: -22,
          width: 20,
          fontSize: 9,
          color: 'rgba(255,255,255,1)',
          whiteSpace: 'nowrap',
          textAlign: 'center',
          transform: 'translateX(-11px)',
        }}
      >
        ID
      </div>
      {/* "TOD" label on left, same height as time row */}
      <div
        style={{
          position: 'absolute',
          bottom: -25,
          left: -22,
          width: 20,
          fontSize: 9,
          color: 'rgba(255,255,255,1)',
          whiteSpace: 'nowrap',
          textAlign: 'center',
          transform: 'translateX(-11px)',
        }}
      >
        TOD
      </div>
      {/* Visual range fill between thumbs */}
      {maxIdx > 0 && (
        <div
          style={{
            position: 'absolute',
            left: `${(wStart / maxIdx) * 100}%`,
            width: `${((wEnd - wStart) / maxIdx) * 100}%`,
            top: '50%',
            transform: 'translateY(-50%)',
            height: 8,
            backgroundColor: 'rgba(180, 159, 204, 0.05)',
            borderRadius: 4,
            pointerEvents: 'none',
            border: '1px solid rgba(180, 159, 204, 0.4)',
          }}
        />
      )}
      {/* ID labels above + ticks pointing down to slider */}
      <div
        style={{
          position: 'absolute',
          top: -23,
          left: 0,
          right: 0,
          height: 0,
        }}
      >
        {ranges.map((r, i) => {
          if (!shouldShowId(i)) return null;
          const frac = maxIdx > 0 ? i / maxIdx : 0;
          return (
            <div
              key={`id-${i}`}
              style={{
                ...labelStyle,
                left: `${frac * 100}%`,
                color: 'rgba(255,255,255,0.5)',
                fontWeight: 400,
              }}
            >
              {r.mcapIdentifier}
              {/* Tick under ID pointing down to slider */}
              <div
                style={{
                  ...tickStyle,
                  top: '100%',
                  marginTop: 2,
                  height: 5,
                }}
              />
            </div>
          );
        })}
      </div>
      {/* MUI track bar (selected range) - must use original style/className for positioning */}
      <span className={className} style={style} {...other} />
      {/* Minute ticks for TOD: bigger for labeled, smaller for unlabeled */}
      <div
        style={{
          position: 'absolute',
          bottom: -8,
          left: 0,
          right: 0,
          height: 0,
        }}
      >
        {minuteMarks.map((m, i) => {
          const hasLabel = shouldShowTod(i);
          return (
            <div
              key={`min-tick-${m.minuteMs}`}
              style={{
                ...tickStyle,
                left: `${m.frac * 100}%`,
                ...(hasLabel
                  ? { top: -3, height: 5, opacity: 0.6 }
                  : { bottom: 0, height: 3, opacity: 0.4 }),
              }}
            />
          );
        })}
      </div>
      {/* TOD labels below (only where shown) - no ticks, ticks are above */}
      <div
        style={{
          position: 'absolute',
          bottom: -12,
          left: 0,
          right: 0,
          height: 0,
        }}
      >
        {minuteMarks.map((m, i) => {
          if (!shouldShowTod(i)) return null;
          const ns = BigInt(Math.round(m.minuteMs)) * BigInt(1_000_000);
          const timeStr = formatNsToTime(String(ns));
          return (
            <div
              key={`tod-${m.minuteMs}`}
              style={{
                ...labelStyle,
                left: `${m.frac * 100}%`,
                top: '100%',
                marginTop: 2,
                color: 'rgba(255,255,255,0.35)',
                fontWeight: 400,
              }}
            >
              {timeStr}
            </div>
          );
        })}
      </div>
    </span>
  );
}

/** Per-rosbag MCAP range UI for inline use (e.g. inside Rosbag selector). */
export const McapRangeFilterItem: React.FC<McapRangeFilterItemProps> = React.memo(({
  rosbag,
  mcapFilters,
  onMcapFiltersChange,
}) => {
  const filter = mcapFilters[rosbag];
  // Local draft values while dragging — keyed by window index. Only the actively
  // dragged window has an entry; all others read from the committed filter state.
  const [drafts, setDrafts] = useState<Record<number, [number, number]>>({});

  if (!filter || filter.ranges.length <= 1) return null;

  const { ranges, windows } = filter;
  const maxIdx = ranges.length - 1;
  const globalFirstTime = formatNsToTime(ranges[0]?.firstTimestampNs);
  const globalLastTime = formatNsToTime(ranges[maxIdx]?.lastTimestampNs);

  const addWindow = () => {
    const ids = ensureWindowIds(filter);
    onMcapFiltersChange({
      ...mcapFilters,
      [rosbag]: {
        ...filter,
        windows: [...filter.windows, [0, maxIdx] as [number, number]],
        windowIds: [...ids, genWindowId()],
      },
    });
  };

  const removeWindow = (windowIdx: number) => {
    const [wStart, wEnd] = filter.windows[windowIdx] ?? [0, maxIdx];
    if (wStart === 0 && wEnd === maxIdx && filter.windows.length <= 1) return;
    const ids = ensureWindowIds(filter);
    if (filter.windows.length <= 1) {
      onMcapFiltersChange({
        ...mcapFilters,
        [rosbag]: {
          ...filter,
          windows: [[0, maxIdx] as [number, number]],
          windowIds: [genWindowId()],
        },
      });
    } else {
      onMcapFiltersChange({
        ...mcapFilters,
        [rosbag]: {
          ...filter,
          windows: filter.windows.filter((_, i) => i !== windowIdx),
          windowIds: ids.filter((_, i) => i !== windowIdx),
        },
      });
    }
  };

  const commitWindow = (windowIdx: number, value: [number, number]) => {
    setDrafts((d) => { const next = { ...d }; delete next[windowIdx]; return next; });
    onMcapFiltersChange({
      ...mcapFilters,
      [rosbag]: {
        ...filter,
        windows: filter.windows.map((w, i) => (i === windowIdx ? value : w)),
      },
    });
  };

  return (
    <Box
      className="mcap-range-filter"
      sx={{
        mt: 0.75,
        mb: 0.5,
        ml: 2.5,
        border: '1px solid rgba(255, 255, 255, 0.15)',
        borderRadius: 1,
        p: 1,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 0.5,
          pb: 0.5,
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Typography
          variant="caption"
          sx={{
            color: 'rgba(255,255,255,0.6)',
            fontSize: '0.7rem',
          }}
        >
          {ranges.length} MCAPs · {globalFirstTime} – {globalLastTime}
        </Typography>
        <IconButton
          size="small"
          onClick={addWindow}
          title="Add MCAP range"
          sx={{ color: 'rgba(255,255,255,0.5)' }}
        >
          <AddIcon fontSize="small" />
        </IconButton>
      </Box>
      {windows.map((window, wIdx) => {
        const draft = drafts[wIdx];
        const [wStart, wEnd] = draft ?? window;
        const windowIds = ensureWindowIds(filter);
        const windowId = windowIds[wIdx] ?? `fallback-${wIdx}`;
        const beginId = ranges[wStart]?.mcapIdentifier ?? '';
        const endId = ranges[wEnd]?.mcapIdentifier ?? '';

        const handleBeginIdBlur = (e: React.FocusEvent<HTMLInputElement>) => {
          const val = e.target.value.trim();
          const idx = ranges.findIndex((r) => r.mcapIdentifier === val);
          if (idx >= 0) commitWindow(wIdx, [idx, Math.max(idx, wEnd)]);
        };
        const handleEndIdBlur = (e: React.FocusEvent<HTMLInputElement>) => {
          const val = e.target.value.trim();
          const idx = ranges.findIndex((r) => r.mcapIdentifier === val);
          if (idx >= 0) commitWindow(wIdx, [Math.min(idx, wStart), idx]);
        };

        return (
          <Box
            key={windowId}
            sx={{
              display: 'flex',
              flexDirection: 'column',
              gap: 0.5,
              mb: 0.5,
              p: 1,
              border: '1px solid rgba(255, 255, 255, 0.12)',
              borderRadius: 1,
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 1 }}>
              <TextField
                key={`${windowId}-begin-${beginId}`}
                size="small"
                label="Begin ID"
                defaultValue={beginId}
                onBlur={handleBeginIdBlur}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                }}
                sx={{
                  flex: 1,
                  '& .MuiInputBase-input': { fontSize: '0.75rem' },
                  '& .MuiInputLabel-root': { fontSize: '0.75rem' },
                }}
              />
              <TextField
                key={`${windowId}-end-${endId}`}
                size="small"
                label="End ID"
                defaultValue={endId}
                onBlur={handleEndIdBlur}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                }}
                sx={{
                  flex: 1,
                  '& .MuiInputBase-input': { fontSize: '0.75rem' },
                  '& .MuiInputLabel-root': { fontSize: '0.75rem' },
                }}
              />
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ flexGrow: 1, pl: 5, pr: 1 }}>
                <Slider
                  value={[wStart, wEnd]}
                  onChange={(_, newValue) =>
                    setDrafts((d) => ({ ...d, [wIdx]: newValue as [number, number] }))
                  }
                  onChangeCommitted={(_, newValue) =>
                    commitWindow(wIdx, newValue as [number, number])
                  }
                  min={0}
                  max={maxIdx}
                  valueLabelDisplay="off"
                  components={{
                    Track: (props) => (
                      <McapRangeTrack
                        {...props}
                        ranges={ranges}
                        maxIdx={maxIdx}
                        value={[wStart, wEnd]}
                      />
                    ),
                  }}
                  sx={{ my: 1 }}
                />
              </Box>
              <IconButton
                size="small"
                onClick={() => removeWindow(wIdx)}
                disabled={windows.length <= 1}
                sx={{
                  color: 'rgba(255,255,255,0.4)',
                  '&.Mui-disabled': { color: 'rgba(255,255,255,0.15)', opacity: 0.6 },
                }}
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            </Box>
          </Box>
        );
      })}
    </Box>
  );
}, (prev, next) =>
  prev.rosbag === next.rosbag &&
  prev.mcapFilters[prev.rosbag] === next.mcapFilters[next.rosbag]
);

const McapRangeFilter: React.FC<McapRangeFilterProps> = ({
  selectedRosbags,
  mcapFilters,
  onMcapFiltersChange,
  pendingMcapIds,
  onPendingMcapIdsConsumed,
  rosbagsToPreload,
  logicOnly = false,
  deferPendingUntilAfterRefresh = false,
}) => {
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState<Set<string>>(new Set());
  // Cache fetched MCAP metadata so we don't refetch on every render
  const metaCache = useRef<Record<string, McapRangeMeta[]>>({});
  // Ref to latest mcapFilters to avoid stale closures in async callbacks
  const filtersRef = useRef(mcapFilters);
  filtersRef.current = mcapFilters;
  // Ref to latest pendingMcapIds so fetch-meta callbacks can apply them inline
  const pendingRef = useRef(pendingMcapIds);
  pendingRef.current = pendingMcapIds;

  // Fetch MCAP metadata for selected rosbags + preload list (e.g. for dropdown display) that aren't cached yet
  useEffect(() => {
    const toPreload = rosbagsToPreload ?? [];
    const allToConsider = Array.from(new Set([...selectedRosbags, ...toPreload]));
    console.log(`\t\t[MCAP-DEBUG] fetch-meta effect: selectedRosbags=${selectedRosbags.length}, preload=${toPreload.length}, cached=${Object.keys(metaCache.current).length}, loading=${loading.size}`);
    const toFetch = allToConsider.filter(
      (r) => !metaCache.current[r] && !loading.has(r)
    );
    if (toFetch.length === 0) {
      console.log('\t\t[MCAP-DEBUG] fetch-meta: nothing to fetch');
      return;
    }
    console.log(`\t\t[MCAP-DEBUG] fetch-meta: fetching ${toFetch.length} rosbags:`, toFetch);

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
        console.log(`\t\t[MCAP-DEBUG] fetch-meta: "${rosbag}" -> ${ranges.length} ranges, IDs=[${ranges.map(r => r.mcapIdentifier).join(',')}]`);

        // Initialize filter state for this rosbag if not present
        if (!filtersRef.current[rosbag]) {
          const maxIdx = Math.max(0, ranges.length - 1);
          let windows: [number, number][] = [[0, maxIdx]];
          let windowIds = [genWindowId()];

          // Apply pending MCAP IDs inline (from positional/map filter) instead of full range
          const pending = pendingRef.current;
          if (pending) {
            const rosbagName = extractRosbagName(rosbag);
            const pendingKey = Object.keys(pending).find(
              (k) => k === rosbagName || k === rosbag
            );
            if (pendingKey) {
              const pendingWindows = mcapIdsToWindows(ranges, pending[pendingKey]);
              if (pendingWindows.length > 0) {
                windows = pendingWindows;
                windowIds = pendingWindows.map(() => genWindowId());
                console.log(`\t\t[MCAP-DEBUG] fetch-meta: initializing "${rosbag}" with PENDING windows ${JSON.stringify(windows)} (from ${pending[pendingKey].length} mcapIds), filtersRef has ${Object.keys(filtersRef.current).length} entries before`);
              } else {
                console.log(`\t\t[MCAP-DEBUG] fetch-meta: initializing "${rosbag}" with full-range window [0, ${maxIdx}] (pending mcapIds produced 0 windows), filtersRef has ${Object.keys(filtersRef.current).length} entries before`);
              }
            } else {
              console.log(`\t\t[MCAP-DEBUG] fetch-meta: initializing "${rosbag}" with full-range window [0, ${maxIdx}], filtersRef has ${Object.keys(filtersRef.current).length} entries before`);
            }
          } else {
            console.log(`\t\t[MCAP-DEBUG] fetch-meta: initializing "${rosbag}" with full-range window [0, ${maxIdx}], filtersRef has ${Object.keys(filtersRef.current).length} entries before`);
          }

          const updated = {
            ...filtersRef.current,
            [rosbag]: { ranges, windows, windowIds },
          };
          // Update ref synchronously so concurrent async callbacks see accumulated state
          filtersRef.current = updated;
          onMcapFiltersChange(updated);
        } else {
          console.log(`\t\t[MCAP-DEBUG] fetch-meta: "${rosbag}" already has filter state, NOT overwriting`);
        }
      } catch (e) {
        console.error(`\t\t[MCAP-DEBUG] fetch-meta: FAILED for "${rosbag}":`, e);
      } finally {
        setLoading((prev) => {
          const next = new Set(prev);
          next.delete(rosbag);
          return next;
        });
      }
    });
  }, [selectedRosbags, rosbagsToPreload]); // eslint-disable-line react-hooks/exhaustive-deps

  // Ensure windowIds exist and are persisted (for stable React keys)
  useEffect(() => {
    let changed = false;
    const next = { ...mcapFilters };
    for (const rosbag of selectedRosbags) {
      const filter = next[rosbag];
      if (!filter?.windows?.length) continue;
      const ids = ensureWindowIds(filter);
      if (ids !== filter.windowIds) {
        next[rosbag] = { ...filter, windowIds: ids };
        changed = true;
      }
    }
    if (changed) onMcapFiltersChange(next);
  }, [mcapFilters, selectedRosbags]); // eslint-disable-line react-hooks/exhaustive-deps

  // Ensure every rosbag with ranges has at least one window (full range) so thumbs are always movable
  useEffect(() => {
    let changed = false;
    const next = { ...mcapFilters };
    for (const rosbag of selectedRosbags) {
      const filter = next[rosbag];
      if (!filter?.ranges?.length) continue;
      if (filter.windows.length === 0) {
        const maxIdx = Math.max(0, filter.ranges.length - 1);
        console.log(`\t\t[MCAP-DEBUG] ensure-full-range: "${extractRosbagName(rosbag)}" has 0 windows, resetting to [0, ${maxIdx}]`);
        next[rosbag] = {
          ...filter,
          windows: [[0, maxIdx] as [number, number]],
          windowIds: [genWindowId()],
        };
        changed = true;
      }
    }
    if (changed) {
      console.log('\t\t[MCAP-DEBUG] ensure-full-range: CHANGED, calling onMcapFiltersChange');
      onMcapFiltersChange(next);
    }
  }, [mcapFilters, selectedRosbags]); // eslint-disable-line react-hooks/exhaustive-deps

  // Clean up filters for rosbags that are neither selected nor in preload (dropdown)
  useEffect(() => {
    const keepSet = new Set([...selectedRosbags, ...(rosbagsToPreload ?? [])]);
    const toRemove = Object.keys(mcapFilters).filter((r) => !keepSet.has(r));
    if (toRemove.length > 0) {
      console.log(`\t\t[MCAP-DEBUG] cleanup: removing ${toRemove.length} deselected rosbags:`, toRemove.map(r => extractRosbagName(r)));
      const next = { ...mcapFilters };
      toRemove.forEach((r) => delete next[r]);
      onMcapFiltersChange(next);
    }
  }, [selectedRosbags, rosbagsToPreload]); // eslint-disable-line react-hooks/exhaustive-deps

  // Apply pending MCAP IDs from the positional filter once ranges are loaded
  useEffect(() => {
    console.log(`\t\t[MCAP-DEBUG] pending effect: pendingMcapIds=${pendingMcapIds ? Object.keys(pendingMcapIds).length + ' keys' : 'null'}, selectedRosbags=${selectedRosbags.length}, mcapFilters keys=${Object.keys(mcapFilters).length}, deferPending=${deferPendingUntilAfterRefresh}`);
    if (!pendingMcapIds) {
      console.log('\t\t[MCAP-DEBUG] pending effect: pendingMcapIds is null, returning');
      return;
    }
    if (deferPendingUntilAfterRefresh) {
      console.log('\t\t[MCAP-DEBUG] pending effect: deferring until file path refresh completes');
      return;
    }
    // Don't process until rosbags have been selected (auto-select may not have run yet)
    if (selectedRosbags.length === 0) {
      console.log('\t\t[MCAP-DEBUG] pending effect: selectedRosbags is empty, waiting for auto-select');
      return;
    }

    // Resolve pendingMcapIds keys (map uses rosbag name) to full paths from selectedRosbags
    const resolveKey = (key: string) =>
      selectedRosbags.find((p) => extractRosbagName(p) === key || p === key) ?? key;

    // Only consider rosbags that are actually in the current selection
    const pendingEntries = Object.entries(pendingMcapIds);
    const relevantEntries = pendingEntries.filter(([key]) => {
      const rosbag = resolveKey(key);
      const isRelevant = selectedRosbags.includes(rosbag);
      if (!isRelevant) {
        console.log(`\t\t[MCAP-DEBUG] pending effect: key="${key}" -> resolved="${rosbag}" NOT in selectedRosbags`);
      }
      return isRelevant;
    });

    console.log(`\t\t[MCAP-DEBUG] pending effect: ${pendingEntries.length} pending, ${relevantEntries.length} relevant`);

    // If no relevant entries (all pending rosbags lack embeddings), consume and return
    if (relevantEntries.length === 0) {
      console.log('\t\t[MCAP-DEBUG] pending effect: NO relevant entries -> consuming');
      onPendingMcapIdsConsumed?.();
      return;
    }

    // Wait until all relevant rosbags have their ranges loaded
    const allLoaded = relevantEntries.every(([key]) => {
      const rosbag = resolveKey(key);
      const hasRanges = mcapFilters[rosbag]?.ranges?.length;
      if (!hasRanges) {
        console.log(`\t\t[MCAP-DEBUG] pending effect: "${key}" -> "${rosbag}" NOT loaded yet (ranges=${mcapFilters[rosbag]?.ranges?.length ?? 'undefined'})`);
      }
      return hasRanges;
    });
    if (!allLoaded) {
      console.log('\t\t[MCAP-DEBUG] pending effect: not all loaded, waiting...');
      return;
    }

    console.log('\t\t[MCAP-DEBUG] pending effect: ALL loaded, applying windows...');
    const next = { ...mcapFilters };
    let applied = false;
    for (const [key, mcapIds] of relevantEntries) {
      const rosbag = resolveKey(key);
      const filter = next[rosbag];
      if (!filter || filter.ranges.length === 0) continue;
      const windows = mcapIdsToWindows(filter.ranges, mcapIds);
      console.log(`\t\t[MCAP-DEBUG] pending effect: "${key}" -> ${mcapIds.length} mcapIds -> ${windows.length} windows: ${JSON.stringify(windows)}`);
      if (windows.length > 0) {
        next[rosbag] = {
          ...filter,
          windows,
          windowIds: windows.map(() => genWindowId()),
        };
        applied = true;
      }
    }
    if (applied) {
      console.log('\t\t[MCAP-DEBUG] pending effect: APPLIED windows, calling onMcapFiltersChange');
      filtersRef.current = next;
      onMcapFiltersChange(next);
      setExpanded(true);
    } else {
      console.log('\t\t[MCAP-DEBUG] pending effect: nothing applied (no windows produced)');
    }
    console.log('\t\t[MCAP-DEBUG] pending effect: consuming pendingMcapIds');
    onPendingMcapIdsConsumed?.();
  }, [pendingMcapIds, mcapFilters, selectedRosbags, deferPendingUntilAfterRefresh]); // eslint-disable-line react-hooks/exhaustive-deps

  const addWindow = (rosbag: string) => {
    const filter = mcapFilters[rosbag];
    if (!filter) return;
    const maxIdx = Math.max(0, filter.ranges.length - 1);
    const ids = ensureWindowIds(filter);
    const next = {
      ...mcapFilters,
      [rosbag]: {
        ...filter,
        windows: [...filter.windows, [0, maxIdx] as [number, number]],
        windowIds: [...ids, genWindowId()],
      },
    };
    onMcapFiltersChange(next);
    if (!expanded) setExpanded(true);
  };

  const removeWindow = (rosbag: string, windowIdx: number) => {
    const filter = mcapFilters[rosbag];
    if (!filter) return;
    const maxIdx = Math.max(0, filter.ranges.length - 1);
    const [wStart, wEnd] = filter.windows[windowIdx] ?? [0, maxIdx];
    // Do nothing when full range is selected
    if (wStart === 0 && wEnd === maxIdx && filter.windows.length <= 1) return;
    const ids = ensureWindowIds(filter);
    // If removing the last window, reset to full range instead of leaving empty
    if (filter.windows.length <= 1) {
      const next = {
        ...mcapFilters,
        [rosbag]: {
          ...filter,
          windows: [[0, maxIdx] as [number, number]],
          windowIds: [genWindowId()],
        },
      };
      onMcapFiltersChange(next);
    } else {
      const next = {
        ...mcapFilters,
        [rosbag]: {
          ...filter,
          windows: filter.windows.filter((_, i) => i !== windowIdx),
          windowIds: ids.filter((_, i) => i !== windowIdx),
        },
      };
      onMcapFiltersChange(next);
    }
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
  const allEntries = selectedRosbags.map((r) => ({ rosbag: r, filter: mcapFilters[r] }));
  const rosbagEntries = allEntries.filter((e) => e.filter && e.filter.ranges.length > 1);

  if (logicOnly) return null;

  const hiddenSingleMcap = allEntries.filter((e) => e.filter && e.filter.ranges.length === 1);
  const hiddenNoFilter = allEntries.filter((e) => !e.filter);
  const hiddenEmptyRanges = allEntries.filter((e) => e.filter && e.filter.ranges.length === 0);
  console.log(`\t\t[MCAP-DEBUG] rosbagEntries: ${rosbagEntries.length} shown, ${hiddenSingleMcap.length} hidden (1 mcap), ${hiddenNoFilter.length} hidden (no filter), ${hiddenEmptyRanges.length} hidden (empty ranges), total selected=${selectedRosbags.length}`);
  if (hiddenNoFilter.length > 0) {
    console.log(`\t\t[MCAP-DEBUG] rosbagEntries hidden (no filter):`, hiddenNoFilter.map(e => extractRosbagName(e.rosbag)));
  }
  rosbagEntries.forEach(e => {
    const w = e.filter!.windows;
    const maxIdx = e.filter!.ranges.length - 1;
    const isFullRange = w.length === 1 && w[0][0] === 0 && w[0][1] === maxIdx;
    console.log(`\t\t[MCAP-DEBUG] rosbagEntry: "${extractRosbagName(e.rosbag)}" ranges=${e.filter!.ranges.length} windows=${JSON.stringify(w)} fullRange=${isFullRange}`);
  });

  if (rosbagEntries.length === 0) return null;

  return (
    <Box
      className="mcap-range-filter"
      sx={{
        border: '1px solid rgba(255, 255, 255, 0.23)',
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
          '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.06)' },
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
        <Box
          sx={{
            px: 2,
            pt: 1,
            pb: 1.5,
            borderTop: '1px solid rgba(255, 255, 255, 0.08)',
          }}
        >
          {rosbagEntries.map(({ rosbag, filter }) => {
            const { ranges, windows } = filter!;
            const maxIdx = ranges.length - 1;
            const displayName = extractRosbagName(rosbag);
            const globalFirstTime = formatNsToTime(ranges[0]?.firstTimestampNs);
            const globalLastTime = formatNsToTime(
              ranges[maxIdx]?.lastTimestampNs
            );

            return (
              <Box
                key={rosbag}
                sx={{
                  mt: 1,
                  border: '1px solid rgba(255, 255, 255, 0.15)',
                  borderRadius: 1,
                  p: 1,
                }}
              >
                {/* Rosbag header */}
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    mb: 0.5,
                    pb: 0.5,
                    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
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

                {/* Slider(s): one per window (always at least one, pre-added as full range) */}
                {windows.map((window, wIdx) => {
                  const [wStart, wEnd] = window;
                  const windowIds = ensureWindowIds(filter!);
                  const windowId = windowIds[wIdx] ?? `fallback-${wIdx}`;
                  const beginId = ranges[wStart]?.mcapIdentifier ?? '';
                  const endId = ranges[wEnd]?.mcapIdentifier ?? '';

                  const handleBeginIdBlur = (e: React.FocusEvent<HTMLInputElement>) => {
                    const val = e.target.value.trim();
                    const idx = ranges.findIndex((r) => r.mcapIdentifier === val);
                    if (idx >= 0) updateWindow(rosbag, wIdx, [idx, Math.max(idx, wEnd)]);
                  };
                  const handleEndIdBlur = (e: React.FocusEvent<HTMLInputElement>) => {
                    const val = e.target.value.trim();
                    const idx = ranges.findIndex((r) => r.mcapIdentifier === val);
                    if (idx >= 0) updateWindow(rosbag, wIdx, [Math.min(idx, wStart), idx]);
                  };

                  return (
                    <Box
                      key={windowId}
                      sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 0.5,
                        mb: 0.5,
                        p: 1,
                        border: '1px solid rgba(255, 255, 255, 0.12)',
                        borderRadius: 1,
                      }}
                    >
                      {/* Top row: beginning ID (left) and end ID (right) */}
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 1 }}>
                        <TextField
                          key={`${windowId}-begin-${beginId}`}
                          size="small"
                          label="Begin ID"
                          defaultValue={beginId}
                          onBlur={handleBeginIdBlur}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                          }}
                          sx={{
                            flex: 1,
                            '& .MuiInputBase-input': { fontSize: '0.75rem' },
                            '& .MuiInputLabel-root': { fontSize: '0.75rem' },
                          }}
                        />
                        <TextField
                          key={`${windowId}-end-${endId}`}
                          size="small"
                          label="End ID"
                          defaultValue={endId}
                          onBlur={handleEndIdBlur}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                          }}
                          sx={{
                            flex: 1,
                            '& .MuiInputBase-input': { fontSize: '0.75rem' },
                            '& .MuiInputLabel-root': { fontSize: '0.75rem' },
                          }}
                        />
                      </Box>
                      {/* Range slider with custom track (ID above, TOD below) + remove button */}
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box sx={{ flexGrow: 1, pl: 5, pr: 1 }}>
                          <Slider
                            value={[wStart, wEnd]}
                            onChange={(_, newValue) =>
                              updateWindow(rosbag, wIdx, newValue as [number, number])
                            }
                            min={0}
                            max={maxIdx}
                            valueLabelDisplay="off"
                            components={{
                              Track: (props) => (
                                <McapRangeTrack
                                  {...props}
                                  ranges={ranges}
                                  maxIdx={maxIdx}
                                  value={[wStart, wEnd]}
                                />
                              ),
                            }}
                            sx={{
                              my: 1,
                            }}
                          />
                        </Box>
                        <IconButton
                          size="small"
                          onClick={() => removeWindow(rosbag, wIdx)}
                          disabled={windows.length <= 1}
                          sx={{
                            color: 'rgba(255,255,255,0.4)',
                            '&.Mui-disabled': {
                              color: 'rgba(255,255,255,0.15)',
                              opacity: 0.6,
                            },
                          }}
                        >
                          <CloseIcon fontSize="small" />
                        </IconButton>
                      </Box>
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
