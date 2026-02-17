import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Menu, MenuItem, styled, useTheme } from '@mui/material';

export type HeatBarSelection = { startFrac: number; endFrac: number };

interface HeatBarProps {
  timestampCount: number;
  searchMarks: { value: number; rank?: number }[];
  mcapBoundaries?: number[];
  sliderValue?: number;
  bins?: number;
  windowSize?: number;
  height?: number;
  onHover?: (fraction: number, e: React.MouseEvent<HTMLDivElement>) => void;
  onLeave?: () => void;
  /** Drag-to-select: callback when selection changes. Passed null when cleared (click elsewhere). */
  onSelectionChange?: (selection: HeatBarSelection | null) => void;
  /** Controlled selection (optional). If not set, uses internal state from drag. */
  selection?: HeatBarSelection | null;
  /** Called when user right-clicks selection and chooses "Export section". */
  onExportSection?: (selection: HeatBarSelection) => void;
  /** Minimum selection width as fraction (0–1). Default 0.02 (2%). Smaller drags are ignored. */
  minSelectionWidth?: number;
}

const DEFAULT_MIN_SELECTION_WIDTH = 0.02;

function getHeatColor(value: number) {
  const hue = (1 - value) * 268;
  const sat = value * 70;
  return `hsl(${hue}, ${sat}%, 50%)`;
}

const TrackWrapper = styled('div')({
  width: '100%',
  position: 'relative',
});

const SegmentBar = styled('div')<{ height?: number }>(({ height }) => ({
  height: height ?? 2,
  width: '100%',
  display: 'flex',
  overflow: 'hidden',
  position: 'relative',
}));

const HeatBarSegment = styled('div')<{ intensity: number }>(({ intensity }) => ({
  flex: 1,
  height: '100%',
  background: getHeatColor(intensity),
}));

export const HeatBar: React.FC<HeatBarProps> = ({
  timestampCount,
  searchMarks,
  mcapBoundaries = [],
  sliderValue = 0,
  bins = 1000,
  windowSize = 50,
  height,
  onHover,
  onLeave,
  onSelectionChange,
  selection: controlledSelection,
  onExportSection,
  minSelectionWidth = DEFAULT_MIN_SELECTION_WIDTH,
}) => {
  const theme = useTheme();
  const primaryMain = theme.palette.primary.main;
  const primaryRgba = useCallback((alpha: number) => {
    let r = 25, g = 118, b = 210;
    if (typeof primaryMain !== 'string') return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    if (primaryMain.startsWith('#') && primaryMain.length >= 7) {
      r = parseInt(primaryMain.slice(1, 3), 16);
      g = parseInt(primaryMain.slice(3, 5), 16);
      b = parseInt(primaryMain.slice(5, 7), 16);
    } else if (primaryMain.startsWith('rgb')) {
      const m = primaryMain.match(/\d+/g);
      if (m && m.length >= 3) [r, g, b] = m.slice(0, 3).map(Number);
    }
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }, [primaryMain]);

  const trackRef = useRef<HTMLDivElement | null>(null);
  const [hoverPos, setHoverPos] = useState<number | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; selection: HeatBarSelection } | null>(null);
  const [dragState, setDragState] = useState<{ start: number; end: number } | null>(null);
  const internalSelectionRef = useRef<HeatBarSelection | null>(null);

  const getFracFromEvent = useCallback((e: MouseEvent | React.MouseEvent): number => {
    if (!trackRef.current) return 0;
    const rect = trackRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    return Math.max(0, Math.min(1, x / rect.width));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const frac = getFracFromEvent(e);
    setDragState({ start: frac, end: frac });
  }, [getFracFromEvent]);

  useEffect(() => {
    if (!dragState) return;
    const handleMove = (e: MouseEvent) => {
      const frac = getFracFromEvent(e);
      setDragState((prev) => prev ? { ...prev, end: frac } : null);
    };
    const handleUp = () => {
      setDragState((prev) => {
        if (!prev) return null;
        const [start, end] = prev.start <= prev.end ? [prev.start, prev.end] : [prev.end, prev.start];
        const width = end - start;
        if (width < minSelectionWidth) {
          internalSelectionRef.current = null;
          onSelectionChange?.(null);
          return null;
        }
        const sel: HeatBarSelection = { startFrac: start, endFrac: end };
        internalSelectionRef.current = sel;
        onSelectionChange?.(sel);
        return null;
      });
    };
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [dragState, getFracFromEvent, onSelectionChange, minSelectionWidth]);

  const rawDragSelection = dragState ? { startFrac: Math.min(dragState.start, dragState.end), endFrac: Math.max(dragState.start, dragState.end) } : null;
  const displaySelection = controlledSelection ?? rawDragSelection ?? internalSelectionRef.current;

  const densityData = useMemo(() => {
    const counts = new Array(bins).fill(0);
    const total = timestampCount;
    // Each mark's contribution is weighted by rank: rank 1 → 1.0, rank 100 → 0.2
    const searchMarkPositions = searchMarks.map((m) => ({
      pos: m.value / total,
      weight: m.rank != null ? 1.0 - 0.80 * Math.min(m.rank - 1, 99) / 99 : 1.0,
    }));

    for (let i = 0; i < bins; i++) {
      const binCenter = i / bins;
      let score = 0;
      for (const smp of searchMarkPositions) {
        const dist = Math.abs(smp.pos - binCenter);
        if (dist < windowSize / bins) {
          score += smp.weight * (1 - dist * bins / windowSize);
        }
      }
      counts[i] = score;
    }

    let max = 1;
    for (let j = 0; j < counts.length; j++) {
      if (counts[j] > max) max = counts[j];
    }
    const normalizedCounts = counts.map((c) => c / (max || 1));

    return normalizedCounts;
  }, [searchMarks, timestampCount, bins, windowSize]);

  // Determine which MCAP segment the slider is currently in
  let activeMcapIndex = 0;
  for (let i = mcapBoundaries.length - 1; i >= 0; i--) {
    if (sliderValue >= mcapBoundaries[i]) {
      activeMcapIndex = i;
      break;
    }
  }

  // Build segment info for MCAP indicators
  const segments = mcapBoundaries.map((startIdx, i) => {
    const endIdx = i < mcapBoundaries.length - 1 ? mcapBoundaries[i + 1] : timestampCount;
    return {
      index: i,
      startFrac: timestampCount > 0 ? startIdx / timestampCount : 0,
      endFrac: timestampCount > 0 ? endIdx / timestampCount : 0,
      isActive: i === activeMcapIndex,
    };
  });

  // Reduce label/line density when many boundaries: >40 → every 2nd, >80 → every 3rd, etc.
  const labelStep = Math.max(1, Math.ceil(mcapBoundaries.length / 40));

  return (
    <TrackWrapper
      ref={trackRef}
      onMouseDown={handleMouseDown}
      onMouseLeave={() => { setHoverPos(null); if (!dragState) onLeave?.(); }}
      onMouseMove={(e) => {
        if (!trackRef.current || dragState) return;
        const rect = trackRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const frac = Math.max(0, Math.min(1, x / rect.width));
        setHoverPos(frac);
        onHover?.(frac, e);
      }}
      style={{ userSelect: 'none', cursor: dragState ? 'col-resize' : 'default' }}
    >
      {/* Heat segments with overflow hidden */}
      <SegmentBar height={height}>
        {densityData.map((val, idx) => (
          <HeatBarSegment key={idx} intensity={val} />
        ))}
      </SegmentBar>
      {/* Selection highlight overlay - right-clickable */}
      {displaySelection && (
        <div
          onContextMenu={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setContextMenu({
              x: e.clientX,
              y: e.clientY,
              selection: displaySelection,
            });
          }}
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: `${displaySelection.startFrac * 100}%`,
            width: `${(displaySelection.endFrac - displaySelection.startFrac) * 100}%`,
            backgroundColor: primaryRgba(0.35),
            border: `1px solid ${primaryRgba(0.6)}`,
            cursor: 'context-menu',
          }}
        />
      )}
      <Menu
        open={contextMenu !== null}
        onClose={() => setContextMenu(null)}
        anchorReference="anchorPosition"
        anchorPosition={contextMenu ? { top: contextMenu.y, left: contextMenu.x } : undefined}
      >
        <MenuItem
          onClick={() => {
            if (contextMenu) {
              onExportSection?.(contextMenu.selection);
              setContextMenu(null);
            }
          }}
        >
          Export section
        </MenuItem>
      </Menu>
      {/* Hover indicator */}
      {hoverPos !== null && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: `${hoverPos * 100}%`,
            width: 2,
            background: 'rgba(255,255,255,0.8)',
            transform: 'translateX(-1px)',
            pointerEvents: 'none',
          }}
        />
      )}
      {/* MCAP boundary lines - always show all */}
      {segments.map((seg) => (
        seg.index > 0 && (
          <div
            key={`mcap-line-${seg.index}`}
            style={{
              position: 'absolute',
              top: -4,
              bottom: -4,
              left: `${seg.startFrac * 100}%`,
              width: 1,
              background: 'rgba(255,255,255,0.45)',
              pointerEvents: 'none',
            }}
          />
        )
      ))}
      {/* MCAP index labels - thinned when count > 40 (every 2nd, 3rd, etc.) */}
      {segments
        .filter((seg) => seg.index % labelStep === 0 || seg.index === activeMcapIndex)
        .map((seg) => (
          <div
            key={`mcap-label-${seg.index}`}
            style={{
              position: 'absolute',
              top: -14,
              left: `${((seg.startFrac + seg.endFrac) / 2) * 100}%`,
              transform: 'translateX(-50%)',
              fontSize: 9,
              color: seg.isActive ? '#90caf9' : 'rgba(255,255,255,0.35)',
              fontWeight: seg.isActive ? 600 : 400,
              pointerEvents: 'none',
              whiteSpace: 'nowrap',
            }}
          >
            {seg.index}
          </div>
        ))}
    </TrackWrapper>
  );
};
