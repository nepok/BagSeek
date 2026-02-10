import React, { useMemo, useRef, useState } from 'react';
import { styled } from '@mui/material/styles';

interface HeatBarProps {
  timestampCount: number;
  searchMarks: { value: number }[];
  mcapBoundaries?: number[];
  sliderValue?: number;
  bins?: number;
  windowSize?: number;
  height?: number;
  onHover?: (fraction: number, e: React.MouseEvent<HTMLDivElement>) => void;
  onLeave?: () => void;
}

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
}) => {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const [hoverPos, setHoverPos] = useState<number | null>(null); // 0..1 fraction

  const densityData = useMemo(() => {
    const counts = new Array(bins).fill(0);
    const total = timestampCount;
    const searchMarkPositions = searchMarks.map((m) => m.value / total);

    for (let i = 0; i < bins; i++) {
      const binCenter = i / bins;
      let score = 0;
      for (const smp of searchMarkPositions) {
        const dist = Math.abs(smp - binCenter);
        if (dist < windowSize / bins) {
          score += 1 - dist * bins / windowSize;
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
      onMouseLeave={() => { setHoverPos(null); onLeave && onLeave(); }}
      onMouseMove={(e) => {
        if (!trackRef.current) return;
        const rect = trackRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const frac = Math.max(0, Math.min(1, x / rect.width));
        setHoverPos(frac);
        onHover && onHover(frac, e);
      }}
    >
      {/* Heat segments with overflow hidden */}
      <SegmentBar height={height}>
        {densityData.map((val, idx) => (
          <HeatBarSegment key={idx} intensity={val} />
        ))}
      </SegmentBar>
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
