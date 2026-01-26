import React, { useMemo, useRef, useState } from 'react';
import { styled } from '@mui/material/styles';

interface HeatBarProps {
  timestampCount: number;
  searchMarks: { value: number }[];
  timestampDensity?: number[];
  bins?: number;
  windowSize?: number;
  height?: number;
  onHover?: (fraction: number, e: React.MouseEvent<HTMLDivElement>) => void;
  onLeave?: () => void;
}

function getHeatColor(value: number, densityFactor: number) {
  const hue = (1 - value) * 268;
  const sat = densityFactor * 30 + value * 70;
  return `hsl(${hue}, ${sat}%, 50%)`;
}

// Math.max(...array) throws when the array is huge; this helper avoids spreading.
function arrayMax(values: number[] | undefined, fallback: number) {
  if (!values || values.length === 0) {
    return fallback;
  }

  let max = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > max) {
      max = values[i];
    }
  }
  return max;
}

const StyledTrack = styled('div')<{ height?: number }>(({ height }) => ({
  height: height ?? 2,
  width: '100%',
  position: 'relative',
  display: 'flex',
  overflow: 'hidden',
}));

const HeatBarSegment = styled('div')<{ intensity: number; densityFactor: number }>(({ intensity, densityFactor }) => ({
  flex: 1,
  height: '100%',
  background: getHeatColor(intensity, densityFactor),
}));

export const HeatBar: React.FC<HeatBarProps> = ({
  timestampCount,
  searchMarks,
  timestampDensity,
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

    const max = arrayMax(counts, 1);
    const normalizedCounts = counts.map((c) => c / (max || 1));

    return normalizedCounts;
  }, [searchMarks, timestampCount, bins, windowSize]);

  const normalizedDensity = useMemo(() => {
    const source = timestampDensity && timestampDensity.length > 0
      ? timestampDensity
      : new Array(bins).fill(0);

    const max = arrayMax(source, 1);
    const normalized = source.map((v) => v / (max || 1));

    const resized = new Array(bins).fill(0);
    const factor = (timestampDensity?.length ?? bins) / bins;

    for (let i = 0; i < bins; i++) {
      const pos = i * factor;
      const low = Math.floor(pos);
      const high = Math.ceil(pos);
      const weight = pos - low;

      const lowVal = normalized[low] ?? 0;
      const highVal = normalized[high] ?? 0;

      resized[i] = lowVal * (1 - weight) + highVal * weight;
    }

    return resized;
  }, [timestampDensity, bins]);

  return (
    <StyledTrack
      ref={trackRef}
      height={height}
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
        {densityData.map((val, idx) => (
        <HeatBarSegment key={idx} intensity={val} densityFactor={normalizedDensity[idx]} />
        ))}
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
    </StyledTrack>
  );
};
