import React, { useMemo } from 'react';
import { styled } from '@mui/material/styles';

interface HeatBarProps {
  timestampCount: number;
  searchMarks: { value: number }[];
  timestampDensity?: number[];
  bins?: number;
  windowSize?: number;
  height?: number;
}

function getHeatColor(value: number, densityFactor: number) {
  const hue = (1 - value) * 268;
  const sat = densityFactor * 30 + value * 70;
  return `hsl(${hue}, ${sat}%, 50%)`;
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
}) => {
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

    const max = Math.max(...counts);
    return counts.map((c) => c / (max || 1));
  }, [searchMarks, timestampCount, bins, windowSize]);

  const normalizedDensity = useMemo(() => {
    const max = Math.max(...(timestampDensity ?? [1]));
    const normalized = (timestampDensity ?? new Array(bins).fill(0)).map((v) => v / (max || 1));

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
        height={height}>
        {densityData.map((val, idx) => (
        <HeatBarSegment key={idx} intensity={val} densityFactor={normalizedDensity[idx]} />
        ))}
    </StyledTrack>
  );
};