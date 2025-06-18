// CustomTrack.tsx
import React, { useMemo } from 'react';
import { styled } from '@mui/material/styles';

interface CustomTrackProps {
    children?: React.ReactNode;
    className?: string;
    timestampCount: number;
    searchMarks: { value: number }[];
    timestampDensity: number[];
    bins?: number;
    windowSize?: number;
}

function getHeatColor(value: number, densityFactor: number) {
  const hue = (1 - value) * 268; // 209 = Blau, 0 = Rot
  const sat = densityFactor * 30 + value * 70; // dynamic saturation based on data availability
  //const sat = densityFactor * 100;
  return `hsl(${hue}, ${sat}%, 50%)`;
}

const StyledTrack = styled('span')(() => ({
  height: 24,
  width: '100%',
  borderRadius: 4,
  position: 'relative',
  background: 'transparent',
}));

const HeatBar = styled('div')<{ intensity: number; densityFactor: number }>(({ intensity, densityFactor }) => ({
  flex: 1,
  height: '100%',
  background: getHeatColor(intensity, densityFactor),
}));

export const CustomTrack: React.FC<CustomTrackProps> = ({
  children,
  className,
  searchMarks,
  timestampCount,
  bins = 1000,
  windowSize = 50,
  timestampDensity,
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
  }, [searchMarks, timestampCount, timestampDensity, bins, windowSize]);

  const normalizedDensity = useMemo(() => {
    const max = Math.max(...timestampDensity);
    const normalized = timestampDensity.map((v) => v / (max || 1));

    const resized = new Array(bins).fill(0);
    const factor = timestampDensity.length / bins;

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
    <span>
      <StyledTrack>
        <div style={{ display: 'flex', width: '100%', height: '100%' }}>
          {densityData.map((val, idx) => (
            <HeatBar key={idx} intensity={val} densityFactor={normalizedDensity[idx]} />
          ))}
        </div>
      </StyledTrack>
      {children}
    </span>
  );
};