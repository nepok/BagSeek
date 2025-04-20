// CustomTrack.tsx
import React, { useMemo } from 'react';
import { styled } from '@mui/material/styles';

interface CustomTrackProps {
    children?: React.ReactNode;
    className?: string;
    marks: { value: number }[]; // e.g., searchMarks
    timestampCount: number;
    bins?: number;
    windowSize?: number;
}

function getHeatColor(value: number) {
  const hue = (1 - value) * 209; // 209 = Blau, 0 = Rot
  return `hsl(${hue}, 81%, 60%)`;
}

const StyledTrack = styled('span')(() => ({
  height: 24,
  width: '100%',
  borderRadius: 4,
  position: 'relative',
  background: 'transparent',
}));

const HeatBar = styled('div')<{ intensity: number }>(({ intensity }) => ({
  flex: 1,
  height: '100%',
  background: getHeatColor(intensity),
}));

export const CustomTrack: React.FC<CustomTrackProps> = ({
  children,
  className,
  marks,
  timestampCount,
  bins = 1000,
  windowSize = 50,
}) => {
  const densityData = useMemo(() => {
    const counts = new Array(bins).fill(0);
    const total = timestampCount;
    const markPositions = marks.map((m) => m.value / total);

    for (let i = 0; i < bins; i++) {
      const binCenter = i / bins;
      let score = 0;
      for (const p of markPositions) {
        const dist = Math.abs(p - binCenter);
        if (dist < windowSize / bins) {
          score += 1 - dist * bins / windowSize;
        }
      }
      counts[i] = score;
    }

    const max = Math.max(...counts);
    return counts.map((c) => c / (max || 1));
  }, [marks, timestampCount, bins, windowSize]);

  return (
    <span className={className}>
      <StyledTrack>
        <div style={{ display: 'flex', width: '100%', height: '100%' }}>
          {densityData.map((val, idx) => (
            <HeatBar key={idx} intensity={val} />
          ))}
        </div>
      </StyledTrack>
      {children}
    </span>
  );
};