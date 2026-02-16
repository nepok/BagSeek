import React from 'react';
import { Box } from '@mui/material';

const ONE_MIN_MS = 60_000;
const labelStepFraction = 50;

function nsToMs(nsStr: string | undefined): number | null {
  if (!nsStr) return null;
  try {
    return Number(BigInt(nsStr) / BigInt(1_000_000));
  } catch {
    return null;
  }
}

function formatNsToTime(nsStr: string | undefined): string {
  const ms = nsToMs(nsStr);
  if (ms === null) return '??:??';
  const d = new Date(ms);
  return d.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
}

export interface McapRangeOverview {
  startIndex: number;
  mcapIdentifier: string;
  firstTimestampNs?: string;
  lastTimestampNs?: string;
}

interface HeatBarTimelineOverviewProps {
  ranges: McapRangeOverview[];
  totalCount: number;
  children?: React.ReactNode;
}

/** Timeline overview (ID + TOD) with HeatBar centered between. Renders children in the middle. */
export const HeatBarTimelineOverview: React.FC<HeatBarTimelineOverviewProps> = ({
  ranges,
  totalCount,
  children,
}) => {
  if (ranges.length === 0 || totalCount === 0) return null;

  const maxIdx = ranges.length - 1;
  const firstMs = nsToMs(ranges[0]?.firstTimestampNs);
  const lastMs = nsToMs(ranges[maxIdx]?.lastTimestampNs);

  const idLabelStep = Math.max(1, Math.ceil(ranges.length / labelStepFraction));
  const shouldShowId = (i: number) => i % idLabelStep === 0;

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

  const labelStyle: React.CSSProperties = {
    position: 'absolute',
    transform: 'translateX(-50%)',
    fontSize: 9,
    pointerEvents: 'none',
    whiteSpace: 'nowrap',
  };

  const tickStyle: React.CSSProperties = {
    position: 'absolute',
    left: '50%',
    transform: 'translateX(-50%)',
    width: 1,
    backgroundColor: 'currentColor',
    pointerEvents: 'none',
  };

  return (
    <Box
      sx={{
        position: 'relative',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: 0,
      }}
    >
      {/* Top row: ID label + ID labels */}
      <Box
        sx={{
          position: 'relative',
          height: 18,
          flexShrink: 0,
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: -22,
            width: 20,
            fontSize: 9,
            color: 'rgba(255,255,255,0.9)',
            whiteSpace: 'nowrap',
            textAlign: 'center',
            transform: 'translateX(-11px)',
          }}
        >
          ID
        </div>
        <div style={{ position: 'absolute', top: -3, left: 0, right: 0, height: 0 }}>
          {ranges.map((r, i) => {
            if (!shouldShowId(i)) return null;
            const endIdx = i < ranges.length - 1 ? ranges[i + 1].startIndex : totalCount;
            const centerFrac = totalCount > 0 ? (r.startIndex + endIdx) / 2 / totalCount : i / ranges.length;
            return (
              <div
                key={`id-${i}`}
                style={{
                  ...labelStyle,
                  left: `${centerFrac * 100}%`,
                  color: 'rgba(255,255,255,0.5)',
                  fontWeight: 400,
                }}
              >
                {r.mcapIdentifier}
                <div style={{ ...tickStyle, top: '100%', marginTop: 2, height: 5 }} />
              </div>
            );
          })}
        </div>
      </Box>
      {/* Middle: HeatBar (centered) */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 20 }}>
        {children}
      </Box>
      {/* Bottom row: TOD label + TOD labels */}
      <Box
        sx={{
          position: 'relative',
          height: 18,
          flexShrink: 0,
        }}
      >
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: -22,
            width: 20,
            fontSize: 9,
            color: 'rgba(255,255,255,0.9)',
            whiteSpace: 'nowrap',
            textAlign: 'center',
            transform: 'translateX(-11px)',
          }}
        >
          TOD
        </div>
        <div style={{ position: 'absolute', bottom: 12, left: 0, right: 0, height: 0 }}>
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
                <div style={{ ...tickStyle, bottom: '100%', marginTop: 2, height: 5 }} />
              </div>
            );
          })}
        </div>
      </Box>
    </Box>
  );
};
