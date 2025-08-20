import React from 'react';
import { HeatBar } from '../HeatBar/HeatBar';

interface CustomTrackProps {
  children?: React.ReactNode;
  className?: string;
  timestampCount: number;
  searchMarks: { value: number }[];
  timestampDensity: number[];
  bins?: number;
  windowSize?: number;
}

export const CustomTrack: React.FC<CustomTrackProps> = (props) => {
  return (
    <span>
      <HeatBar {...props} />
      {props.children}
    </span>
  );
};