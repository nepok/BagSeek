import React from 'react';
import { HeatBar } from '../HeatBar/HeatBar';

interface CustomTrackProps {
  children?: React.ReactNode;
  className?: string;
  timestampCount: number;
  searchMarks: { value: number; rank?: number }[];
  mcapBoundaries?: number[];
  bins?: number;
  windowSize?: number;
  sliderValue?: number;
}

export const CustomTrack: React.FC<CustomTrackProps> = (props) => {
  return (
    <span>
      <HeatBar {...props} />
      {props.children}
    </span>
  );
};