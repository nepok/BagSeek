import React, { createContext, useContext, useCallback, useState, ReactNode } from 'react';

export interface ExportPreselection {
  rosbagPath: string;
  topic?: string;              // single topic (HeatBar "Export section")
  topics?: string[];           // multiple topics (Explore view)
  mcapIds?: [number, number];  // MCAP range (HeatBar "Export section")
  timestampIndex?: number;     // current Explore position (index marker on MCAP range slider)
  searchMarks?: { value: number; rank?: number }[];  // heatmap marks from Explore/Search
}

type ExportPreselectionContextType = {
  exportPreselection: ExportPreselection | null;
  openExportWithPreselection: (data: ExportPreselection) => void;
  openExport: () => void;
  clearPreselection: () => void;
};

const ExportPreselectionContext = createContext<ExportPreselectionContextType | undefined>(undefined);

export const useExportPreselection = () => {
  const ctx = useContext(ExportPreselectionContext);
  if (!ctx) throw new Error('useExportPreselection must be used within ExportPreselectionProvider');
  return ctx;
};

interface ExportPreselectionProviderProps {
  children: ReactNode;
  onOpenExport: () => void;
}

export const ExportPreselectionProvider: React.FC<ExportPreselectionProviderProps> = ({
  children,
  onOpenExport,
}) => {
  const [exportPreselection, setExportPreselection] = useState<ExportPreselection | null>(null);

  const openExportWithPreselection = useCallback((data: ExportPreselection) => {
    setExportPreselection(data);
    onOpenExport();
  }, [onOpenExport]);

  const openExport = useCallback(() => {
    setExportPreselection(null);
    onOpenExport();
  }, [onOpenExport]);

  const clearPreselection = useCallback(() => {
    setExportPreselection(null);
  }, []);

  return (
    <ExportPreselectionContext.Provider
      value={{
        exportPreselection,
        openExportWithPreselection,
        openExport,
        clearPreselection,
      }}
    >
      {children}
    </ExportPreselectionContext.Provider>
  );
};
