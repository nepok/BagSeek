import React, { createContext, useContext, useCallback, useState, ReactNode } from 'react';

export interface ExportPreselection {
  rosbagPath: string;
  topic: string;
  mcapIds: [number, number];
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
