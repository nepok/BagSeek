import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export type SearchResultRow = {
  rank: number;
  rosbag: string;
  mcap_identifier: string;
  embedding_path: string;
  similarityScore: number;
  topic: string;
  timestamp: string;
  minuteOfDay: string;
  model: string;
};

interface MarksItem { value: number }
type MarksStructure = Record<string, Record<string, Record<string, { marks: MarksItem[] }>>>;

/** Context only for results + marks (expensive data). Filters use module-level cache. */
export type SearchResultsCacheState = {
  searchResults: SearchResultRow[];
  marksPerTopic: MarksStructure;
  searchDone: boolean;
  confirmedModels: string[];
  confirmedRosbags: string[];
  searchStatus: { progress: number; status: string; message: string };
};

const initialResultsCache: SearchResultsCacheState = {
  searchResults: [],
  marksPerTopic: {},
  searchDone: false,
  confirmedModels: [],
  confirmedRosbags: [],
  searchStatus: { progress: -1, status: 'idle', message: '' },
};

type SearchResultsCacheContextType = {
  cache: SearchResultsCacheState;
  updateCache: (updates: Partial<SearchResultsCacheState>) => void;
  clearCache: () => void;
};

const SearchResultsCacheContext = createContext<SearchResultsCacheContextType | undefined>(undefined);

export const useSearchResultsCache = () => {
  const ctx = useContext(SearchResultsCacheContext);
  if (!ctx) throw new Error('useSearchResultsCache must be used within SearchResultsCacheProvider');
  return ctx;
};

export const SearchResultsCacheProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [cache, setCache] = useState<SearchResultsCacheState>(initialResultsCache);

  const updateCache = useCallback((updates: Partial<SearchResultsCacheState>) => {
    setCache((prev) => ({ ...prev, ...updates }));
  }, []);

  const clearCache = useCallback(() => {
    setCache(initialResultsCache);
  }, []);

  return (
    <SearchResultsCacheContext.Provider value={{ cache, updateCache, clearCache }}>
      {children}
    </SearchResultsCacheContext.Provider>
  );
};
