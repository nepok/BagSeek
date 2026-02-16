import type { McapFilterState } from '../McapRangeFilter/McapRangeFilter';

/** Module-level cache for search filters. Survives tab switch, clears on reload. */
export type SearchFilterCache = {
  search: string;
  models: string[];
  rosbags: string[];
  viewMode: 'images' | 'rosbags';
  timeRange: number[];
  mcapFilters: McapFilterState;
  selectedTopics: string[];
  sampling: number;
  enhancePrompt: boolean;
};

const defaultFilters: SearchFilterCache = {
  search: '',
  models: [],
  rosbags: [],
  viewMode: 'images',
  timeRange: [0, 1439],
  mcapFilters: {},
  selectedTopics: [],
  sampling: 10,
  enhancePrompt: true,
};

export const searchFilterCache: SearchFilterCache = { ...defaultFilters };

export const clearFilterCache = () => {
  Object.assign(searchFilterCache, defaultFilters);
};

/** Read a value from cache, or default if Apply to Search. */
export const getFilter = <K extends keyof SearchFilterCache>(
  key: K,
  applyToSearch: boolean
): SearchFilterCache[K] =>
  applyToSearch ? defaultFilters[key] : (searchFilterCache[key] ?? defaultFilters[key]);
