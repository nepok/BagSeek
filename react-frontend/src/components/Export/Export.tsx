import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Button,
  Box,
  Typography,
  LinearProgress,
  Collapse,
  IconButton,
  ButtonGroup,
  Checkbox,
  ListItemText,
  Divider,
  TextField,
  CircularProgress,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SelectAllIcon from '@mui/icons-material/SelectAll';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import McapRangeFilter, { McapRangeFilterItem, formatNsToTime, type McapFilterState } from '../McapRangeFilter/McapRangeFilter';
import { useExportPreselection } from './ExportPreselectionContext';
import { extractRosbagName } from '../../utils/rosbag';
import HelpPopover from '../HelpPopover/HelpPopover';
import { sortTopicsObject, sortTopics } from '../../utils/topics';

interface ExportProps {
  timestampCount?: number;
  availableTopics?: Record<string, string>;
  isVisible: boolean;
  onClose: () => void;
  selectedRosbag?: string | null;
  /** Pre-selected rosbag path when opening from different contexts (Explore, Search, MAP) */
  preSelectedRosbag?: string | null;
}

const DRAWER_WIDTH = 560;


const Export: React.FC<ExportProps> = ({
  timestampCount: timestampCountProp,
  availableTopics: availableTopicsProp,
  isVisible,
  onClose,
  selectedRosbag: selectedRosbagProp,
  preSelectedRosbag,
}) => {
  const { exportPreselection, clearPreselection } = useExportPreselection();

  const [availableRosbags, setAvailableRosbags] = useState<string[]>([]);
  const [selectedRosbagPaths, setSelectedRosbagPaths] = useState<string[]>([]);
  const [perRosbagTopics, setPerRosbagTopics] = useState<Record<string, Record<string, string>>>({});
  const [timestampCounts, setTimestampCounts] = useState<Record<string, number>>({});
  const [exportStatus, setExportStatus] = useState<{
    progress: number;
    status: string;
    message?: string;
  } | null>(null);
  const [selectionMode, setSelectionMode] = useState<'topic' | 'type'>('topic');
  const [selectedTopics, setSelectedTopics] = useState<string[]>([]);
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  const [expandedTopics, setExpandedTopics] = useState(true);
  const [expandedSourceRosbag, setExpandedSourceRosbag] = useState(true);
  const [mcapFilters, setMcapFilters] = useState<McapFilterState>({});
  const [loadingMcapRosbags, setLoadingMcapRosbags] = useState<Set<string>>(new Set());
  const [pendingMcapIds, setPendingMcapIds] = useState<Record<string, string[]> | null>(null);
  const [includeRosbagName, setIncludeRosbagName] = useState(true);
  const [includeMcapRange, setIncludeMcapRange] = useState(false);
  const [includePartNumber, setIncludePartNumber] = useState(false);
  const [useSameCustomText, setUseSameCustomText] = useState(false);
  const [userCustomExportPart, setUserCustomExportPart] = useState('');
  const [userCustomExportParts, setUserCustomExportParts] = useState<string[]>([]);

  // Topic preset state
  const [topicPresets, setTopicPresets] = useState<Record<string, string[]>>({});
  const [showTopicPresetSaveField, setShowTopicPresetSaveField] = useState(false);
  const [newTopicPresetName, setNewTopicPresetName] = useState('');
  const [savingTopicPreset, setSavingTopicPreset] = useState(false);
  const [deletingTopicPreset, setDeletingTopicPreset] = useState<string | null>(null);
  const [loadingTopicPresets, setLoadingTopicPresets] = useState(false);

  const handleClose = () => {
    clearPreselection();
    setShowTopicPresetSaveField(false);
    onClose();
  };

  // Merge topics from all selected rosbags
  const mergedTopics = useMemo(() => {
    const merged: Record<string, string> = {};
    for (const path of selectedRosbagPaths) {
      const t = perRosbagTopics[path];
      if (t) Object.assign(merged, t);
    }
    return sortTopicsObject(merged);
  }, [selectedRosbagPaths, perRosbagTopics]);

  // Use props when provided (Explore context), otherwise merged topics
  const effectiveTopics =
    availableTopicsProp && Object.keys(availableTopicsProp).length > 0
      ? availableTopicsProp
      : mergedTopics;

  const topics = Object.keys(effectiveTopics);
  const topicTypes = effectiveTopics;
  const allTypes = Array.from(new Set(Object.values(topicTypes)));
  const sortedTopics = useMemo(() => sortTopics(topics, topicTypes), [topics, topicTypes]);

  // Effective timestamp count: use prop if provided, otherwise check if any selected rosbag has timestamps
  const effectiveTimestampCount =
    timestampCountProp !== undefined && timestampCountProp > 0
      ? timestampCountProp
      : Object.values(timestampCounts).reduce((sum, c) => sum + c, 0);

  // Resolve pre-selection: Export section (HeatBar) first, then props, MAP view, first available
  const mapSelected = isVisible ? (() => {
    try {
      return sessionStorage.getItem('__BagSeekMapSelectedRosbag') || '';
    } catch {
      return '';
    }
  })() : '';
  const preSelectPath = (isVisible && exportPreselection?.rosbagPath) || preSelectedRosbag || selectedRosbagProp || mapSelected || '';

  // Fetch available rosbags and pre-select (Export section preselection first, then polygon/MAP)
  useEffect(() => {
    if (!isVisible) return;
    fetch('/api/get-file-paths')
      .then((res) => res.json())
      .then((data) => {
        const paths = data.paths || [];
        setAvailableRosbags(paths);
        if (paths.length === 0) return;
        let match: string | null = null;
        try {
          // Highest priority: Export section (HeatBar "Export section")
          if (exportPreselection?.rosbagPath) {
            const preselNorm = extractRosbagName(exportPreselection.rosbagPath);
            match = paths.find(
              (p: string) => extractRosbagName(p) === preselNorm || p === exportPreselection!.rosbagPath
            ) ?? null;
          }
          if (!match) {
          const mapMcapRaw = sessionStorage.getItem('__BagSeekMapMcapFilter');
          if (mapMcapRaw) {
            const mapMcap = JSON.parse(mapMcapRaw) as Record<string, string[]>;
            const mapKey = Object.keys(mapMcap).find(
              (k) => Array.isArray(mapMcap[k]) && mapMcap[k].length > 0
            );
            if (mapKey) {
              const norm = extractRosbagName(mapKey);
              const mapMatch = paths.find((p: string) => extractRosbagName(p) === norm || p === mapKey);
              if (mapMatch) match = mapMatch;
            }
          }
          if (!match) {
            const filterRaw = sessionStorage.getItem('__BagSeekPositionalFilter');
            const mcapFilterRaw = sessionStorage.getItem('__BagSeekPositionalMcapFilter');
            if (filterRaw && mcapFilterRaw) {
              const filterNames: string[] = JSON.parse(filterRaw);
              const mcapFilter = JSON.parse(mcapFilterRaw) as Record<string, string[]>;
              const filterNorm = new Set(filterNames.map((n) => extractRosbagName(String(n).trim())));
              const mcapKeysWithData = new Set(
                Object.entries(mcapFilter)
                  .filter(([, ids]) => Array.isArray(ids) && ids.length > 0)
                  .map(([k]) => extractRosbagName(k))
              );
              const polygonFilterMatch = paths.find(
                (p: string) =>
                  filterNorm.has(extractRosbagName(p)) && mcapKeysWithData.has(extractRosbagName(p))
              );
              if (polygonFilterMatch) match = polygonFilterMatch;
            }
          }
          }
        } catch {}
        if (!match && preSelectPath) {
          match = paths.find(
            (p: string) =>
              extractRosbagName(p) === extractRosbagName(preSelectPath) || p === preSelectPath
          ) ?? null;
        }
        const initial = match || paths[0];
        setSelectedRosbagPaths(initial ? [initial] : []);
      })
      .catch((err) => console.error('Failed to fetch rosbags:', err));
  }, [isVisible, preSelectPath, exportPreselection?.rosbagPath]);

  // When selected rosbags change, fetch topics and timestamp summary for each
  const fetchedRosbags = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (selectedRosbagPaths.length === 0) {
      setPerRosbagTopics({});
      setTimestampCounts({});
      fetchedRosbags.current.clear();
      return;
    }

    // Clean up deselected rosbags
    const selectedSet = new Set(selectedRosbagPaths);
    setPerRosbagTopics((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const key of Object.keys(next)) {
        if (!selectedSet.has(key)) { delete next[key]; changed = true; }
      }
      return changed ? next : prev;
    });
    setTimestampCounts((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const key of Object.keys(next)) {
        if (!selectedSet.has(key)) { delete next[key]; changed = true; }
      }
      return changed ? next : prev;
    });
    // Clean up fetched tracking
    fetchedRosbags.current.forEach((key) => {
      if (!selectedSet.has(key)) fetchedRosbags.current.delete(key);
    });

    // Fetch for newly selected rosbags
    for (const path of selectedRosbagPaths) {
      if (fetchedRosbags.current.has(path)) continue;
      fetchedRosbags.current.add(path);
      const rosbagParam = encodeURIComponent(path);

      fetch(`/api/get-topics-for-rosbag?rosbag=${rosbagParam}`)
        .then((res) => res.json())
        .then((data) => {
          const topicsDict = data.topics || {};
          const sorted = sortTopicsObject(topicsDict);
          setPerRosbagTopics((prev) => ({ ...prev, [path]: sorted }));
        })
        .catch((err) => console.error('Failed to fetch topics:', err));

      fetch(`/api/get-timestamp-summary?rosbag=${rosbagParam}`)
        .then((res) => res.json())
        .then((data) => {
          const count = data.count ?? 0;
          setTimestampCounts((prev) => ({ ...prev, [path]: count }));
        })
        .catch((err) => console.error('Failed to fetch timestamp summary:', err));
    }
  }, [selectedRosbagPaths]);

  // Reset custom text when selection changes
  const selectionKey = selectedRosbagPaths.join('|');
  useEffect(() => {
    if (selectedRosbagPaths.length > 0) {
      setUserCustomExportPart('');
    }
  }, [selectionKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // Compute MCAP range suffix from filter for single-rosbag-single-window case
  const mcapRangeSuffix = useMemo(() => {
    if (selectedRosbagPaths.length !== 1) return null;
    const filter = mcapFilters[selectedRosbagPaths[0]];
    if (!filter?.ranges?.length || !filter?.windows?.length) return null;
    const { ranges, windows } = filter;
    const maxIdx = ranges.length - 1;
    let minStartIdx = maxIdx;
    let maxEndIdx = 0;
    for (const [wStart, wEnd] of windows) {
      const s = Math.max(0, Math.min(wStart, maxIdx));
      const e = Math.max(0, Math.min(wEnd, maxIdx));
      minStartIdx = Math.min(minStartIdx, s);
      maxEndIdx = Math.max(maxEndIdx, e);
    }
    const startId = String(ranges[minStartIdx]?.mcapIdentifier ?? '0');
    const endId = String(ranges[maxEndIdx]?.mcapIdentifier ?? '0');
    return startId === endId ? `_id_${startId}` : `_ids_${startId}_to_${endId}`;
  }, [selectedRosbagPaths, mcapFilters]);

  const handleExportNameChange = (customPart: string) => {
    setUserCustomExportPart(customPart.replace(/\//g, '_'));
  };
  const handleExportPartChange = (index: number, value: string) => {
    const sanitized = value.replace(/\//g, '_');
    setUserCustomExportParts((prev) =>
      useSameCustomText ? prev.map(() => sanitized) : prev.map((v, i) => (i === index ? sanitized : v))
    );
  };

  // Load pending MCAP IDs: Export section preselection first, then MAP, then Apply-to-Search
  const loadPendingMcapIds = React.useCallback(() => {
    const primaryPath = selectedRosbagPaths[0];
    if (!primaryPath) {
      setPendingMcapIds(null);
      return;
    }
    const rosbagName = extractRosbagName(primaryPath);
    if (exportPreselection?.mcapIds && (extractRosbagName(exportPreselection.rosbagPath) === rosbagName || exportPreselection.rosbagPath === primaryPath)) {
      const [startId, endId] = exportPreselection.mcapIds;
      const ids: string[] = [];
      for (let i = Math.min(startId, endId); i <= Math.max(startId, endId); i++) {
        ids.push(String(i));
      }
      setPendingMcapIds(ids.length > 0 ? { [primaryPath]: ids } : null);
      return;
    }
    const findMatch = (parsed: Record<string, string[]>) => {
      const key = Object.keys(parsed).find(
        (k) => extractRosbagName(k) === rosbagName || k === rosbagName
      );
      return key && Array.isArray(parsed[key]) && parsed[key].length > 0 ? { [key]: parsed[key] } : null;
    };
    try {
      const mapRaw = sessionStorage.getItem('__BagSeekMapMcapFilter');
      if (mapRaw) {
        const mapParsed = JSON.parse(mapRaw) as Record<string, string[]>;
        const match = findMatch(mapParsed);
        if (match) {
          setPendingMcapIds(match);
          return;
        }
      }
      const posRaw = sessionStorage.getItem('__BagSeekPositionalMcapFilter');
      if (posRaw) {
        const posParsed = JSON.parse(posRaw) as Record<string, string[]>;
        const match = findMatch(posParsed);
        if (match) {
          setPendingMcapIds(match);
          return;
        }
      }
      setPendingMcapIds(null);
    } catch {
      setPendingMcapIds(null);
    }
  }, [selectedRosbagPaths, exportPreselection]);

  useEffect(() => {
    loadPendingMcapIds();
  }, [loadPendingMcapIds]);

  // Re-load when polygon filter changes (e.g. Apply to Search, Clear)
  useEffect(() => {
    if (!isVisible) return;
    const handleFilterChange = () => loadPendingMcapIds();
    window.addEventListener('__BagSeekPositionalFilterChanged', handleFilterChange);
    return () => window.removeEventListener('__BagSeekPositionalFilterChanged', handleFilterChange);
  }, [isVisible, loadPendingMcapIds]);

  // When topics load, default to all selected (or preselected topic from Export section)
  useEffect(() => {
    const keys = Object.keys(effectiveTopics);
    if (keys.length === 0) {
      setSelectedTopics([]);
      setSelectedTypes([]);
      return;
    }
    if (exportPreselection?.topic && keys.includes(exportPreselection.topic)) {
      setSelectedTopics([exportPreselection.topic]);
      setSelectedTypes([effectiveTopics[exportPreselection.topic] ?? '']);
      return;
    }
    if (exportPreselection?.topics && exportPreselection.topics.length > 0) {
      const matching = exportPreselection.topics.filter((t) => keys.includes(t));
      if (matching.length > 0) {
        setSelectedTopics(matching);
        setSelectedTypes(Array.from(new Set(matching.map((t) => effectiveTopics[t]))));
        return;
      }
    }
    setSelectedTopics(keys);
    setSelectedTypes(Array.from(new Set(Object.values(effectiveTopics))));
  }, [effectiveTopics, exportPreselection]);

  const handleTopicToggle = (topic: string) => {
    if (topic === 'ALL') {
      const next = selectedTopics.length === topics.length ? [] : [...topics];
      setSelectedTopics(next);
      setSelectedTypes(Array.from(new Set(next.map((t) => topicTypes[t]))));
    } else {
      const next = selectedTopics.includes(topic)
        ? selectedTopics.filter((t) => t !== topic)
        : [...selectedTopics, topic];
      setSelectedTopics(next);
      setSelectedTypes(Array.from(new Set(next.map((t) => topicTypes[t]))));
    }
  };

  const handleTypeToggle = (type: string) => {
    if (type === 'ALL') {
      const next = selectedTypes.length === allTypes.length ? [] : [...allTypes];
      setSelectedTypes(next);
      setSelectedTopics(topics.filter((t) => next.includes(topicTypes[t])));
    } else {
      const next = selectedTypes.includes(type)
        ? selectedTypes.filter((t) => t !== type)
        : [...selectedTypes, type];
      setSelectedTypes(next);
      setSelectedTopics(topics.filter((t) => next.includes(topicTypes[t])));
    }
  };

  // Fetch topic presets when export drawer opens
  useEffect(() => {
    if (!isVisible) {
      setShowTopicPresetSaveField(false);
      return;
    }
    let cancelled = false;
    const fetchPresets = async () => {
      setLoadingTopicPresets(true);
      try {
        const res = await fetch('/api/topic-presets');
        if (res.ok) {
          const data = await res.json();
          if (!cancelled) setTopicPresets(data);
        }
      } catch (err) {
        console.error('Failed to load topic presets:', err);
      } finally {
        if (!cancelled) setLoadingTopicPresets(false);
      }
    };
    fetchPresets();
    return () => { cancelled = true; };
  }, [isVisible]);

  const handleLoadTopicPreset = (presetName: string) => {
    const presetTopics = topicPresets[presetName];
    if (!presetTopics) return;
    const available = presetTopics.filter((t) => topics.includes(t));
    setSelectedTopics(available);
    setSelectedTypes(Array.from(new Set(available.map((t) => topicTypes[t]))));
  };

  const handleSaveTopicPreset = async (name: string) => {
    if (!name.trim() || savingTopicPreset) return;
    setSavingTopicPreset(true);
    try {
      const res = await fetch('/api/save-topic-preset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), topics: selectedTopics }),
      });
      if (res.ok) {
        const listRes = await fetch('/api/topic-presets');
        if (listRes.ok) setTopicPresets(await listRes.json());
        setNewTopicPresetName('');
        setShowTopicPresetSaveField(false);
      }
    } catch (err) {
      console.error('Failed to save topic preset:', err);
    } finally {
      setSavingTopicPreset(false);
    }
  };

  const handleDeleteTopicPreset = async (presetName: string) => {
    if (deletingTopicPreset) return;
    setDeletingTopicPreset(presetName);
    try {
      const res = await fetch('/api/delete-topic-preset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: presetName }),
      });
      if (res.ok) {
        const listRes = await fetch('/api/topic-presets');
        if (listRes.ok) setTopicPresets(await listRes.json());
      }
    } catch (err) {
      console.error('Failed to delete topic preset:', err);
    } finally {
      setDeletingTopicPreset(null);
    }
  };

  // Per-part data: one export per MCAP window per rosbag
  interface PartData {
    rosbagPath: string;
    startIndex: number;
    endIndex: number;
    mcapRanges: [number, number][];
    mcapRangeSuffix: string | null;
  }

  // Helper to compute parts for a single rosbag
  const computePartsForRosbag = (path: string, filter: McapFilterState[string] | undefined, tCount: number): PartData[] => {
    const hasFilter = filter && filter.ranges?.length > 0 && filter.windows?.length > 0;

    if (!hasFilter || !filter) {
      // No MCAP filter — single part covering everything
      return [{
        rosbagPath: path,
        startIndex: 0,
        endIndex: Math.max(0, tCount - 1),
        mcapRanges: [],
        mcapRangeSuffix: null,
      }];
    }

    const { ranges, windows } = filter;
    const maxIdx = ranges.length - 1;

    if (windows.length <= 1) {
      // Single window — compute its range
      const [wStart, wEnd] = windows[0] ?? [0, maxIdx];
      const s = Math.max(0, Math.min(wStart, maxIdx));
      const e = Math.max(0, Math.min(wEnd, maxIdx));
      const startId = parseInt(ranges[s]?.mcapIdentifier ?? '0', 10);
      const endId = parseInt(ranges[e]?.mcapIdentifier ?? '0', 10);
      const startIndex = ranges[s]?.startIndex ?? 0;
      const lastRange = ranges[e];
      const endIndex = Math.min(
        lastRange ? lastRange.startIndex + (lastRange.count ?? 1) - 1 : 0,
        Math.max(0, tCount - 1)
      );
      const suffix = startId === endId ? `_id_${startId}` : `_ids_${startId}_to_${endId}`;
      return [{
        rosbagPath: path,
        startIndex,
        endIndex,
        mcapRanges: [[startId, endId]],
        mcapRangeSuffix: suffix,
      }];
    }

    // Multiple windows — one part per window
    return windows.map(([wStart, wEnd]) => {
      const s = Math.max(0, Math.min(wStart, maxIdx));
      const e = Math.max(0, Math.min(wEnd, maxIdx));
      const startId = parseInt(ranges[s]?.mcapIdentifier ?? '0', 10);
      const endId = parseInt(ranges[e]?.mcapIdentifier ?? '0', 10);
      const startIndex = ranges[s]?.startIndex ?? 0;
      const lastRange = ranges[e];
      const endIndex = Math.min(
        lastRange ? lastRange.startIndex + (lastRange.count ?? 1) - 1 : 0,
        Math.max(0, tCount - 1)
      );
      const suffix = startId === endId ? `_id_${startId}` : `_ids_${startId}_to_${endId}`;
      return { rosbagPath: path, startIndex, endIndex, mcapRanges: [[startId, endId]] as [number, number][], mcapRangeSuffix: suffix };
    });
  };

  const { partCount, partsData, rosbagGroupBoundaries, isSingleRosbagSingleWindow } = useMemo(() => {
    if (selectedRosbagPaths.length === 0) {
      return { partCount: 0, partsData: [] as PartData[], rosbagGroupBoundaries: [] as number[], isSingleRosbagSingleWindow: false };
    }

    const allParts: PartData[] = [];
    const boundaries: number[] = [];

    for (const path of selectedRosbagPaths) {
      boundaries.push(allParts.length);
      const filter = mcapFilters[path];
      const tCount = timestampCounts[path] ?? 0;
      const parts = computePartsForRosbag(path, filter, tCount);
      allParts.push(...parts);
    }

    // Single rosbag with single window = original "single export" mode
    const isSingle = selectedRosbagPaths.length === 1 && allParts.length <= 1;

    return {
      partCount: allParts.length,
      partsData: allParts,
      rosbagGroupBoundaries: boundaries,
      isSingleRosbagSingleWindow: isSingle,
    };
  }, [selectedRosbagPaths, mcapFilters, timestampCounts]);

  // Sync userCustomExportParts length when partCount changes
  useEffect(() => {
    if (partCount <= 1) return;
    setUserCustomExportParts((prev) => {
      if (prev.length === partCount) return prev;
      if (prev.length > partCount) return prev.slice(0, partCount);
      // When growing from 1 to N, seed first slot from userCustomExportPart
      const seed = prev.length === 0 && userCustomExportPart ? userCustomExportPart : '';
      return [...prev, seed, ...Array(partCount - prev.length - 1).fill('')];
    });
  }, [partCount, userCustomExportPart]);

  // Whether any selected rosbag has MCAP filter data
  const anyMcapFilter = selectedRosbagPaths.some((p) => {
    const f = mcapFilters[p];
    return f?.ranges?.length > 0 && f?.windows?.length > 0;
  });

  // Build full export name(s): rosbag + mcap range + _Part_N_ (optional) + custom
  const builtExportNames = useMemo(() => {
    if (isSingleRosbagSingleWindow) {
      const rosbagName = selectedRosbagPaths[0] ? extractRosbagName(selectedRosbagPaths[0]).replace(/\//g, '_') : '';
      const parts: string[] = [];
      if (includeRosbagName && rosbagName) parts.push(rosbagName);
      if (includeMcapRange && mcapRangeSuffix) parts.push(mcapRangeSuffix.replace(/^_/, ''));
      if (userCustomExportPart.trim()) parts.push(userCustomExportPart.trim());
      return [parts.join('_')];
    }
    return partsData.map((pd, i) => {
      const rosbagName = extractRosbagName(pd.rosbagPath).replace(/\//g, '_');
      const p: string[] = [];
      if (includeRosbagName && rosbagName) p.push(rosbagName);
      if (includeMcapRange && pd.mcapRangeSuffix) p.push(pd.mcapRangeSuffix.replace(/^_/, ''));
      const custom = useSameCustomText
        ? (userCustomExportParts[0] ?? '').trim()
        : (userCustomExportParts[i] ?? '').trim();
      if (custom) p.push(custom);
      if (includePartNumber) p.push(`export_Part_${i + 1}`);
      return p.join('_');
    });
  }, [selectedRosbagPaths, includeRosbagName, includeMcapRange, includePartNumber, useSameCustomText, userCustomExportPart, userCustomExportParts, partCount, partsData, mcapRangeSuffix, isSingleRosbagSingleWindow]);

  const builtExportName = builtExportNames[0] ?? '';

  // Indices where the built export name is duplicated (we can't have multiple rosbags with same name)
  const duplicateNameIndices = useMemo(() => {
    if (partCount <= 1) return new Set<number>();
    const seen = new Map<string, number[]>();
    builtExportNames.forEach((name, i) => {
      const key = (name ?? '').trim();
      if (!key) return;
      if (!seen.has(key)) seen.set(key, []);
      seen.get(key)!.push(i);
    });
    const dupes = new Set<number>();
    seen.forEach((indices) => {
      if (indices.length > 1) indices.forEach((idx) => dupes.add(idx));
    });
    return dupes;
  }, [builtExportNames, partCount]);

  // Poll export status
  useEffect(() => {
    let retryCount = 0;
    let interval: ReturnType<typeof setInterval>;

    if (exportStatus?.status === 'running' || exportStatus?.status === 'starting') {
      interval = setInterval(async () => {
        try {
          const response = await fetch('/api/export-status');
          const data = await response.json();
          setExportStatus(data);
          retryCount = 0;
        } catch (err) {
          console.error('Failed to fetch export status:', err);
          retryCount++;
          if (retryCount >= 3) clearInterval(interval);
        }
      }, 1000);
    }

    return () => clearInterval(interval!);
  }, [exportStatus?.status]);

  useEffect(() => {
    if (exportStatus?.status === 'completed' || exportStatus?.status === 'done') {
      const t = setTimeout(() => setExportStatus(null), 3000);
      return () => clearTimeout(t);
    }
  }, [exportStatus]);

  const handleExport = async () => {
    if (effectiveTimestampCount === 0) return;

    const isMultiPart = !isSingleRosbagSingleWindow && partCount > 0 && partsData.length > 0;

    setExportStatus({ progress: -1, status: 'starting', message: 'Starting export...' });
    handleClose();

    try {
      if (isMultiPart) {
        const exportsPayload = partsData
          .map((pd, i) => {
            const name = builtExportNames[i] ?? '';
            return name.trim() ? { name, pd } : null;
          })
          .filter((x): x is { name: string; pd: PartData } => x !== null);

        if (exportsPayload.length === 0) return;

        const exportItems = exportsPayload.map(({ name, pd }) => {
          const item: Record<string, unknown> = {
            new_rosbag_name: name,
            topics: selectedTopics,
            start_index: pd.startIndex,
            end_index: pd.endIndex,
            source_rosbag: pd.rosbagPath,
          };
          if (pd.mcapRanges.length > 0) {
            item.start_mcap_id = String(pd.mcapRanges[0][0]);
            item.end_mcap_id = String(pd.mcapRanges[pd.mcapRanges.length - 1][1]);
            item.mcap_ranges = pd.mcapRanges;
          }
          return item;
        });

        const res = await fetch('/api/export-rosbag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ exports: exportItems }),
        });
        const ct = res.headers.get('content-type');
        let result: any = {};
        if (ct?.includes('application/json')) {
          const text = await res.text();
          if (text) result = JSON.parse(text);
        }
        if (!res.ok) {
          try {
            const st = await fetch('/api/export-status');
            setExportStatus(st.ok ? await st.json() : { status: 'error', progress: -1, message: result?.error || 'Batch export failed' });
          } catch {
            setExportStatus({ status: 'error', progress: -1, message: result?.error || 'Batch export failed' });
          }
          return;
        }
        setExportStatus({
          progress: 1,
          status: 'completed',
          message: `Exported ${exportItems.length} rosbag(s)`,
        });
      } else {
        // Single export (original logic) — one rosbag, one window
        const singlePath = selectedRosbagPaths[0] ?? '';
        const filter = mcapFilters[singlePath];
        const useMcapFilter = filter?.windows?.length > 0 && filter?.ranges?.length > 0;
        const tCount = timestampCounts[singlePath] ?? 0;

        let startIndex: number;
        let endIndex: number;
        let startMcapId: string | null = null;
        let endMcapId: string | null = null;
        let mcapRanges: [number, number][] | undefined;

        if (useMcapFilter) {
          const { ranges, windows } = filter!;
          const maxIdx = ranges.length - 1;
          let minStartIdx = maxIdx;
          let maxEndIdx = 0;
          mcapRanges = [];
          for (const [wStart, wEnd] of windows) {
            const s = Math.max(0, Math.min(wStart, maxIdx));
            const e = Math.max(0, Math.min(wEnd, maxIdx));
            minStartIdx = Math.min(minStartIdx, s);
            maxEndIdx = Math.max(maxEndIdx, e);
            const startId = parseInt(ranges[s]?.mcapIdentifier ?? '0', 10);
            const endId = parseInt(ranges[e]?.mcapIdentifier ?? '0', 10);
            mcapRanges.push([startId, endId]);
          }
          startIndex = ranges[minStartIdx]?.startIndex ?? 0;
          const lastRange = ranges[maxEndIdx];
          endIndex = Math.min(
            lastRange ? lastRange.startIndex + (lastRange.count ?? 1) - 1 : tCount - 1,
            tCount - 1
          );
          startMcapId = ranges[minStartIdx]?.mcapIdentifier ?? null;
          endMcapId = ranges[maxEndIdx]?.mcapIdentifier ?? null;
        } else {
          startIndex = 0;
          endIndex = Math.max(0, tCount - 1);
          try {
            const summaryRes = await fetch(
              `/api/get-timestamp-summary?rosbag=${encodeURIComponent(singlePath)}`
            );
            if (summaryRes.ok) {
              const d = await summaryRes.json();
              const ranges = d.mcapRanges || [];
              const total = d.count ?? 0;
              const findMcap = (idx: number) => {
                for (let i = 0; i < ranges.length; i++) {
                  const next = i < ranges.length - 1 ? ranges[i + 1].startIndex : total;
                  if (idx >= ranges[i].startIndex && idx < next) return ranges[i].mcapIdentifier;
                }
                return null;
              };
              startMcapId = findMcap(startIndex);
              endMcapId = findMcap(endIndex);
            }
          } catch (e) {
            console.error('Export: timestamp summary error', e);
            setExportStatus(null);
            return;
          }
        }

        const exportData: Record<string, unknown> = {
          new_rosbag_name: builtExportName.trim(),
          topics: selectedTopics,
          start_index: startIndex,
          end_index: endIndex,
          start_mcap_id: startMcapId,
          end_mcap_id: endMcapId,
          source_rosbag: singlePath,
        };
        if (mcapRanges && mcapRanges.length > 0) exportData.mcap_ranges = mcapRanges;

        const res = await fetch('/api/export-rosbag', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(exportData),
        });
        const ct = res.headers.get('content-type');
        let result: any = {};
        if (ct?.includes('application/json')) {
          const text = await res.text();
          if (text) result = JSON.parse(text);
        }
        if (!res.ok) {
          try {
            const st = await fetch('/api/export-status');
            if (st.ok) setExportStatus(await st.json());
            else setExportStatus({ status: 'error', progress: -1, message: result?.error || 'Export failed' });
          } catch {
            setExportStatus({ status: 'error', progress: -1, message: result?.error || 'Export failed' });
          }
        } else {
          setExportStatus({
            progress: 1,
            status: 'completed',
            message: result?.message || 'Export completed!',
          });
        }
      }
    } catch (err) {
      try {
        const st = await fetch('/api/export-status');
        if (st.ok) setExportStatus(await st.json());
        else setExportStatus({ status: 'error', progress: -1, message: (err as Error).message });
      } catch {
        setExportStatus({ status: 'error', progress: -1, message: (err as Error).message });
      }
    }
  };

  const handleExportRaw = async () => {
    if (effectiveTimestampCount === 0) return;

    const isMultiPart = !isSingleRosbagSingleWindow && partCount > 0 && partsData.length > 0;

    setExportStatus({ progress: -1, status: 'starting', message: 'Starting raw export...' });
    handleClose();

    try {
      if (isMultiPart) {
        const exportsPayload = partsData
          .map((pd, i) => {
            const name = builtExportNames[i] ?? '';
            return name.trim() ? { name, pd } : null;
          })
          .filter((x): x is { name: string; pd: PartData } => x !== null);

        if (exportsPayload.length === 0) return;

        const exportItems = exportsPayload.map(({ name, pd }) => ({
          new_rosbag_name: name,
          topics: selectedTopics,
          start_index: pd.startIndex,
          end_index: pd.endIndex,
          start_mcap_id: String(pd.mcapRanges[0]?.[0] ?? '0'),
          end_mcap_id: String(pd.mcapRanges[0]?.[1] ?? '0'),
          mcap_ranges: pd.mcapRanges,
          source_rosbag: pd.rosbagPath,
        }));

        const res = await fetch('/api/export-raw', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ exports: exportItems }),
        });
        const ct = res.headers.get('content-type');
        let result: any = {};
        if (ct?.includes('application/json')) {
          const text = await res.text();
          if (text) result = JSON.parse(text);
        }
        if (!res.ok) {
          try {
            const st = await fetch('/api/export-status');
            setExportStatus(st.ok ? await st.json() : { status: 'error', progress: -1, message: result?.error || 'Raw batch export failed' });
          } catch {
            setExportStatus({ status: 'error', progress: -1, message: result?.error || 'Raw batch export failed' });
          }
          return;
        }
        setExportStatus({
          progress: 1,
          status: 'completed',
          message: `Raw exported ${exportItems.length} part(s)`,
        });
      } else {
        const singlePath = selectedRosbagPaths[0] ?? '';
        const filter = mcapFilters[singlePath];
        const useMcapFilter = filter?.windows?.length > 0 && filter?.ranges?.length > 0;
        const tCount = timestampCounts[singlePath] ?? 0;

        let startIndex: number;
        let endIndex: number;
        let startMcapId: string | null = null;
        let endMcapId: string | null = null;
        let mcapRanges: [number, number][] | undefined;

        if (useMcapFilter) {
          const { ranges, windows } = filter!;
          const maxIdx = ranges.length - 1;
          let minStartIdx = maxIdx;
          let maxEndIdx = 0;
          mcapRanges = [];
          for (const [wStart, wEnd] of windows) {
            const s = Math.max(0, Math.min(wStart, maxIdx));
            const e = Math.max(0, Math.min(wEnd, maxIdx));
            minStartIdx = Math.min(minStartIdx, s);
            maxEndIdx = Math.max(maxEndIdx, e);
            const startId = parseInt(ranges[s]?.mcapIdentifier ?? '0', 10);
            const endId = parseInt(ranges[e]?.mcapIdentifier ?? '0', 10);
            mcapRanges.push([startId, endId]);
          }
          startIndex = ranges[minStartIdx]?.startIndex ?? 0;
          const lastRange = ranges[maxEndIdx];
          endIndex = Math.min(
            lastRange ? lastRange.startIndex + (lastRange.count ?? 1) - 1 : tCount - 1,
            tCount - 1
          );
          startMcapId = ranges[minStartIdx]?.mcapIdentifier ?? null;
          endMcapId = ranges[maxEndIdx]?.mcapIdentifier ?? null;
        } else {
          startIndex = 0;
          endIndex = Math.max(0, tCount - 1);
          try {
            const summaryRes = await fetch(
              `/api/get-timestamp-summary?rosbag=${encodeURIComponent(singlePath)}`
            );
            if (summaryRes.ok) {
              const d = await summaryRes.json();
              const ranges = d.mcapRanges || [];
              const total = d.count ?? 0;
              const findMcap = (idx: number) => {
                for (let i = 0; i < ranges.length; i++) {
                  const next = i < ranges.length - 1 ? ranges[i + 1].startIndex : total;
                  if (idx >= ranges[i].startIndex && idx < next) return ranges[i].mcapIdentifier;
                }
                return null;
              };
              startMcapId = findMcap(startIndex);
              endMcapId = findMcap(endIndex);
            }
          } catch (e) {
            console.error('Raw export: timestamp summary error', e);
            setExportStatus(null);
            return;
          }
        }

        const exportData: Record<string, unknown> = {
          new_rosbag_name: builtExportName.trim(),
          topics: selectedTopics,
          start_index: startIndex,
          end_index: endIndex,
          start_mcap_id: startMcapId,
          end_mcap_id: endMcapId,
          source_rosbag: singlePath,
        };
        if (mcapRanges && mcapRanges.length > 0) exportData.mcap_ranges = mcapRanges;

        const res = await fetch('/api/export-raw', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(exportData),
        });
        const ct = res.headers.get('content-type');
        let result: any = {};
        if (ct?.includes('application/json')) {
          const text = await res.text();
          if (text) result = JSON.parse(text);
        }
        if (!res.ok) {
          try {
            const st = await fetch('/api/export-status');
            if (st.ok) setExportStatus(await st.json());
            else setExportStatus({ status: 'error', progress: -1, message: result?.error || 'Raw export failed' });
          } catch {
            setExportStatus({ status: 'error', progress: -1, message: result?.error || 'Raw export failed' });
          }
        } else {
          setExportStatus({
            progress: 1,
            status: 'completed',
            message: result?.message || 'Raw export completed!',
          });
        }
      }
    } catch (err) {
      try {
        const st = await fetch('/api/export-status');
        if (st.ok) setExportStatus(await st.json());
        else setExportStatus({ status: 'error', progress: -1, message: (err as Error).message });
      } catch {
        setExportStatus({ status: 'error', progress: -1, message: (err as Error).message });
      }
    }
  };

  const allTopicsCount = Object.keys(effectiveTopics).length;

  // Invisible when closed - but still render status popup
  if (!isVisible) {
    return (
      <>
        {exportStatus && (
          <Box
            sx={{
              position: 'fixed',
              bottom: 80,
              left: 20,
              bgcolor: exportStatus.status === 'error' ? 'error.main' : '#202020',
              color: 'white',
              p: '8px 16px',
              borderRadius: 2,
              zIndex: 9999,
              boxShadow: 3,
              maxWidth: 500,
            }}
          >
            <Typography variant="body2" fontWeight={exportStatus.status === 'error' ? 'bold' : 'normal'}>
              {exportStatus.status === 'error'
                ? exportStatus.message || 'Export failed!'
                : exportStatus.progress === -1
                  ? exportStatus.message || 'Loading...'
                  : exportStatus.progress === 1 || exportStatus.status === 'completed'
                    ? exportStatus.message || 'Finished!'
                    : exportStatus.message || `${(exportStatus.progress * 100).toFixed(0)}%`}
            </Typography>
            <Box sx={{ width: 200, mt: 1 }}>
              {exportStatus.status === 'error' ? (
                <LinearProgress variant="determinate" value={100} color="error" />
              ) : exportStatus.progress === -1 ? (
                <LinearProgress />
              ) : (
                <LinearProgress variant="determinate" value={exportStatus.progress * 100} />
              )}
            </Box>
          </Box>
        )}
      </>
    );
  }

  // Helper to render rosbag info (MCAP count + time range badges)
  const renderRosbagBadges = (path: string) => {
    const filter = mcapFilters[path];
    const mcapCount = filter?.ranges?.length ?? 0;
    const firstTime = mcapCount > 0 ? formatNsToTime(filter!.ranges[0]?.firstTimestampNs) : null;
    const lastTime = mcapCount > 1 ? formatNsToTime(filter!.ranges[mcapCount - 1]?.lastTimestampNs) : null;
    const timeRange = mcapCount === 0 ? null : mcapCount === 1 ? firstTime : lastTime ? `${firstTime}\u2013${lastTime}` : null;
    const mcapLabel = mcapCount > 0 ? `${mcapCount} MCAP${mcapCount !== 1 ? 's' : ''}` : null;
    if (mcapCount === 0) return null;
    return (
      <Box sx={{ display: 'flex', gap: 0.5, flexShrink: 0 }}>
        {mcapLabel && <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', bgcolor: (t: any) => `${t.palette.secondary.main}59`, color: 'secondary.main' }}>{mcapLabel}</Box>}
        {timeRange && <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', bgcolor: (t: any) => `${t.palette.warning.main}59`, color: 'warning.main' }}>{timeRange}</Box>}
      </Box>
    );
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 1300,
        display: 'flex',
        bgcolor: 'rgba(0,0,0,0.5)',
        alignItems: 'stretch',
        justifyContent: 'flex-end',
      }}
      onClick={handleClose}
    >
      {/* Click-to-close overlay area */}
      <Box sx={{ flex: 1 }} />

      <Box
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column',
          bgcolor: 'background.paper',
          borderLeft: '1px solid rgba(255,255,255,0.12)',
          overflow: 'hidden',
          boxShadow: 4,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top bar */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            py: 1,
            px: 1.5,
            borderBottom: '1px solid rgba(255,255,255,0.08)',
            minHeight: 48,
            flexShrink: 0,
          }}
        >
          <Typography variant="body1" sx={{ color: 'rgba(255,255,255,0.87)', fontWeight: 500 }}>
            Export
          </Typography>
          <IconButton
            size="small"
            onClick={handleClose}
            sx={{ color: 'rgba(255,255,255,0.7)', p: 0.25 }}
            aria-label="Close"
          >
            <CloseIcon sx={{ fontSize: 22 }} />
          </IconButton>
        </Box>

        <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <Box sx={{ overflowY: 'auto', overflowX: 'hidden', flex: 1, minHeight: 0, minWidth: 0, py: 1.5, px: 1.5, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            {/* McapRangeFilter logic: fetch metadata, apply pending from MAP polygon filter */}
            <McapRangeFilter
              selectedRosbags={selectedRosbagPaths}
              mcapFilters={mcapFilters}
              onMcapFiltersChange={setMcapFilters}
              pendingMcapIds={pendingMcapIds}
              onPendingMcapIdsConsumed={() => {
                setPendingMcapIds(null);
                try {
                  sessionStorage.removeItem('__BagSeekPositionalMcapFilter');
                } catch {}
              }}
              rosbagsToPreload={availableRosbags}
              logicOnly
              onLoadingChange={setLoadingMcapRosbags}
            />

            {/* Source Rosbag – collapsible: expanded = full list, collapsed = selected rosbags + MCAP ranges only */}
            <Box sx={{ border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1, flexShrink: 0 }}>
              <Box
                sx={{
                  px: 1.5,
                  py: 0.75,
                  cursor: 'pointer',
                  '&:hover': { bgcolor: 'rgba(255,255,255,0.06)' },
                  borderBottom: expandedSourceRosbag ? '1px solid rgba(255,255,255,0.08)' : 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 1,
                  minWidth: 0,
                }}
                onClick={() => setExpandedSourceRosbag(!expandedSourceRosbag)}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>
                    Source Rosbag{selectedRosbagPaths.length > 1 ? 's' : ''}
                  </Typography>
                  {selectedRosbagPaths.length > 1 && (
                    <Box
                      component="span"
                      sx={{
                        px: 1,
                        py: 0.25,
                        borderRadius: '50px',
                        fontSize: '0.65rem',
                        bgcolor: (t: any) => `${t.palette.primary.main}59`,
                        color: 'primary.main',
                      }}
                    >
                      {selectedRosbagPaths.length} selected
                    </Box>
                  )}
                </Box>
                <ExpandMoreIcon sx={{ color: 'rgba(255,255,255,0.6)', transform: expandedSourceRosbag ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s', fontSize: 20, flexShrink: 0 }} />
              </Box>
              {expandedSourceRosbag ? (
                <Box sx={{ p: 1 }}>
                    <Box sx={{ py: 0.5, px: 0.5 }}>
                      {availableRosbags.map((path) => {
                        const isSelected = selectedRosbagPaths.includes(path);
                        return (
                          <Box key={path}>
                            <Box
                              onClick={() => {
                                setSelectedRosbagPaths((prev) =>
                                  prev.includes(path)
                                    ? prev.filter((p) => p !== path)
                                    : [...prev, path]
                                );
                              }}
                              sx={{
                                display: 'flex',
                                alignItems: 'center',
                                py: 0,
                                cursor: 'pointer',
                                '&:hover': { bgcolor: 'rgba(255,255,255,0.04)' },
                                borderRadius: 0.5,
                                px: 0.5,
                              }}
                            >
                              <Checkbox size="small" checked={isSelected} sx={{ p: 0.5 }} />
                              <ListItemText
                                primary={
                                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, minWidth: 0, fontSize: '0.75rem' }}>
                                    <Box component="span" sx={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>{path}</Box>
                                    {renderRosbagBadges(path)}
                                  </Box>
                                }
                              />
                            </Box>
                            {isSelected && (
                              <Box sx={{ ml: 2.5 }}>
                                <McapRangeFilterItem
                                  rosbag={path}
                                  mcapFilters={mcapFilters}
                                  onMcapFiltersChange={setMcapFilters}
                                  noIndent
                                  isLoading={loadingMcapRosbags.has(path)}
                                  currentTimestampIndex={exportPreselection?.timestampIndex}
                                  searchMarks={exportPreselection?.searchMarks}
                                />
                              </Box>
                            )}
                          </Box>
                        );
                      })}
                    </Box>
                  </Box>
              ) : (
                selectedRosbagPaths.length > 0 && (
                  <Box sx={{ p: 1 }}>
                    <Box sx={{ py: 0.5, px: 0.5 }}>
                      {selectedRosbagPaths.map((path) => (
                        <Box key={path} sx={{ mb: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                            <ListItemText
                              primary={
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, minWidth: 0, fontSize: '0.75rem' }}>
                                  <Box component="span" sx={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>{path}</Box>
                                  {renderRosbagBadges(path)}
                                </Box>
                              }
                            />
                          </Box>
                          <McapRangeFilterItem
                            rosbag={path}
                            mcapFilters={mcapFilters}
                            onMcapFiltersChange={setMcapFilters}
                            noIndent
                            isLoading={loadingMcapRosbags.has(path)}
                            currentTimestampIndex={exportPreselection?.timestampIndex}
                            searchMarks={exportPreselection?.searchMarks}
                          />
                        </Box>
                      ))}
                    </Box>
                  </Box>
                )
              )}
            </Box>

            {/* Topics – selection by topic or type (inline list like Search sidebar) */}
            <Box sx={{ border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1, flexShrink: 0 }}>
              <Box
                sx={{
                  px: 1.5,
                  py: 0.75,
                  borderBottom: expandedTopics ? '1px solid rgba(255,255,255,0.08)' : 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                  '&:hover': { bgcolor: 'rgba(255,255,255,0.06)' },
                }}
                onClick={() => setExpandedTopics(!expandedTopics)}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>
                    Topics
                  </Typography>
                  {allTopicsCount > 0 && (
                    <Box
                      component="span"
                      sx={{
                        px: 1,
                        py: 0.25,
                        borderRadius: '50px',
                        fontSize: '0.65rem',
                        bgcolor: (t: any) => `${t.palette.primary.main}59`,
                        color: 'primary.main',
                      }}
                    >
                      {selectedTopics.length}/{allTopicsCount}
                    </Box>
                  )}
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      selectionMode === 'topic' ? handleTopicToggle('ALL') : handleTypeToggle('ALL');
                    }}
                    title="Toggle all"
                    sx={{
                      color:
                        (selectionMode === 'topic'
                          ? selectedTopics.length === topics.length
                          : selectedTypes.length === allTypes.length) && (topics.length > 0 || allTypes.length > 0)
                        ? 'primary.main'
                        : 'rgba(255,255,255,0.4)',
                      p: 0.25,
                    }}
                  >
                    <SelectAllIcon sx={{ fontSize: 16 }} />
                  </IconButton>
                  <ExpandMoreIcon sx={{ color: 'rgba(255,255,255,0.6)', transform: expandedTopics ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s', fontSize: 20 }} />
                </Box>
              </Box>
              <Collapse in={expandedTopics}>
                <Box sx={{ p: 1 }}>
                  {/* Topic Presets - inline pills */}
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
                    {loadingTopicPresets ? (
                      <CircularProgress size={16} sx={{ m: 0.5 }} />
                    ) : (
                      Object.keys(topicPresets).map((presetName) => (
                          <Box
                            key={presetName}
                            onClick={() => deletingTopicPreset !== presetName && handleLoadTopicPreset(presetName)}
                            sx={{
                              position: 'relative',
                              display: 'inline-flex',
                              alignItems: 'center',
                              px: 1.25,
                              py: 0.25,
                              borderRadius: '50px',
                              fontSize: '0.7rem',
                              bgcolor: '#B49FCC25',
                              color: '#B49FCC',
                              border: '1px solid #B49FCC50',
                              cursor: 'pointer',
                              transition: 'all 0.15s',
                              userSelect: 'none',
                              '&:hover': {
                                bgcolor: '#B49FCC40',
                              },
                              '&:hover .preset-delete': {
                                opacity: 1,
                                width: 14,
                              },
                              ...(deletingTopicPreset === presetName && { opacity: 0.5, pointerEvents: 'none' }),
                            }}
                          >
                            {presetName}
                            <Box
                              className="preset-delete"
                              component="span"
                              onClick={(e: React.MouseEvent) => {
                                e.stopPropagation();
                                handleDeleteTopicPreset(presetName);
                              }}
                              sx={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                ml: 0.5,
                                opacity: 0,
                                width: 0,
                                overflow: 'hidden',
                                transition: 'all 0.15s',
                                borderRadius: '50%',
                                cursor: 'pointer',
                                color: 'rgba(255,255,255,0.6)',
                                '&:hover': { color: '#ff5555' },
                              }}
                            >
                              {deletingTopicPreset === presetName ? (
                                <CircularProgress size={12} color="inherit" />
                              ) : (
                                <DeleteIcon sx={{ fontSize: 14 }} />
                              )}
                            </Box>
                          </Box>
                      ))
                    )}
                    {/* Add preset pill */}
                    {showTopicPresetSaveField ? (
                      <TextField
                        autoFocus
                        size="small"
                        variant="outlined"
                        placeholder="Name..."
                        value={newTopicPresetName}
                        onChange={(e) => setNewTopicPresetName(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSaveTopicPreset(newTopicPresetName)}
                        onBlur={() => {
                          if (!newTopicPresetName.trim()) setShowTopicPresetSaveField(false);
                        }}
                        disabled={savingTopicPreset}
                        InputProps={{
                          endAdornment: savingTopicPreset ? (
                            <CircularProgress size={14} />
                          ) : (
                            <IconButton size="small" onClick={() => handleSaveTopicPreset(newTopicPresetName)} disabled={!newTopicPresetName.trim()} sx={{ p: 0.25 }}>
                              <AddIcon sx={{ fontSize: 16 }} />
                            </IconButton>
                          ),
                        }}
                        sx={{
                          width: 140,
                          '& .MuiOutlinedInput-root': {
                            height: 26,
                            fontSize: '0.7rem',
                            borderRadius: '50px',
                            bgcolor: 'rgba(255,255,255,0.06)',
                            '& fieldset': { borderColor: 'rgba(255,255,255,0.2)' },
                            '&:hover fieldset': { borderColor: 'rgba(255,255,255,0.4)' },
                            '&.Mui-focused fieldset': { borderColor: 'rgba(255,255,255,0.5)' },
                          },
                          input: { color: 'white', px: 1.25, py: 0 },
                        }}
                      />
                    ) : (
                      <Box
                        onClick={() => selectedTopics.length > 0 && setShowTopicPresetSaveField(true)}
                        sx={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          px: 1,
                          py: 0.25,
                          borderRadius: '50px',
                          fontSize: '0.7rem',
                          border: '1px dashed rgba(255,255,255,0.25)',
                          color: 'rgba(255,255,255,0.4)',
                          cursor: selectedTopics.length > 0 ? 'pointer' : 'default',
                          opacity: selectedTopics.length > 0 ? 1 : 0.4,
                          transition: 'all 0.15s',
                          '&:hover': selectedTopics.length > 0 ? {
                            borderColor: 'rgba(255,255,255,0.5)',
                            color: 'rgba(255,255,255,0.7)',
                          } : {},
                        }}
                      >
                        <AddIcon sx={{ fontSize: 16 }} />
                      </Box>
                    )}
                  </Box>
                  <ButtonGroup fullWidth size="small" sx={{ mb: 1 }}>
                    <Button
                      variant={selectionMode === 'topic' ? 'contained' : 'outlined'}
                      onClick={() => setSelectionMode('topic')}
                      sx={{ flex: 1 }}
                    >
                      Topics
                    </Button>
                    <Button
                      variant={selectionMode === 'type' ? 'contained' : 'outlined'}
                      onClick={() => setSelectionMode('type')}
                      sx={{ flex: 1 }}
                    >
                      Types
                    </Button>
                  </ButtonGroup>
                  {selectionMode === 'topic' ? (
                    <Box sx={{ py: 0.5, px: 0.5 }}>
                      {sortedTopics.map((topic) => (
                        <Box
                          key={topic}
                          onClick={() => handleTopicToggle(topic)}
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            py: 0,
                            cursor: 'pointer',
                            '&:hover': { bgcolor: 'rgba(255,255,255,0.04)' },
                            borderRadius: 0.5,
                            px: 0.5,
                          }}
                        >
                          <Checkbox size="small" checked={selectedTopics.includes(topic)} sx={{ p: 0.5 }} />
                          <ListItemText primary={topic} primaryTypographyProps={{ fontSize: '0.75rem', sx: { wordBreak: 'break-all' } }} />
                        </Box>
                      ))}
                    </Box>
                  ) : (
                    <Box sx={{ py: 0.5, px: 0.5 }}>
                      {allTypes.map((type) => (
                        <Box
                          key={type}
                          onClick={() => handleTypeToggle(type)}
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            py: 0,
                            cursor: 'pointer',
                            '&:hover': { bgcolor: 'rgba(255,255,255,0.04)' },
                            borderRadius: 0.5,
                            px: 0.5,
                          }}
                        >
                          <Checkbox size="small" checked={selectedTypes.includes(type)} sx={{ p: 0.5 }} />
                          <ListItemText primary={type} primaryTypographyProps={{ fontSize: '0.75rem', sx: { wordBreak: 'break-all' } }} />
                        </Box>
                      ))}
                    </Box>
                  )}
                </Box>
              </Collapse>
            </Box>

            {/* Export name – checkboxes + custom text input(s); multi-part when multiple MCAP ranges or rosbags */}
            <Box sx={{ border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1, width: DRAWER_WIDTH - 48, maxWidth: '100%', overflow: 'hidden', flexShrink: 0 }}>
              <Box sx={{ px: 1.5, py: 0.75, borderBottom: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>
                  Export name
                </Typography>
                <HelpPopover
                  title="Export filename"
                  content={
                    <Box component="ul" sx={{ m: 0, pl: 2 }}>
                      <Box component="li" sx={{ mb: 0.5 }}>The checkboxes control which parts appear in the exported filename.</Box>
                      <Box component="li" sx={{ mb: 0.5 }}><strong>Rosbag Name</strong> — the source rosbag folder name.</Box>
                      <Box component="li" sx={{ mb: 0.5 }}><strong>MCAP Range</strong> — the start/end MCAP identifiers of the exported slice (only available when a range is set).</Box>
                      <Box component="li" sx={{ mb: 0.5 }}><strong>Part Number</strong> — a sequential index when exporting multiple ranges.</Box>
                      <Box component="li" sx={{ mb: 0.5 }}><strong>Shared custom text</strong> — one free-text suffix applied to all parts at once.</Box>
                      <Box component="li">You can also type an individual suffix into each part's text field below.</Box>
                    </Box>
                  }
                />
              </Box>
              <Box sx={{ p: 1, display: 'flex', flexDirection: 'column', gap: 1, minWidth: 0, width: '100%' }}>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }} onClick={() => setIncludeRosbagName((v) => !v)}>
                    <Checkbox size="small" checked={includeRosbagName} sx={{ p: 0.5 }} />
                    <Typography variant="body2" sx={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.87)', cursor: 'pointer' }}>
                      Rosbag Name
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center' }} onClick={() => setIncludeMcapRange((v) => !v)}>
                    <Checkbox size="small" checked={includeMcapRange} disabled={!anyMcapFilter && isSingleRosbagSingleWindow} sx={{ p: 0.5 }} />
                    <Typography variant="body2" sx={{ fontSize: '0.8rem', color: (anyMcapFilter || !isSingleRosbagSingleWindow) ? 'rgba(255,255,255,0.87)' : 'rgba(255,255,255,0.5)', cursor: (anyMcapFilter || !isSingleRosbagSingleWindow) ? 'pointer' : 'default' }}>
                      MCAP Range
                    </Typography>
                  </Box>
                  {partCount > 1 && (
                    <Box sx={{ display: 'flex', alignItems: 'center' }} onClick={() => setIncludePartNumber((v) => !v)}>
                      <Checkbox size="small" checked={includePartNumber} sx={{ p: 0.5 }} />
                      <Typography variant="body2" sx={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.87)', cursor: 'pointer' }}>
                        Part Number
                      </Typography>
                    </Box>
                  )}
                  {partCount > 1 && (
                    <Box sx={{ display: 'flex', alignItems: 'center' }} onClick={() => setUseSameCustomText((v) => !v)}>
                      <Checkbox size="small" checked={useSameCustomText} sx={{ p: 0.5 }} />
                      <Typography variant="body2" sx={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.87)', cursor: 'pointer' }}>
                        Shared custom text
                      </Typography>
                    </Box>
                  )}
                </Box>
                {isSingleRosbagSingleWindow ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
                    <Box
                      sx={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        alignItems: 'center',
                        gap: 0.5,
                        minHeight: 40,
                        py: 1,
                        px: 1,
                        bgcolor: 'rgba(255,255,255,0.06)',
                        border: '1px solid rgba(255,255,255,0.2)',
                        borderRadius: 1,
                        '&:focus-within': { borderColor: 'primary.main', outline: '1px solid' },
                      }}
                    >
                    {includeRosbagName && selectedRosbagPaths[0] && (
                      <Box
                        sx={{
                          px: 1,
                          py: 0.5,
                          borderRadius: '50px',
                          bgcolor: 'rgba(255,255,255,0.1)',
                          maxWidth: 120,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          fontSize: '0.875rem',
                          color: 'rgba(255,255,255,0.7)',
                        }}
                      >
                        {extractRosbagName(selectedRosbagPaths[0]).replace('/', '_')}
                      </Box>
                    )}
                    {includeMcapRange && mcapRangeSuffix && (
                      <Box
                        sx={{
                          px: 1,
                          py: 0.5,
                          borderRadius: '50px',
                          bgcolor: 'rgba(255,255,255,0.1)',
                          maxWidth: 100,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          fontSize: '0.875rem',
                          color: 'rgba(255,255,255,0.7)',
                        }}
                      >
                        {mcapRangeSuffix.replace(/^_/, '')}
                      </Box>
                    )}
                    <Box sx={{ flex: 1, minWidth: 80, minHeight: 32 }}>
                      <Box
                        component="input"
                        type="text"
                        value={userCustomExportPart}
                        onChange={(e) => handleExportNameChange(e.target.value)}
                        placeholder={(includeRosbagName || includeMcapRange) ? '+ custom name...' : 'Enter custom name...'}
                        sx={{
                          width: '100%',
                          height: '100%',
                          minHeight: 32,
                          boxSizing: 'border-box',
                          border: 'none',
                          background: 'transparent',
                          fontSize: '0.875rem',
                          color: 'rgba(255,255,255,0.87)',
                          outline: 'none',
                          '&::placeholder': { color: 'rgba(255,255,255,0.4)' },
                        }}
                      />
                    </Box>
                  </Box>
                  {builtExportName && (
                    <Typography sx={{ fontSize: '0.65rem', color: 'rgba(255,255,255,0.45)', wordBreak: 'break-all', lineHeight: 1.2, px: 0.5 }}>
                      {builtExportName}
                    </Typography>
                  )}
                </Box>
                ) : (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, minWidth: 0, width: '100%' }}>
                    {partsData.map((pd, i) => {
                      const hasError = duplicateNameIndices.has(i);
                      const isNewRosbagGroup = rosbagGroupBoundaries.includes(i) && i > 0;
                      const isGroupStart = rosbagGroupBoundaries.includes(i);
                      return (
                      <React.Fragment key={i}>
                        {isNewRosbagGroup && (
                          <Divider sx={{ my: 0.5, borderColor: 'rgba(255,255,255,0.2)' }} />
                        )}
                        {isGroupStart && selectedRosbagPaths.length > 1 && (
                          <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.7rem', px: 0.5 }}>
                            {extractRosbagName(pd.rosbagPath)}
                          </Typography>
                        )}
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
                        <Box
                          sx={{
                            display: 'flex',
                            flexWrap: 'wrap',
                            alignItems: 'center',
                            gap: 0.5,
                            minHeight: 40,
                            py: 1,
                            px: 1,
                            bgcolor: hasError ? (theme: any) => alpha(theme.palette.error.main, 0.12) : 'rgba(255,255,255,0.06)',
                            border: '1px solid',
                            borderColor: hasError ? 'error.main' : 'rgba(255,255,255,0.2)',
                            borderRadius: 1,
                            '&:focus-within': {
                              borderColor: hasError ? 'error.main' : 'primary.main',
                              outline: '1px solid',
                            },
                          }}
                        >
                        {includeRosbagName && pd.rosbagPath && (
                          <Box
                            sx={{
                              px: 1,
                              py: 0.5,
                              borderRadius: '50px',
                              bgcolor: 'rgba(255,255,255,0.1)',
                              maxWidth: 120,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                              fontSize: '0.875rem',
                              color: 'rgba(255,255,255,0.7)',
                            }}
                          >
                            {extractRosbagName(pd.rosbagPath).replace('/', '_')}
                          </Box>
                        )}
                        {includeMcapRange && pd.mcapRangeSuffix && (
                          <Box
                            sx={{
                              px: 1,
                              py: 0.5,
                              borderRadius: '50px',
                              bgcolor: 'rgba(255,255,255,0.1)',
                              maxWidth: 100,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                              fontSize: '0.875rem',
                              color: 'rgba(255,255,255,0.7)',
                            }}
                          >
                            {pd.mcapRangeSuffix.replace(/^_/, '')}
                          </Box>
                        )}
                        <Box sx={{ flex: 1, minWidth: 80, minHeight: 32 }}>
                          <Box
                            component="input"
                            type="text"
                            value={userCustomExportParts[i] ?? ''}
                            onChange={(e) => handleExportPartChange(i, e.target.value)}
                            placeholder={(includeRosbagName || includeMcapRange || includePartNumber) ? '+ custom name...' : 'Enter custom name...'}
                            sx={{
                              width: '100%',
                              height: '100%',
                              minHeight: 32,
                              boxSizing: 'border-box',
                              border: 'none',
                              background: 'transparent',
                              fontSize: '0.875rem',
                              color: 'rgba(255,255,255,0.87)',
                              outline: 'none',
                              '&::placeholder': { color: 'rgba(255,255,255,0.4)' },
                            }}
                          />
                        </Box>
                        {includePartNumber && (
                          <Box
                            sx={{
                              px: 1,
                              py: 0.5,
                              borderRadius: '50px',
                              bgcolor: 'rgba(255,255,255,0.1)',
                              fontSize: '0.875rem',
                              color: 'rgba(255,255,255,0.7)',
                              flexShrink: 0,
                            }}
                          >
                            export_Part_{i + 1}
                          </Box>
                        )}
                        </Box>
                        {(builtExportNames[i] ?? '').trim() && (
                          <Typography
                            sx={{
                              fontSize: '0.65rem',
                              color: hasError ? 'error.main' : 'rgba(255,255,255,0.45)',
                              wordBreak: 'break-all',
                              lineHeight: 1.2,
                              px: 0.5,
                            }}
                          >
                            {builtExportNames[i]}
                          </Typography>
                        )}
                      </Box>
                      </React.Fragment>
                    );
                    })}
                  </Box>
                )}
              </Box>
            </Box>
          </Box>

          {/* Actions – fixed at bottom */}
          <Box sx={{ flexShrink: 0, display: 'flex', gap: 1, p: 1.5, borderTop: '1px solid rgba(255,255,255,0.12)' }}>
            <Button variant="outlined" onClick={handleClose} sx={{ minWidth: 80 }}>
              Cancel
            </Button>
            <Button
              variant="contained"
              onClick={handleExport}
              fullWidth
              disabled={selectedRosbagPaths.length === 0 || effectiveTimestampCount === 0 || builtExportNames.some((n) => !n.trim()) || duplicateNameIndices.size > 0}
            >
              Export as Rosbag
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleExportRaw}
              fullWidth
              disabled={selectedRosbagPaths.length === 0 || effectiveTimestampCount === 0 || builtExportNames.some((n) => !n.trim()) || duplicateNameIndices.size > 0}
            >
              Export Raw Data
            </Button>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default Export;
