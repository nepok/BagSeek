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
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import CloseIcon from '@mui/icons-material/Close';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SelectAllIcon from '@mui/icons-material/SelectAll';
import McapRangeFilter, { McapRangeFilterItem, formatNsToTime, type McapFilterState } from '../McapRangeFilter/McapRangeFilter';
import { useExportPreselection } from './ExportPreselectionContext';
import { extractRosbagName } from '../../utils/rosbag';
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
  const [selectedRosbagPath, setSelectedRosbagPath] = useState<string>('');
  const [availableTopics, setAvailableTopicsState] = useState<Record<string, string>>({});
  const [timestampCount, setTimestampCount] = useState(0);
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

  const handleClose = () => {
    clearPreselection();
    onClose();
  };

  const topics = Object.keys(availableTopics);
  const topicTypes = availableTopics;
  const allTypes = Array.from(new Set(Object.values(topicTypes)));
  const sortedTopics = useMemo(() => sortTopics(topics, topicTypes), [topics, topicTypes]);

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
        setSelectedRosbagPath(match || paths[0]);
      })
      .catch((err) => console.error('Failed to fetch rosbags:', err));
  }, [isVisible, preSelectPath, exportPreselection?.rosbagPath]);

  // When selected rosbag changes, fetch topics and timestamp summary
  useEffect(() => {
    if (!selectedRosbagPath) {
      setAvailableTopicsState({});
      setTimestampCount(0);
      return;
    }

    const rosbagParam = encodeURIComponent(selectedRosbagPath);

    fetch(`/api/get-topics-for-rosbag?rosbag=${rosbagParam}`)
      .then((res) => res.json())
      .then((data) => {
        const topicsDict = data.topics || {};
        const sorted = sortTopicsObject(topicsDict);
        setAvailableTopicsState(sorted);
      })
      .catch((err) => console.error('Failed to fetch topics:', err));

    fetch(`/api/get-timestamp-summary?rosbag=${rosbagParam}`)
      .then((res) => res.json())
      .then((data) => {
        const count = data.count ?? 0;
        setTimestampCount(count);
      })
      .catch((err) => console.error('Failed to fetch timestamp summary:', err));
  }, [selectedRosbagPath]);

  // Reset custom part when rosbag changes
  useEffect(() => {
    if (selectedRosbagPath) {
      setUserCustomExportPart('');
    }
  }, [selectedRosbagPath]);

  // Compute MCAP range suffix from filter (ids_start_to_end)
  const mcapRangeSuffix = useMemo(() => {
    const filter = selectedRosbagPath ? mcapFilters[selectedRosbagPath] : null;
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
  }, [selectedRosbagPath, mcapFilters]);

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
    if (!selectedRosbagPath) {
      setPendingMcapIds(null);
      return;
    }
    const rosbagName = extractRosbagName(selectedRosbagPath);
    if (exportPreselection && (extractRosbagName(exportPreselection.rosbagPath) === rosbagName || exportPreselection.rosbagPath === selectedRosbagPath)) {
      const [startId, endId] = exportPreselection.mcapIds;
      const ids: string[] = [];
      for (let i = Math.min(startId, endId); i <= Math.max(startId, endId); i++) {
        ids.push(String(i));
      }
      setPendingMcapIds(ids.length > 0 ? { [selectedRosbagPath]: ids } : null);
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
  }, [selectedRosbagPath, exportPreselection]);

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
    const keys = Object.keys(availableTopics);
    if (keys.length === 0) {
      setSelectedTopics([]);
      setSelectedTypes([]);
      return;
    }
    if (exportPreselection && keys.includes(exportPreselection.topic)) {
      setSelectedTopics([exportPreselection.topic]);
      setSelectedTypes([availableTopics[exportPreselection.topic] ?? '']);
      return;
    }
    setSelectedTopics(keys);
    setSelectedTypes(Array.from(new Set(Object.values(availableTopics))));
  }, [availableTopics, exportPreselection]);

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

  // Use props when provided (Explore context)
  const effectiveTopics =
    availableTopicsProp && Object.keys(availableTopicsProp).length > 0
      ? availableTopicsProp
      : availableTopics;
  const effectiveTimestampCount =
    timestampCountProp !== undefined && timestampCountProp > 0
      ? timestampCountProp
      : timestampCount;

  // Per-part data when multiple MCAP windows: one export per window
  interface PartData {
    startIndex: number;
    endIndex: number;
    mcapRanges: [number, number][];
    mcapRangeSuffix: string | null;
  }
  const { partCount, partsData } = useMemo(() => {
    const filter = selectedRosbagPath ? mcapFilters[selectedRosbagPath] : null;
    const useMcapFilter = !!(filter && filter.windows?.length && filter.ranges?.length);
    if (!useMcapFilter || !filter) {
      return { partCount: 1, partsData: [] as PartData[] };
    }
    const { ranges, windows } = filter;
    const maxIdx = ranges.length - 1;
    if (windows.length <= 1) {
      return { partCount: 1, partsData: [] as PartData[] };
    }
    const data: PartData[] = [];
    for (const [wStart, wEnd] of windows) {
      const s = Math.max(0, Math.min(wStart, maxIdx));
      const e = Math.max(0, Math.min(wEnd, maxIdx));
      const startId = parseInt(ranges[s]?.mcapIdentifier ?? '0', 10);
      const endId = parseInt(ranges[e]?.mcapIdentifier ?? '0', 10);
      const startIndex = ranges[s]?.startIndex ?? 0;
      const lastRange = ranges[e];
      const endIndex = Math.min(
        lastRange ? lastRange.startIndex + (lastRange.count ?? 1) - 1 : 0,
        Math.max(0, effectiveTimestampCount - 1)
      );
      const mcapRangeSuffixVal = startId === endId ? `_id_${startId}` : `_ids_${startId}_to_${endId}`;
      data.push({ startIndex, endIndex, mcapRanges: [[startId, endId]], mcapRangeSuffix: mcapRangeSuffixVal });
    }
    return { partCount: data.length, partsData: data };
  }, [selectedRosbagPath, mcapFilters, effectiveTimestampCount]);

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

  // Build full export name(s): rosbag + mcap range + _Part_N_ (optional) + custom
  const builtExportNames = useMemo(() => {
    const rosbagName = selectedRosbagPath ? extractRosbagName(selectedRosbagPath).replace(/\//g, '_') : '';
    if (partCount <= 1) {
      const parts: string[] = [];
      if (includeRosbagName && rosbagName) parts.push(rosbagName);
      if (includeMcapRange && mcapRangeSuffix) parts.push(mcapRangeSuffix.replace(/^_/, ''));
      if (userCustomExportPart.trim()) parts.push(userCustomExportPart.trim());
      return [parts.join('_')];
    }
    return partsData.map((pd, i) => {
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
  }, [selectedRosbagPath, includeRosbagName, includeMcapRange, includePartNumber, useSameCustomText, userCustomExportPart, userCustomExportParts, partCount, partsData, mcapRangeSuffix]);

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

    const filter = mcapFilters[selectedRosbagPath];
    const useMcapFilter = filter?.windows?.length > 0 && filter?.ranges?.length > 0;
    const isMultiPart = partCount > 1 && partsData.length > 0;

    const doSingleExport = async (
      name: string,
      startIdx: number,
      endIdx: number,
      mcapRangesArg: [number, number][],
      startMcap: string | null,
      endMcap: string | null
    ) => {
      const exportData: Record<string, unknown> = {
        new_rosbag_name: name.trim(),
        topics: selectedTopics,
        start_index: startIdx,
        end_index: endIdx,
        start_mcap_id: startMcap,
        end_mcap_id: endMcap,
        source_rosbag: selectedRosbagPath,
      };
      if (mcapRangesArg.length > 0) exportData.mcap_ranges = mcapRangesArg;

      const res = await fetch('/api/export-rosbag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(exportData),
      });
      return res;
    };

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

        const exportItems = exportsPayload.map(({ name, pd }) => ({
          new_rosbag_name: name,
          topics: selectedTopics,
          start_index: pd.startIndex,
          end_index: pd.endIndex,
          start_mcap_id: String(pd.mcapRanges[0]?.[0] ?? '0'),
          end_mcap_id: String(pd.mcapRanges[0]?.[1] ?? '0'),
          mcap_ranges: pd.mcapRanges,
          source_rosbag: selectedRosbagPath,
        }));

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
        // Single export (original logic)
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
            lastRange ? lastRange.startIndex + (lastRange.count ?? 1) - 1 : effectiveTimestampCount - 1,
            effectiveTimestampCount - 1
          );
          startMcapId = ranges[minStartIdx]?.mcapIdentifier ?? null;
          endMcapId = ranges[maxEndIdx]?.mcapIdentifier ?? null;
        } else {
          startIndex = 0;
          endIndex = Math.max(0, effectiveTimestampCount - 1);
          try {
            const summaryRes = await fetch(
              `/api/get-timestamp-summary?rosbag=${encodeURIComponent(selectedRosbagPath)}`
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

        const res = await doSingleExport(
          builtExportName.trim(),
          startIndex,
          endIndex,
          mcapRanges ?? [],
          startMcapId,
          endMcapId
        );
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
              maxWidth: 400,
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
              selectedRosbags={selectedRosbagPath ? [selectedRosbagPath] : []}
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

            {/* Source Rosbag – collapsible: expanded = full list, collapsed = selected rosbag + MCAP ranges only */}
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
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>
                  Source Rosbag
                </Typography>
                <ExpandMoreIcon sx={{ color: 'rgba(255,255,255,0.6)', transform: expandedSourceRosbag ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s', fontSize: 20, flexShrink: 0 }} />
              </Box>
              {expandedSourceRosbag ? (
                <Box sx={{ p: 1 }}>
                    <Box sx={{ py: 0.5, px: 0.5 }}>
                      {availableRosbags.map((path) => {
                        const filter = mcapFilters[path];
                        const mcapCount = filter?.ranges?.length ?? 0;
                        const firstTime = mcapCount > 0 ? formatNsToTime(filter!.ranges[0]?.firstTimestampNs) : null;
                        const lastTime = mcapCount > 1 ? formatNsToTime(filter!.ranges[mcapCount - 1]?.lastTimestampNs) : null;
                        const timeRange = mcapCount === 0 ? null : mcapCount === 1 ? firstTime : lastTime ? `${firstTime}–${lastTime}` : null;
                        const mcapLabel = mcapCount > 0 ? `${mcapCount} MCAP${mcapCount !== 1 ? 's' : ''}` : null;
                        const isSelected = selectedRosbagPath === path;
                        return (
                          <Box key={path}>
                            <Box
                              onClick={() => setSelectedRosbagPath(path)}
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
                                    {mcapCount > 0 && (
                                      <Box sx={{ display: 'flex', gap: 0.5, flexShrink: 0 }}>
                                        {mcapLabel && <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', bgcolor: (t) => `${t.palette.secondary.main}59`, color: 'secondary.main' }}>{mcapLabel}</Box>}
                                        {timeRange && <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', bgcolor: (t) => `${t.palette.warning.main}59`, color: 'warning.main' }}>{timeRange}</Box>}
                                      </Box>
                                    )}
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
                                />
                              </Box>
                            )}
                          </Box>
                        );
                      })}
                    </Box>
                  </Box>
              ) : (
                selectedRosbagPath && (
                  <Box sx={{ p: 1 }}>
                    <Box sx={{ py: 0.5, px: 0.5 }}>
                      {(() => {
                        const path = selectedRosbagPath;
                        const filter = mcapFilters[path];
                        const mcapCount = filter?.ranges?.length ?? 0;
                        const firstTime = mcapCount > 0 ? formatNsToTime(filter!.ranges[0]?.firstTimestampNs) : null;
                        const lastTime = mcapCount > 1 ? formatNsToTime(filter!.ranges[mcapCount - 1]?.lastTimestampNs) : null;
                        const timeRange = mcapCount === 0 ? null : mcapCount === 1 ? firstTime : lastTime ? `${firstTime}–${lastTime}` : null;
                        const mcapLabel = mcapCount > 0 ? `${mcapCount} MCAP${mcapCount !== 1 ? 's' : ''}` : null;
                        return (
                          <Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                              <ListItemText
                                primary={
                                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, minWidth: 0, fontSize: '0.75rem' }}>
                                    <Box component="span" sx={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>{path}</Box>
                                    {mcapCount > 0 && (
                                      <Box sx={{ display: 'flex', gap: 0.5, flexShrink: 0 }}>
                                        {mcapLabel && <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', bgcolor: (t) => `${t.palette.secondary.main}59`, color: 'secondary.main' }}>{mcapLabel}</Box>}
                                        {timeRange && <Box component="span" sx={{ px: 1, py: 0.25, borderRadius: '50px', fontSize: '0.65rem', bgcolor: (t) => `${t.palette.warning.main}59`, color: 'warning.main' }}>{timeRange}</Box>}
                                      </Box>
                                    )}
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
                            />
                          </Box>
                        );
                      })()}
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
                        bgcolor: (t) => `${t.palette.primary.main}59`,
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

            {/* Export name – checkboxes + custom text input(s); multi-part when multiple MCAP ranges */}
            <Box sx={{ border: '1px solid rgba(255,255,255,0.2)', borderRadius: 1, width: DRAWER_WIDTH - 48, maxWidth: '100%', overflow: 'hidden', flexShrink: 0 }}>
              <Box sx={{ px: 1.5, py: 0.75, borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.87)', fontSize: '0.875rem' }}>
                  Export name
                </Typography>
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
                    <Checkbox size="small" checked={includeMcapRange} disabled={!mcapRangeSuffix && partCount <= 1} sx={{ p: 0.5 }} />
                    <Typography variant="body2" sx={{ fontSize: '0.8rem', color: (mcapRangeSuffix || partCount > 1) ? 'rgba(255,255,255,0.87)' : 'rgba(255,255,255,0.5)', cursor: (mcapRangeSuffix || partCount > 1) ? 'pointer' : 'default' }}>
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
                {partCount <= 1 ? (
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
                    {includeRosbagName && selectedRosbagPath && (
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
                        {extractRosbagName(selectedRosbagPath).replace('/', '_')}
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
                        placeholder="Custom name..."
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
                      return (
                      <Box key={i} sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
                        <Box
                          sx={{
                            display: 'flex',
                            flexWrap: 'wrap',
                            alignItems: 'center',
                            gap: 0.5,
                            minHeight: 40,
                            py: 1,
                            px: 1,
                            bgcolor: hasError ? (theme) => alpha(theme.palette.error.main, 0.12) : 'rgba(255,255,255,0.06)',
                            border: '1px solid',
                            borderColor: hasError ? 'error.main' : 'rgba(255,255,255,0.2)',
                            borderRadius: 1,
                            '&:focus-within': {
                              borderColor: hasError ? 'error.main' : 'primary.main',
                              outline: '1px solid',
                            },
                          }}
                        >
                        {includeRosbagName && selectedRosbagPath && (
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
                            {extractRosbagName(selectedRosbagPath).replace('/', '_')}
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
                            placeholder="Custom..."
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
                    );
                    })}
                  </Box>
                )}
              </Box>
            </Box>
          </Box>

          {/* Actions – fixed at bottom */}
          <Box sx={{ flexShrink: 0, display: 'flex', gap: 1, p: 1.5, borderTop: '1px solid rgba(255,255,255,0.12)' }}>
            <Button variant="outlined" onClick={handleClose} fullWidth>
              Cancel
            </Button>
            <Button
              variant="contained"
              onClick={handleExport}
              fullWidth
              disabled={effectiveTimestampCount === 0 || builtExportNames.some((n) => !n.trim()) || duplicateNameIndices.size > 0}
            >
              Export
            </Button>
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default Export;
