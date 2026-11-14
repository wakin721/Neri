/* ==========================================================================
   Neri Web Application — MegaDetectorV6 LiteRT Local Processing,
   Species Validation Studio, IndexedDB Persistence & Batch CSV Exporter
   ========================================================================== */
(function () {
  'use strict';

  // ========================================================================
  // 1. CONSTANTS & NERI PALETTES
  // ========================================================================
  const SPECIES_COLOR_PALETTE = [
    '#0078d4', '#d13438', '#00bcf2', '#107c10', '#5c2d91', '#ff8c00',
    '#e3008c', '#00b294', '#40e0d0', '#f7630c', '#ffb900', '#bad80a',
    '#00b4ff', '#e74856', '#ffd700', '#32cd32', '#da70d6', '#4682b4',
    '#8E44AD', '#2ECC71', '#F39C12', '#D35400', '#C0392B', '#16A085'
  ];

  const SEED_COLOR_OPTIONS = [
    { label: '珊瑚红', color: '#FA9D85' },
    { label: '抹茶绿', color: '#94CC84' },
    { label: '薄荷绿', color: '#69D2B0' },
    { label: '晴空蓝', color: '#63D1E6' },
    { label: '矢车菊蓝', color: '#7CB5FB' },
    { label: '长春花蓝', color: '#9AA5FA' },
    { label: '丁香紫', color: '#B29AFB' },
    { label: '紫藤花', color: '#D6A1FB' },
    { label: '樱花粉', color: '#FA98D2' },
    { label: '蜜橘橙', color: '#FBB769' },
    { label: '橄榄黄', color: '#D3D667' }
  ];

  const DEFAULT_EXPORT_COLUMNS = [
    '文件名', '格式', '拍摄日期', '拍摄时间', '工作天数',
    '物种名称', '学名',
    '目名', '目拉丁名', '科名', '科拉丁名', '属名', '属拉丁名',
    '物种类型', '物种数量', '最低置信度', '独立探测首只', '备注'
  ];

  const DEFAULT_QUICK_SPECIES = [
    '灰雁', '大白鹭', '绿头鸭', '赤嘴潜鸭', '赤麻鸭',
    '青头潜鸭', '马', '黑翅长脚鹬', '未知鸟', '岩羊',
    '赤狐', '野猪', '豹猫', '赤麂', '家牛', '人'
  ];

  const DEFAULT_QUANTITY_BUTTONS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '25', '50'
  ];

  const IMAGE_EXTENSIONS = new Set(['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp', 'tiff', 'tif']);
  const VIDEO_EXTENSIONS = new Set(['mp4', 'mov', 'webm', 'avi', 'mkv', 'm4v']);
  const PERSONNEL_NAMES = new Set(['人', '牧民', '人员', 'person', 'human']);

  // Adler-32 deterministic species color matching system/config.py
  function getSpeciesColor(speciesName) {
    if (!speciesName || speciesName === '空') return '#808080';
    const bytes = new TextEncoder().encode(speciesName);
    let a = 1, b = 0;
    for (let i = 0; i < bytes.length; i++) {
      a = (a + bytes[i]) % 65521;
      b = (b + a) % 65521;
    }
    const hash = ((b << 16) | a) >>> 0;
    return SPECIES_COLOR_PALETTE[hash % SPECIES_COLOR_PALETTE.length];
  }

  // ========================================================================
  // 2. TAXONOMY DATABASE (2026《中国生物物种名录》)
  // ========================================================================
  const taxonomyMap = new Map();
  const taxonomyList = [];
  const translateMap = Object.assign({
    animal: '动物',
    person: '人',
    human: '人',
    vehicle: '车辆',
    car: '车辆',
    truck: '车辆',
    cow: '家牛'
  }, (window.NERI_TAXONOMY_RAW && window.NERI_TAXONOMY_RAW.translate) || {});

  function initTaxonomy() {
    const raw = window.NERI_TAXONOMY_RAW;
    if (!raw || !raw.entries) return;
    const { types, orders, families, genera, entries } = raw;
    for (const [cnName, arr] of Object.entries(entries)) {
      const [sci, ti, oi, fi, gi] = arr;
      const ord = orders[oi] || ['', ''];
      const fam = families[fi] || ['', ''];
      const gen = genera[gi] || ['', ''];
      const stype = types[ti] || '';
      const info = {
        中文名: cnName,
        学名: sci || '',
        物种类型: stype,
        目名: ord[0] || '',
        目拉丁名: ord[1] || '',
        科名: fam[0] || '',
        科拉丁名: fam[1] || '',
        属名: gen[0] || '',
        属拉丁名: gen[1] || ''
      };
      taxonomyMap.set(cnName, info);
      taxonomyList.push(info);
    }
  }
  initTaxonomy();

  function lookupSpeciesTaxonomy(speciesName) {
    if (!speciesName) return null;
    const cleaned = speciesName.trim();
    if (PERSONNEL_NAMES.has(cleaned)) {
      return {
        中文名: cleaned, 学名: 'Homo sapiens', 物种类型: '人员',
        目名: '灵长目', 目拉丁名: 'PRIMATES', 科名: '人科', 科拉丁名: 'Hominidae',
        属名: '人属', 属拉丁名: 'Homo'
      };
    }
    if (taxonomyMap.has(cleaned)) return taxonomyMap.get(cleaned);
    if (cleaned === '动物' || cleaned === 'animal') {
      return { 中文名: '动物', 学名: '', 物种类型: '兽/鸟', 目名: '', 目拉丁名: '', 科名: '', 科拉丁名: '', 属名: '', 属拉丁名: '' };
    }
    if (cleaned === '车辆' || cleaned === 'vehicle') {
      return { 中文名: '车辆', 学名: '', 物种类型: '其他', 目名: '', 目拉丁名: '', 科名: '', 科拉丁名: '', 属名: '', 属拉丁名: '' };
    }
    if (cleaned.includes('鸟')) {
      return { 中文名: cleaned, 学名: '', 物种类型: '鸟', 目名: '', 目拉丁名: '', 科名: '', 科拉丁名: '', 属名: '', 属拉丁名: '' };
    }
    if (['马', '家牛', '牛', '羊', '山羊', '绵羊', '家犬', '狗', '家猫', '猪', '骆驼', '牦牛'].includes(cleaned)) {
      return { 中文名: cleaned, 学名: '', 物种类型: '家畜', 目名: '', 目拉丁名: '', 科名: '', 科拉丁名: '', 属名: '', 属拉丁名: '' };
    }
    return null;
  }

  // ========================================================================
  // 3. STATE & SETTINGS MANAGEMENT
  // ========================================================================
  const SETTINGS_STORAGE_KEY = 'neri_web_settings_v1';

  const defaultSettings = {
    themeMode: 'system',
    seedColor: '#FA9D85',
    keepResultsOnClose: true,
    previewCacheLimitMb: 256,
    modelVariant: 'md_v1000.0.0-sorrel-litert',
    inputSize: 640,
    confidence: 0.25,
    iou: 0.45,
    agnosticNms: true,
    detectAnimal: true,
    detectPerson: true,
    detectVehicle: true,
    videoMode: 'fast',
    vidStride: 5,
    minFrameRatio: 0.10,
    independentMinutes: 30,
    autoGroup: true,
    collapseGroups: true,
    burstSize: 3,
    gapSeconds: 30,
    undoSteps: 50,
    autoSortQuickMarks: false,
    quickSpeciesList: [...DEFAULT_QUICK_SPECIES],
    quickSpeciesUsage: {},
    quantityButtons: [...DEFAULT_QUANTITY_BUTTONS],
    exportScopeMode: 'ask',
    emptyExportPolicy: 'keep',
    exportColumns: [...DEFAULT_EXPORT_COLUMNS]
  };

  let settings = loadSettings();

  function loadSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
      if (!raw) return { ...defaultSettings };
      const parsed = JSON.parse(raw);
      return {
        ...defaultSettings,
        ...parsed,
        quickSpeciesList: Array.isArray(parsed.quickSpeciesList) && parsed.quickSpeciesList.length
          ? parsed.quickSpeciesList
          : [...DEFAULT_QUICK_SPECIES],
        exportColumns: Array.isArray(parsed.exportColumns) && parsed.exportColumns.length
          ? parsed.exportColumns
          : [...DEFAULT_EXPORT_COLUMNS]
      };
    } catch (_) {
      return { ...defaultSettings };
    }
  }

  function saveSettings() {
    try {
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    } catch (_) {}
  }

  // Runtime Application State
  const state = {
    activeView: 'start',
    rootFolderName: '',
    selectedSubfolder: '__ALL__',
    items: [], // Array of DetectionItem objects
    itemsByPath: new Map(),
    fileBlobsByPath: new Map(), // Live File objects in current session
    thumbnailsByPath: new Map(), // Cached DataURL thumbnails
    jobs: [],
    activeJobId: null,
    cancelRequestedJobIds: new Set(),

    // Validation Studio State
    studioViewMode: 'split', // 'split' | 'grid'
    studioStatusFilter: 'all',
    studioSearchQuery: '',
    showDetections: true,
    filterSpecies: 'global',
    confidenceSettings: { global: 0.25 },
    buckets: [],
    selectedBucketKey: null,
    visibleRows: [],
    selectedPath: null,
    selectedPaths: new Set(),
    selectionAnchorPath: null,
    expandedGroupSignatures: new Set(),
    pendingSpeciesNames: [],
    pendingItemPath: null,
    markHistory: [], // Undo stack

    // Canvas Stage Interaction State
    zoom: 1,
    panX: 0,
    panY: 0,
    drawBoxMode: false,
    drawingStart: null,
    drawingCurrent: null,
    selectedBoxIndex: -1,
    activeImageElement: null,

    // LiteRT Model State
    customTfliteName: null,
    customTfliteBuffer: null,
    tfliteModelInstance: null,
    tfliteInterpreters: new Map(),
    modelStatus: 'ready'
  };

  // ========================================================================
  // 4. MD3 EXPRESSIVE DYNAMIC THEME ENGINE
  // ========================================================================
  function hexToHsl(hex) {
    const clean = hex.replace('#', '');
    const num = parseInt(clean, 16);
    const r = ((num >> 16) & 255) / 255;
    const g = ((num >> 8) & 255) / 255;
    const b = (num & 255) / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h = 0, s = 0;
    const l = (max + min) / 2;
    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch (max) {
        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
        case g: h = (b - r) / d + 2; break;
        case b: h = (r - g) / d + 4; break;
      }
      h /= 6;
    }
    return { h: Math.round(h * 360), s: Math.round(s * 100), l: Math.round(l * 100) };
  }

  function applyTheme() {
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = settings.themeMode === 'dark' || (settings.themeMode === 'system' && prefersDark);
    document.documentElement.classList.toggle('dark', isDark);

    const { h, s } = hexToHsl(settings.seedColor || '#FA9D85');
    const sat = Math.max(35, Math.min(82, s));
    const root = document.documentElement.style;

    if (!isDark) {
      root.setProperty('--md-primary', `hsl(${h}, ${sat}%, 36%)`);
      root.setProperty('--md-on-primary', `#ffffff`);
      root.setProperty('--md-primary-container', `hsl(${h}, ${Math.min(92, sat + 14)}%, 90%)`);
      root.setProperty('--md-on-primary-container', `hsl(${h}, ${sat}%, 12%)`);
      root.setProperty('--md-secondary', `hsl(${h}, ${Math.round(sat * 0.45)}%, 38%)`);
      root.setProperty('--md-on-secondary', `#ffffff`);
      root.setProperty('--md-secondary-container', `hsl(${h}, ${Math.round(sat * 0.65)}%, 89%)`);
      root.setProperty('--md-on-secondary-container', `hsl(${h}, ${Math.round(sat * 0.6)}%, 14%)`);
      root.setProperty('--md-tertiary-container', `hsl(${(h + 38) % 360}, 65%, 88%)`);
      root.setProperty('--md-on-tertiary-container', `hsl(${(h + 38) % 360}, 65%, 14%)`);
      root.setProperty('--md-surface', `hsl(${h}, 30%, 98.5%)`);
      root.setProperty('--md-on-surface', `hsl(${h}, 22%, 11%)`);
      root.setProperty('--md-surface-variant', `hsl(${h}, 26%, 90%)`);
      root.setProperty('--md-on-surface-variant', `hsl(${h}, 16%, 30%)`);
      root.setProperty('--md-surface-container-lowest', `#ffffff`);
      root.setProperty('--md-surface-container-low', `hsl(${h}, 28%, 96%)`);
      root.setProperty('--md-surface-container', `hsl(${h}, 25%, 94%)`);
      root.setProperty('--md-surface-container-high', `hsl(${h}, 24%, 91%)`);
      root.setProperty('--md-surface-container-highest', `hsl(${h}, 22%, 88%)`);
      root.setProperty('--md-outline', `hsl(${h}, 14%, 48%)`);
      root.setProperty('--md-outline-variant', `hsl(${h}, 20%, 80%)`);
    } else {
      root.setProperty('--md-primary', `hsl(${h}, ${Math.min(92, sat + 12)}%, 78%)`);
      root.setProperty('--md-on-primary', `hsl(${h}, ${sat}%, 16%)`);
      root.setProperty('--md-primary-container', `hsl(${h}, ${sat}%, 27%)`);
      root.setProperty('--md-on-primary-container', `hsl(${h}, ${Math.min(95, sat + 15)}%, 90%)`);
      root.setProperty('--md-secondary', `hsl(${h}, ${Math.round(sat * 0.5)}%, 76%)`);
      root.setProperty('--md-on-secondary', `hsl(${h}, ${Math.round(sat * 0.5)}%, 18%)`);
      root.setProperty('--md-secondary-container', `hsl(${h}, ${Math.round(sat * 0.45)}%, 27%)`);
      root.setProperty('--md-on-secondary-container', `hsl(${h}, ${Math.round(sat * 0.65)}%, 90%)`);
      root.setProperty('--md-tertiary-container', `hsl(${(h + 38) % 360}, 45%, 26%)`);
      root.setProperty('--md-on-tertiary-container', `hsl(${(h + 38) % 360}, 70%, 88%)`);
      root.setProperty('--md-surface', `hsl(${h}, 22%, 8%)`);
      root.setProperty('--md-on-surface', `hsl(${h}, 20%, 90%)`);
      root.setProperty('--md-surface-variant', `hsl(${h}, 16%, 27%)`);
      root.setProperty('--md-on-surface-variant', `hsl(${h}, 18%, 78%)`);
      root.setProperty('--md-surface-container-lowest', `hsl(${h}, 24%, 5.5%)`);
      root.setProperty('--md-surface-container-low', `hsl(${h}, 20%, 11%)`);
      root.setProperty('--md-surface-container', `hsl(${h}, 19%, 13.5%)`);
      root.setProperty('--md-surface-container-high', `hsl(${h}, 18%, 17%)`);
      root.setProperty('--md-surface-container-highest', `hsl(${h}, 17%, 21%)`);
      root.setProperty('--md-outline', `hsl(${h}, 14%, 56%)`);
      root.setProperty('--md-outline-variant', `hsl(${h}, 16%, 27%)`);
    }

    renderSeedPaletteSwatches();
  }

  // ========================================================================
  // 5. INDEXEDDB PERSISTENCE & CACHE ENGINE
  // ========================================================================
  const IDB_NAME = 'neri_web_idb_v1';
  const IDB_VERSION = 1;
  let dbPromise = null;

  function openDatabase() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
      if (!window.indexedDB) {
        resolve(null);
        return;
      }
      const req = indexedDB.open(IDB_NAME, IDB_VERSION);
      req.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains('meta')) {
          db.createObjectStore('meta', { keyPath: 'key' });
        }
        if (!db.objectStoreNames.contains('items')) {
          db.createObjectStore('items', { keyPath: 'path' });
        }
        if (!db.objectStoreNames.contains('thumbnails')) {
          db.createObjectStore('thumbnails', { keyPath: 'path' });
        }
        if (!db.objectStoreNames.contains('jobs')) {
          db.createObjectStore('jobs', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('models')) {
          db.createObjectStore('models', { keyPath: 'id' });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
    return dbPromise;
  }

  async function idbPut(storeName, value) {
    if (!settings.keepResultsOnClose && (storeName === 'items' || storeName === 'thumbnails')) return;
    const db = await openDatabase();
    if (!db) return;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(storeName, 'readwrite');
        tx.objectStore(storeName).put(value);
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
      } catch (_) {
        resolve(false);
      }
    });
  }

  async function idbPutBatch(storeName, values) {
    if (!settings.keepResultsOnClose && (storeName === 'items' || storeName === 'thumbnails')) return;
    const db = await openDatabase();
    if (!db || !values || !values.length) return;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(storeName, 'readwrite');
        const store = tx.objectStore(storeName);
        for (const v of values) store.put(v);
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
      } catch (_) {
        resolve(false);
      }
    });
  }

  async function idbGetAll(storeName) {
    const db = await openDatabase();
    if (!db) return [];
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(storeName, 'readonly');
        const req = tx.objectStore(storeName).getAll();
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => resolve([]);
      } catch (_) {
        resolve([]);
      }
    });
  }

  async function idbClearStore(storeName) {
    const db = await openDatabase();
    if (!db) return;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(storeName, 'readwrite');
        tx.objectStore(storeName).clear();
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
      } catch (_) {
        resolve(false);
      }
    });
  }

  function serializeItemForStorage(item) {
    return {
      path: item.path,
      filename: item.filename,
      subfolder: item.subfolder,
      ext: item.ext,
      isVideo: item.isVideo,
      size: item.size || 0,
      lastModified: item.lastModified || 0,
      dateTaken: item.dateTaken || '',
      timeTaken: item.timeTaken || '',
      timestamp: item.timestamp || 0,
      width: item.width || 0,
      height: item.height || 0,
      processed: Boolean(item.processed),
      detectionBoxes: Array.isArray(item.detectionBoxes) ? item.detectionBoxes : [],
      manualValidated: Boolean(item.manualValidated),
      manualSpecies: item.manualSpecies || '',
      manualCounts: item.manualCounts || '',
      manualTypes: item.manualTypes || '',
      remark: item.remark || '',
      favorite: Boolean(item.favorite),
      totalFramesProcessed: item.totalFramesProcessed || 1
    };
  }

  async function persistItemsBatch(items) {
    if (!settings.keepResultsOnClose) return;
    const records = items.map(serializeItemForStorage);
    await idbPutBatch('items', records);
    await idbPut('meta', {
      key: 'session_meta',
      rootFolderName: state.rootFolderName,
      selectedSubfolder: state.selectedSubfolder,
      updatedAt: Date.now()
    });
    updatePersistenceChip();
  }

  async function persistJobs() {
    const db = await openDatabase();
    if (!db) return;
    await idbClearStore('jobs');
    await idbPutBatch('jobs', state.jobs);
  }

  async function restoreSessionFromIndexedDB() {
    const [metaList, savedItems, savedThumbs, savedJobs, savedModels] = await Promise.all([
      idbGetAll('meta'),
      idbGetAll('items'),
      idbGetAll('thumbnails'),
      idbGetAll('jobs'),
      idbGetAll('models')
    ]);

    // Restore cached LiteRT .tflite model if present
    if (savedModels.length > 0) {
      const latestModel = savedModels[savedModels.length - 1];
      if (latestModel && latestModel.buffer) {
        state.customTfliteName = latestModel.name;
        state.customTfliteBuffer = latestModel.buffer;
        state.customTfliteMeta = inspectTfliteFlatBuffer(latestModel.buffer);
        const metaStr = state.customTfliteMeta ? ` · ${state.customTfliteMeta.summary}` : '';
        const el = document.getElementById('loadedTfliteName');
        if (el) el.textContent = `已从缓存加载模型: ${latestModel.name} (${formatBytes(latestModel.buffer.byteLength)}${metaStr})`;
        tryInitTfliteRuntime(latestModel.buffer);
        updateModelStatusUI();
      }
    }

    for (const t of savedThumbs) {
      if (t && t.path && t.dataUrl) {
        state.thumbnailsByPath.set(t.path, t.dataUrl);
      }
    }

    if (savedJobs.length > 0) {
      state.jobs = savedJobs.map((j) => {
        if (j.state === 'running') {
          return { ...j, state: 'paused', message: '网页关闭时已自动保存进度，点击继续可接着处理' };
        }
        return j;
      });
    }

    if (savedItems.length > 0) {
      const sessionMeta = metaList.find((m) => m.key === 'session_meta');
      state.rootFolderName = (sessionMeta && sessionMeta.rootFolderName) || savedItems[0].path.split('/')[0] || '恢复的文件夹';
      state.selectedSubfolder = (sessionMeta && sessionMeta.selectedSubfolder) || '__ALL__';
      state.items = savedItems.sort(compareMediaItems);
      state.itemsByPath.clear();
      for (const item of state.items) {
        state.itemsByPath.set(item.path, item);
      }

      // Show Session Restore Banner
      const banner = document.getElementById('sessionRestoreBanner');
      const bannerTitle = document.getElementById('restoreBannerTitle');
      const bannerDesc = document.getElementById('restoreBannerDesc');
      if (banner && bannerTitle && bannerDesc) {
        const validatedCount = state.items.filter((i) => i.manualValidated).length;
        const processedCount = state.items.filter((i) => i.processed).length;
        banner.style.display = 'block';
        bannerTitle.textContent = `已自动恢复文件夹「${state.rootFolderName}」的处理与标记结果`;
        bannerDesc.textContent = `共恢复 ${state.items.length} 个媒体文件（已探测 ${processedCount} 项，已人工标记 ${validatedCount} 项）。您可以直接进入“校验标记”继续工作或导出 CSV。`;
      }
    }

    refreshAllUI();
    updateCacheBreakdownUI();
  }

  function updatePersistenceChip() {
    const textEl = document.getElementById('persistenceStatusText');
    const chipEl = document.getElementById('persistenceStatusChip');
    if (!textEl || !chipEl) return;
    if (!settings.keepResultsOnClose) {
      chipEl.className = 'status-chip warning';
      textEl.textContent = '断点保存已关闭';
      return;
    }
    chipEl.className = 'status-chip ready';
    if (state.items.length > 0) {
      textEl.textContent = `已自动保存 ${state.items.length} 项记录`;
    } else {
      textEl.textContent = '自动保存已就绪';
    }
  }

  // ========================================================================
  // 6. EXIF PARSER & THUMBNAIL GENERATOR
  // ========================================================================
  async function extractExifDateFromFile(file) {
    try {
      // Only read first 128KB for EXIF header speed
      const slice = file.slice(0, 131072);
      const buffer = await slice.arrayBuffer();
      const view = new DataView(buffer);
      if (view.byteLength < 12 || view.getUint16(0, false) !== 0xFFD8) {
        return parseFallbackDate(file);
      }
      let offset = 2;
      while (offset + 4 < view.byteLength) {
        const marker = view.getUint16(offset, false);
        offset += 2;
        if (marker === 0xFFE1) {
          const length = view.getUint16(offset, false);
          if (offset + 8 < view.byteLength &&
              view.getUint32(offset + 2, false) === 0x45786966 &&
              view.getUint16(offset + 6, false) === 0x0000) {
            const tiffStart = offset + 8;
            const littleEndian = view.getUint16(tiffStart, false) === 0x4949;
            const firstIfdOffset = view.getUint32(tiffStart + 4, littleEndian);
            const dateStr = readExifIfdDate(view, tiffStart, tiffStart + firstIfdOffset, littleEndian);
            if (dateStr) {
              const parsed = parseExifString(dateStr);
              if (parsed) return parsed;
            }
          }
          offset += length;
        } else if ((marker & 0xFF00) === 0xFF00) {
          offset += view.getUint16(offset, false);
        } else {
          break;
        }
      }
    } catch (_) {}
    return parseFallbackDate(file);
  }

  function readExifIfdDate(view, tiffStart, ifdOffset, littleEndian) {
    if (ifdOffset + 2 > view.byteLength) return null;
    const entries = view.getUint16(ifdOffset, littleEndian);
    let fallbackDate = null;
    let exifSubIfdOffset = null;

    for (let i = 0; i < entries; i++) {
      const entryOffset = ifdOffset + 2 + i * 12;
      if (entryOffset + 12 > view.byteLength) break;
      const tag = view.getUint16(entryOffset, littleEndian);
      const count = view.getUint32(entryOffset + 4, littleEndian);
      const valOffset = tiffStart + view.getUint32(entryOffset + 8, littleEndian);

      if (tag === 0x8769) {
        exifSubIfdOffset = valOffset;
      } else if ((tag === 0x9003 || tag === 0x9004 || tag === 0x0132) && count >= 19 && valOffset + 19 <= view.byteLength) {
        let str = '';
        for (let c = 0; c < 19; c++) {
          str += String.fromCharCode(view.getUint8(valOffset + c));
        }
        if (tag === 0x9003) return str;
        fallbackDate = str;
      }
    }

    if (exifSubIfdOffset && exifSubIfdOffset < view.byteLength) {
      const subDate = readExifIfdDate(view, tiffStart, exifSubIfdOffset, littleEndian);
      if (subDate) return subDate;
    }
    return fallbackDate;
  }

  function parseExifString(str) {
    // Format: "YYYY:MM:DD HH:MM:SS" or "YYYY-MM-DD HH:MM:SS"
    const m = String(str).trim().match(/^(\d{4})[:\-](\d{2})[:\-](\d{2})\s+(\d{2}):(\d{2}):(\d{2})/);
    if (!m) return null;
    const [, y, mo, d, h, mi, s] = m;
    const dateTaken = `${y}-${mo}-${d}`;
    const timeTaken = `${h}:${mi}:${s}`;
    const dt = new Date(`${dateTaken}T${timeTaken}`);
    return {
      dateTaken,
      timeTaken,
      timestamp: isNaN(dt.getTime()) ? Date.now() : dt.getTime()
    };
  }

  function parseFallbackDate(file) {
    // Try camera-trap filename date pattern first, e.g. IMG_20260512_081522.jpg
    const nameMatch = file.name.match(/(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_T\s]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})/);
    if (nameMatch) {
      const [, y, mo, d, h, mi, s] = nameMatch;
      const dateTaken = `${y}-${mo}-${d}`;
      const timeTaken = `${h}:${mi}:${s}`;
      const dt = new Date(`${dateTaken}T${timeTaken}`);
      if (!isNaN(dt.getTime())) {
        return { dateTaken, timeTaken, timestamp: dt.getTime() };
      }
    }
    const ts = file.lastModified || Date.now();
    const dt = new Date(ts);
    const pad = (n) => String(n).padStart(2, '0');
    return {
      dateTaken: `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}`,
      timeTaken: `${pad(dt.getHours())}:${pad(dt.getMinutes())}:${pad(dt.getSeconds())}`,
      timestamp: ts
    };
  }

  // ========================================================================
  // 7. MEGADETECTOR V6 LITERT (.TFLITE) INFERENCE & LOCAL VISION ENGINE
  // ========================================================================
  const BUILTIN_MODEL_FILES = {
    'md_v1000.0.0-sorrel-litert': 'md_v1000.0.0-sorrel.tflite',
    'mdv6-yolov10-c-litert': 'MDV6-yolov10-c.tflite'
  };

  function updateModelStatusUI() {
    const chip = document.getElementById('modelStatusChip');
    const text = document.getElementById('modelStatusText');
    const loadedEl = document.getElementById('loadedTfliteName');
    if (!chip || !text) return;
    chip.className = 'status-chip ready';
    const modelSelect = document.getElementById('selectLiteRtModel');
    const variant = modelSelect ? modelSelect.value : settings.modelVariant;
    if (variant === 'custom-litert' && state.customTfliteName) {
      text.textContent = `LiteRT: ${state.customTfliteName}`;
      if (loadedEl) loadedEl.textContent = `当前使用自定义模型：${state.customTfliteName}`;
    } else {
      const fname = BUILTIN_MODEL_FILES[variant] || 'md_v1000.0.0-sorrel.tflite';
      const label = variant === 'mdv6-yolov10-c-litert'
        ? 'MDV6-yolov10-c LiteRT'
        : 'md_v1000.0.0-sorrel (YOLO11s LiteRT)';
      text.textContent = `${label} · 本地引擎就绪`;
      if (loadedEl && !state.customTfliteName) {
        loadedEl.textContent = `已就绪：models/${fname} (${label})`;
      }
    }
  }

  // Inspect .tflite / LiteRT FlatBuffer header for version, description & input size hints
  function inspectTfliteFlatBuffer(arrayBuffer) {
    try {
      if (!arrayBuffer || arrayBuffer.byteLength < 16) return null;
      const bytes = new Uint8Array(arrayBuffer);
      const magic = String.fromCharCode(bytes[4], bytes[5], bytes[6], bytes[7]);
      const isTflite3 = magic === 'TFL3';
      // Scan ASCII strings in buffer for YOLO/MegaDetector metadata (e.g. imgsz, stride, names)
      const sliceLen = Math.min(bytes.length, 65536);
      const tailStart = Math.max(0, bytes.length - 65536);
      const decoder = new TextDecoder('utf-8', { fatal: false });
      const headText = decoder.decode(bytes.subarray(0, sliceLen));
      const tailText = tailStart > 0 ? decoder.decode(bytes.subarray(tailStart)) : '';
      const combined = headText + '\n' + tailText;
      let detectedInputSize = null;
      const sizeMatch = combined.match(/imgsz['":\s]+\[?(\d{3,4})/i);
      if (sizeMatch) {
        detectedInputSize = parseInt(sizeMatch[1], 10);
      }
      return {
        isTflite3,
        magic: isTflite3 ? 'TFL3 (LiteRT v3)' : 'LiteRT Binary',
        detectedInputSize,
        summary: `${isTflite3 ? 'TFL3 标准格式' : 'LiteRT 模型'}${detectedInputSize ? ` · 输入 ${detectedInputSize}×${detectedInputSize}` : ''}`
      };
    } catch (_) {
      return null;
    }
  }

  async function tryInitTfliteRuntime(arrayBuffer) {
    if (!arrayBuffer) return;
    try {
      if (window.LiteRTRuntime && window.LiteRTRuntime.LiteRTInterpreter) {
        state.tfliteModelInstance = new window.LiteRTRuntime.LiteRTInterpreter(arrayBuffer);
        state.tfliteInterpreters.set('custom-litert', state.tfliteModelInstance);
        return;
      }
      if (window.tflite && typeof window.tflite.loadTFLiteModel === 'function') {
        state.tfliteModelInstance = await window.tflite.loadTFLiteModel(arrayBuffer);
        return;
      }
    } catch (e) {
      console.warn('LiteRT WASM load skipped:', e);
    }
  }

  async function ensureLiteRtModelLoaded(modelVariant) {
    const variant = modelVariant || settings.modelVariant || 'md_v1000.0.0-sorrel-litert';
    if (variant === 'custom-litert' && state.tfliteModelInstance) {
      return state.tfliteModelInstance;
    }
    if (state.tfliteInterpreters.has(variant)) {
      return state.tfliteInterpreters.get(variant);
    }

    const fname = BUILTIN_MODEL_FILES[variant] || 'md_v1000.0.0-sorrel.tflite';
    let buffer = null;

    if (variant === 'custom-litert' && state.customTfliteBuffer) {
      buffer = state.customTfliteBuffer;
    } else if (typeof window.getBundledTfliteArrayBuffer === 'function') {
      buffer = window.getBundledTfliteArrayBuffer(fname);
    }

    if (!buffer) {
      const resp = await fetch(`models/${fname}`);
      if (!resp.ok) {
        throw new Error(`无法加载模型文件 models/${fname} (HTTP ${resp.status})`);
      }
      buffer = await resp.arrayBuffer();
    }

    if (!window.LiteRTRuntime || !window.LiteRTRuntime.LiteRTInterpreter) {
      throw new Error('LiteRT 浏览器推理引擎 (litert_runtime.js) 未加载');
    }

    const interp = new window.LiteRTRuntime.LiteRTInterpreter(buffer);
    state.tfliteInterpreters.set(variant, interp);
    state.tfliteModelInstance = interp;
    return interp;
  }

  // IoU calculation for Non-Maximum Suppression
  function computeIoU(boxA, boxB) {
    const xA = Math.max(boxA[0], boxB[0]);
    const yA = Math.max(boxA[1], boxB[1]);
    const xB = Math.min(boxA[2], boxB[2]);
    const yB = Math.min(boxA[3], boxB[3]);
    const interW = Math.max(0, xB - xA);
    const interH = Math.max(0, yB - yA);
    const interArea = interW * interH;
    if (interArea <= 0) return 0;
    const areaA = Math.max(0, boxA[2] - boxA[0]) * Math.max(0, boxA[3] - boxA[1]);
    const areaB = Math.max(0, boxB[2] - boxB[0]) * Math.max(0, boxB[3] - boxB[1]);
    return interArea / (areaA + areaB - interArea + 1e-6);
  }

  function applyNMS(boxes, iouThreshold, agnostic) {
    const sorted = [...boxes].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
    const kept = [];
    for (const candidate of sorted) {
      let keep = true;
      for (const existing of kept) {
        if (!agnostic && existing.species !== candidate.species) continue;
        if (computeIoU(candidate.bbox, existing.bbox) > iouThreshold) {
          keep = false;
          break;
        }
      }
      if (keep) kept.push(candidate);
    }
    return kept;
  }

  async function detectImageElement(imgEl, item, jobOptions) {
    const confThresh = jobOptions.confidence ?? settings.confidence;
    const iouThresh = jobOptions.iou ?? settings.iou;
    const agnostic = jobOptions.agnosticNms ?? settings.agnosticNms;

    // 1. If item has synthetic demo ground-truth boxes attached, run realistic noisy detector simulation + NMS
    if (Array.isArray(item._demoGroundTruthBoxes)) {
      const rawBoxes = item._demoGroundTruthBoxes.map((b) => ({
        species: translateMap[b.rawClass] || b.species || '动物',
        rawClass: b.rawClass || 'animal',
        confidence: Number(b.confidence.toFixed(3)),
        bbox: [...b.bbox]
      })).filter((b) => b.confidence >= 0.05);
      return applyNMS(rawBoxes, iouThresh, agnostic).filter((b) => isClassAllowed(b, jobOptions));
    }

    // 2. Run real LiteRT (.tflite) inference using md_v1000.0.0-sorrel.tflite (YOLO11s) or MDV6-yolov10-c.tflite
    const interp = await ensureLiteRtModelLoaded(jobOptions.modelVariant || settings.modelVariant);
    const boxes = await runRealTfliteInference(imgEl, interp, jobOptions);
    return applyNMS(boxes, iouThresh, agnostic).filter((b) => isClassAllowed(b, jobOptions));
  }

  function isClassAllowed(box, jobOptions) {
    const sp = box.species;
    if ((sp === '动物' || box.rawClass === 'animal') && jobOptions.detectAnimal === false) return false;
    if ((sp === '人' || box.rawClass === 'person') && jobOptions.detectPerson === false) return false;
    if ((sp === '车辆' || box.rawClass === 'vehicle') && jobOptions.detectVehicle === false) return false;
    return true;
  }

  async function runRealTfliteInference(imgEl, tfliteModel, jobOptions) {
    const inDetails = typeof tfliteModel.getInputDetails === 'function' ? tfliteModel.getInputDetails() : null;
    const modelInputSize = (inDetails && inDetails[0] && inDetails[0].shape && inDetails[0].shape[1]) || 640;
    const inputSize = modelInputSize;
    const confThresh = jobOptions.confidence ?? settings.confidence ?? 0.25;

    const canvas = document.createElement('canvas');
    canvas.width = inputSize;
    canvas.height = inputSize;
    const ctx = canvas.getContext('2d');

    const iw = imgEl.naturalWidth || imgEl.width || inputSize;
    const ih = imgEl.naturalHeight || imgEl.height || inputSize;
    const scale = Math.min(inputSize / iw, inputSize / ih);
    const nw = Math.round(iw * scale);
    const nh = Math.round(ih * scale);
    const padX = Math.floor((inputSize - nw) / 2);
    const padY = Math.floor((inputSize - nh) / 2);

    ctx.fillStyle = '#727272';
    ctx.fillRect(0, 0, inputSize, inputSize);
    ctx.drawImage(imgEl, padX, padY, nw, nh);

    const rgba = ctx.getImageData(0, 0, inputSize, inputSize).data;
    const numPixels = inputSize * inputSize;
    const inputF32 = new Float32Array(numPixels * 3);
    const inv255 = 1.0 / 255.0;
    for (let i = 0; i < numPixels; i++) {
      const rOff = i * 4;
      const dOff = i * 3;
      inputF32[dOff] = rgba[rOff] * inv255;
      inputF32[dOff + 1] = rgba[rOff + 1] * inv255;
      inputF32[dOff + 2] = rgba[rOff + 2] * inv255;
    }

    const outTensor = typeof tfliteModel.predictAsync === 'function'
      ? await tfliteModel.predictAsync(inputF32)
      : tfliteModel.predict(inputF32);
    const data = await outTensor.data();
    const shape = outTensor.shape || [1, 7, 8400];

    const mdClasses = ['动物', '人', '车辆'];
    const rawClasses = ['animal', 'person', 'vehicle'];
    const boxes = [];

    // Handle [1, max_det, 6] end2end NMS format: [x1, y1, x2, y2, score, cls]
    if (shape.length === 3 && shape[2] === 6) {
      const numDet = shape[1];
      for (let i = 0; i < numDet; i++) {
        const base = i * 6;
        const score = data[base + 4];
        if (score < confThresh) continue;
        const clsIdx = Math.round(data[base + 5]);
        let x1 = data[base], y1 = data[base + 1], x2 = data[base + 2], y2 = data[base + 3];
        if (x2 > 1.5 || y2 > 1.5) {
          x1 = (x1 - padX) / nw; y1 = (y1 - padY) / nh;
          x2 = (x2 - padX) / nw; y2 = (y2 - padY) / nh;
        } else {
          x1 = (x1 * inputSize - padX) / nw; y1 = (y1 * inputSize - padY) / nh;
          x2 = (x2 * inputSize - padX) / nw; y2 = (y2 * inputSize - padY) / nh;
        }
        boxes.push({
          species: mdClasses[clsIdx] || `类别${clsIdx}`,
          rawClass: rawClasses[clsIdx] || 'animal',
          confidence: Number(score.toFixed(3)),
          bbox: [clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)]
        });
      }
      return boxes;
    }

    // Handle YOLO [1, 4+C, N] or [1, N, 4+C]
    const dim1 = shape[1] || 7;
    const dim2 = shape[2] || 8400;
    const channelsFirst = dim1 < dim2;
    const numAnchors = channelsFirst ? dim2 : dim1;
    const numChannels = channelsFirst ? dim1 : dim2;
    const numClasses = Math.max(1, numChannels - 4);

    for (let i = 0; i < numAnchors; i++) {
      let bestScore = 0;
      let bestCls = 0;
      for (let c = 0; c < numClasses; c++) {
        const score = channelsFirst ? data[(4 + c) * numAnchors + i] : data[i * numChannels + 4 + c];
        if (score > bestScore) {
          bestScore = score;
          bestCls = c;
        }
      }
      if (bestScore < confThresh) continue;
      let cx = channelsFirst ? data[0 * numAnchors + i] : data[i * numChannels + 0];
      let cy = channelsFirst ? data[1 * numAnchors + i] : data[i * numChannels + 1];
      let w = channelsFirst ? data[2 * numAnchors + i] : data[i * numChannels + 2];
      let h = channelsFirst ? data[3 * numAnchors + i] : data[i * numChannels + 3];
      if (cx <= 1.5 && cy <= 1.5 && w <= 1.5 && h <= 1.5) {
        cx *= inputSize; cy *= inputSize; w *= inputSize; h *= inputSize;
      }
      const x1 = ((cx - w / 2) - padX) / nw;
      const y1 = ((cy - h / 2) - padY) / nh;
      const x2 = ((cx + w / 2) - padX) / nw;
      const y2 = ((cy + h / 2) - padY) / nh;
      boxes.push({
        species: mdClasses[bestCls] || '动物',
        rawClass: rawClasses[bestCls] || 'animal',
        confidence: Number(bestScore.toFixed(3)),
        bbox: [clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)]
      });
    }
    return boxes;
  }

  function clamp01(v) {
    return Math.max(0, Math.min(1, Number(v) || 0));
  }

  function generateThumbnailDataUrl(imgEl, maxDim = 360) {
    const iw = imgEl.naturalWidth || imgEl.width || 360;
    const ih = imgEl.naturalHeight || imgEl.height || 240;
    const scale = Math.min(1, maxDim / Math.max(iw, ih));
    const tw = Math.max(1, Math.round(iw * scale));
    const th = Math.max(1, Math.round(ih * scale));
    const c = document.createElement('canvas');
    c.width = tw;
    c.height = th;
    const ctx = c.getContext('2d');
    ctx.drawImage(imgEl, 0, 0, tw, th);
    return c.toDataURL('image/jpeg', 0.74);
  }

  // ========================================================================
  // 8. FOLDER IMPORT & SYNTHETIC DEMO CAMERA TRAP GENERATOR
  // ========================================================================
  function compareMediaItems(a, b) {
    if (a.subfolder !== b.subfolder) return a.subfolder.localeCompare(b.subfolder, 'zh-CN');
    if (a.timestamp && b.timestamp && a.timestamp !== b.timestamp) return a.timestamp - b.timestamp;
    return a.filename.localeCompare(b.filename, 'zh-CN', { numeric: true });
  }

  async function importFileList(fileList, explicitRootName) {
    const files = Array.from(fileList || []);
    const mediaFiles = files.filter((f) => {
      const ext = (f.name.split('.').pop() || '').toLowerCase();
      return IMAGE_EXTENSIONS.has(ext) || VIDEO_EXTENSIONS.has(ext);
    });

    if (mediaFiles.length === 0) {
      showSnackbar('所选文件夹中未找到受支持的图片或视频文件');
      return;
    }

    let rootName = explicitRootName || '';
    if (!rootName && mediaFiles[0].webkitRelativePath) {
      rootName = mediaFiles[0].webkitRelativePath.split('/')[0];
    }
    if (!rootName) rootName = '本地导入文件夹';

    state.rootFolderName = rootName;
    let relinkedCount = 0;
    const newItems = [];

    for (const file of mediaFiles) {
      const relPath = (file.webkitRelativePath || `${rootName}/${file.name}`).replace(/\\/g, '/');
      const parts = relPath.split('/');
      const filename = parts[parts.length - 1];
      const subfolder = parts.length > 2 ? parts.slice(1, -1).join('/') : '根目录';
      const ext = (filename.split('.').pop() || '').toLowerCase();
      const isVideo = VIDEO_EXTENSIONS.has(ext);

      state.fileBlobsByPath.set(relPath, file);

      // Check if item already exists from restored IndexedDB session (preserve recognition & manual marks!)
      const existing = state.itemsByPath.get(relPath);
      if (existing) {
        existing.size = file.size;
        existing.lastModified = file.lastModified;
        relinkedCount++;
        newItems.push(existing);
        continue;
      }

      const exifInfo = isVideo ? parseFallbackDate(file) : await extractExifDateFromFile(file);
      const item = {
        path: relPath,
        filename,
        subfolder,
        ext,
        isVideo,
        size: file.size,
        lastModified: file.lastModified,
        dateTaken: exifInfo.dateTaken,
        timeTaken: exifInfo.timeTaken,
        timestamp: exifInfo.timestamp,
        width: 0,
        height: 0,
        processed: false,
        detectionBoxes: [],
        manualValidated: false,
        manualSpecies: '',
        manualCounts: '',
        manualTypes: '',
        remark: '',
        favorite: false,
        totalFramesProcessed: 1
      };
      state.itemsByPath.set(relPath, item);
      newItems.push(item);
    }

    state.items = newItems.sort(compareMediaItems);
    state.selectedSubfolder = '__ALL__';
    await persistItemsBatch(state.items);

    const dropMain = document.getElementById('dropzoneMainText');
    const dropSub = document.getElementById('dropzoneSubText');
    if (dropMain) dropMain.textContent = `已加载文件夹：${state.rootFolderName}（共 ${state.items.length} 个媒体文件）`;
    if (dropSub) {
      const subfolders = new Set(state.items.map((i) => i.subfolder));
      dropSub.textContent = `包含 ${subfolders.size} 个子目录 · 已自动提取 EXIF 拍摄时间`;
    }

    const banner = document.getElementById('sessionRestoreBanner');
    if (banner && relinkedCount > 0) {
      banner.style.display = 'none';
      showSnackbar(`已关联 ${relinkedCount} 个本地原文件，原有识别与人工标记完整保留`);
    } else {
      showSnackbar(`成功导入 ${state.items.length} 个媒体文件`);
    }

    refreshAllUI();
  }

  // Generate realistic synthetic camera-trap dataset with subfolders, bursts, EXIF timestamps & animals
  async function loadDemoCameraTrapDataset() {
    const demoSpecs = [
      // Subfolder 1: 样线A-01号红外相机 (3-photo burst of 岩羊 + 2-photo burst of 绿头鸭 + 1 empty + 1 person)
      {
        subfolder: '样线A-01号相机',
        filename: 'CT01_20261012_063210_01.jpg',
        dateTaken: '2026-10-12', timeTaken: '06:32:10',
        scene: 'mountain_dawn',
        gtBoxes: [
          { rawClass: 'animal', species: '动物', confidence: 0.93, bbox: [0.24, 0.34, 0.56, 0.78], hintLabel: '岩羊 #1' },
          { rawClass: 'animal', species: '动物', confidence: 0.87, bbox: [0.58, 0.40, 0.82, 0.79], hintLabel: '岩羊 #2' }
        ]
      },
      {
        subfolder: '样线A-01号相机',
        filename: 'CT01_20261012_063213_02.jpg',
        dateTaken: '2026-10-12', timeTaken: '06:32:13',
        scene: 'mountain_dawn',
        gtBoxes: [
          { rawClass: 'animal', species: '动物', confidence: 0.91, bbox: [0.26, 0.35, 0.57, 0.78], hintLabel: '岩羊 #1' },
          { rawClass: 'animal', species: '动物', confidence: 0.85, bbox: [0.60, 0.41, 0.84, 0.79], hintLabel: '岩羊 #2' }
        ]
      },
      {
        subfolder: '样线A-01号相机',
        filename: 'CT01_20261012_063216_03.jpg',
        dateTaken: '2026-10-12', timeTaken: '06:32:16',
        scene: 'mountain_dawn',
        gtBoxes: [
          { rawClass: 'animal', species: '动物', confidence: 0.89, bbox: [0.29, 0.35, 0.59, 0.78], hintLabel: '岩羊 #1' }
        ]
      },
      {
        subfolder: '样线A-01号相机',
        filename: 'CT01_20261013_112005_04.jpg',
        dateTaken: '2026-10-13', timeTaken: '11:20:05',
        scene: 'forest_day',
        gtBoxes: [] // Empty wind trigger
      },
      {
        subfolder: '样线A-01号相机',
        filename: 'CT01_20261014_151040_05.jpg',
        dateTaken: '2026-10-14', timeTaken: '15:10:40',
        scene: 'forest_day',
        gtBoxes: [
          { rawClass: 'person', species: '人', confidence: 0.95, bbox: [0.38, 0.20, 0.62, 0.86], hintLabel: '巡护人员' }
        ]
      },
      // Subfolder 2: 湿地B-02号红外相机 (Waterfowl bursts)
      {
        subfolder: '湿地B-02号相机',
        filename: 'CT02_20261015_074512_01.jpg',
        dateTaken: '2026-10-15', timeTaken: '07:45:12',
        scene: 'wetland',
        gtBoxes: [
          { rawClass: 'animal', species: '动物', confidence: 0.94, bbox: [0.16, 0.44, 0.41, 0.74], hintLabel: '灰雁' },
          { rawClass: 'animal', species: '动物', confidence: 0.88, bbox: [0.46, 0.48, 0.69, 0.76], hintLabel: '灰雁' },
          { rawClass: 'animal', species: '动物', confidence: 0.79, bbox: [0.71, 0.50, 0.90, 0.75], hintLabel: '绿头鸭' }
        ]
      },
      {
        subfolder: '湿地B-02号相机',
        filename: 'CT02_20261015_074515_02.jpg',
        dateTaken: '2026-10-15', timeTaken: '07:45:15',
        scene: 'wetland',
        gtBoxes: [
          { rawClass: 'animal', species: '动物', confidence: 0.92, bbox: [0.18, 0.44, 0.43, 0.74], hintLabel: '灰雁' },
          { rawClass: 'animal', species: '动物', confidence: 0.86, bbox: [0.48, 0.48, 0.71, 0.76], hintLabel: '灰雁' }
        ]
      },
      {
        subfolder: '湿地B-02号相机',
        filename: 'CT02_20261016_211830_03.jpg',
        dateTaken: '2026-10-16', timeTaken: '21:18:30',
        scene: 'night_ir',
        gtBoxes: [
          { rawClass: 'animal', species: '动物', confidence: 0.84, bbox: [0.32, 0.45, 0.68, 0.82], hintLabel: '夜行兽类 (赤狐)' }
        ]
      }
    ];

    state.rootFolderName = '演示红外相机监测站_2026Q4';
    state.selectedSubfolder = '__ALL__';
    state.items = [];
    state.itemsByPath.clear();
    state.thumbnailsByPath.clear();

    for (const spec of demoSpecs) {
      const relPath = `${state.rootFolderName}/${spec.subfolder}/${spec.filename}`;
      const dataUrl = renderDemoCameraTrapImage(spec);
      const dt = new Date(`${spec.dateTaken}T${spec.timeTaken}`);
      const item = {
        path: relPath,
        filename: spec.filename,
        subfolder: spec.subfolder,
        ext: 'jpg',
        isVideo: false,
        size: 184320,
        lastModified: dt.getTime(),
        dateTaken: spec.dateTaken,
        timeTaken: spec.timeTaken,
        timestamp: dt.getTime(),
        width: 960,
        height: 640,
        processed: false,
        detectionBoxes: [],
        _demoGroundTruthBoxes: spec.gtBoxes,
        manualValidated: false,
        manualSpecies: '',
        manualCounts: '',
        manualTypes: '',
        remark: '',
        favorite: false,
        totalFramesProcessed: 1
      };
      state.items.push(item);
      state.itemsByPath.set(relPath, item);
      state.thumbnailsByPath.set(relPath, dataUrl);
      await idbPut('thumbnails', { path: relPath, dataUrl });
    }

    await persistItemsBatch(state.items);
    const dropMain = document.getElementById('dropzoneMainText');
    const dropSub = document.getElementById('dropzoneSubText');
    if (dropMain) dropMain.textContent = `已加载演示文件夹：${state.rootFolderName}（共 ${state.items.length} 张红外相机照片）`;
    if (dropSub) dropSub.textContent = `包含 2 个相机位点子目录 · 点击下方“开始本地处理”即可运行 MegaDetectorV6 LiteRT 探测`;

    refreshAllUI();
    showSnackbar('已加载演示红外相机文件夹，请点击“开始本地处理”运行探测');
  }

  function renderDemoCameraTrapImage(spec) {
    const w = 960, h = 640;
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d');

    // Background gradient by scene
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    if (spec.scene === 'night_ir') {
      grad.addColorStop(0, '#1e2220');
      grad.addColorStop(0.6, '#2d3330');
      grad.addColorStop(1, '#181c1a');
    } else if (spec.scene === 'wetland') {
      grad.addColorStop(0, '#4f778a');
      grad.addColorStop(0.5, '#7297a6');
      grad.addColorStop(0.55, '#3b5d6b');
      grad.addColorStop(1, '#2d493e');
    } else if (spec.scene === 'mountain_dawn') {
      grad.addColorStop(0, '#615463');
      grad.addColorStop(0.45, '#9c7a69');
      grad.addColorStop(1, '#4c4637');
    } else {
      grad.addColorStop(0, '#4a6b4e');
      grad.addColorStop(0.6, '#39543c');
      grad.addColorStop(1, '#273828');
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    // Draw terrain / foliage silhouettes
    ctx.fillStyle = 'rgba(20, 28, 22, 0.35)';
    ctx.beginPath();
    ctx.moveTo(0, h * 0.65);
    ctx.quadraticCurveTo(w * 0.3, h * 0.52, w * 0.65, h * 0.64);
    ctx.quadraticCurveTo(w * 0.85, h * 0.70, w, h * 0.58);
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fill();

    // Draw animal / person silhouettes inside gtBoxes
    for (const box of spec.gtBoxes || []) {
      const [x1, y1, x2, y2] = box.bbox;
      const bx = x1 * w, by = y1 * h, bw = (x2 - x1) * w, bh = (y2 - y1) * h;
      ctx.save();
      ctx.fillStyle = spec.scene === 'night_ir' ? '#d8e0db' : '#dfd3c3';
      ctx.strokeStyle = 'rgba(0,0,0,0.45)';
      ctx.lineWidth = 3;

      if (box.rawClass === 'person') {
        // Person silhouette
        ctx.beginPath();
        ctx.arc(bx + bw * 0.5, by + bh * 0.16, Math.min(bw, bh) * 0.14, 0, Math.PI * 2);
        ctx.roundRect(bx + bw * 0.28, by + bh * 0.31, bw * 0.44, bh * 0.42, 12);
        ctx.roundRect(bx + bw * 0.32, by + bh * 0.70, bw * 0.14, bh * 0.28, 6);
        ctx.roundRect(bx + bw * 0.54, by + bh * 0.70, bw * 0.14, bh * 0.28, 6);
        ctx.fill();
        ctx.stroke();
      } else {
        // Wildlife silhouette (body + head + legs/wings)
        ctx.beginPath();
        ctx.ellipse(bx + bw * 0.48, by + bh * 0.55, bw * 0.36, bh * 0.25, 0, 0, Math.PI * 2);
        ctx.ellipse(bx + bw * 0.78, by + bh * 0.32, bw * 0.14, bh * 0.13, -0.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillRect(bx + bw * 0.24, by + bh * 0.68, bw * 0.07, bh * 0.28);
        ctx.fillRect(bx + bw * 0.62, by + bh * 0.68, bw * 0.07, bh * 0.28);
      }
      ctx.restore();
    }

    // Camera Trap OSD Timestamp Bar at bottom
    ctx.fillStyle = 'rgba(0, 0, 0, 0.72)';
    ctx.fillRect(0, h - 40, w, 40);
    ctx.fillStyle = '#f5d76e';
    ctx.font = 'bold 20px monospace';
    ctx.fillText(`NERI-CAM  ${spec.subfolder}   ${spec.dateTaken} ${spec.timeTaken}   14°C`, 18, h - 13);

    return c.toDataURL('image/jpeg', 0.82);
  }

  // ========================================================================
  // 9. LOCAL BATCH DETECTION JOB RUNNER
  // ========================================================================
  async function startLocalDetectionJob(targetItemsOverride) {
    const targetItems = targetItemsOverride || getScopedItems();
    if (!targetItems || targetItems.length === 0) {
      showSnackbar('请先选择包含图片或视频的文件夹（或点击“加载演示相机文件夹”）');
      return;
    }

    const videoMode = document.getElementById('selectVideoMode').value || settings.videoMode;
    const itemsToProcess = videoMode === 'skip'
      ? targetItems.filter((i) => !i.isVideo)
      : targetItems;

    const jobOptions = {
      modelVariant: document.getElementById('selectLiteRtModel').value,
      inputSize: Number(document.getElementById('selectInputSize').value) || 640,
      videoMode,
      vidStride: Number(document.getElementById('selectVidStride').value) || 5,
      confidence: Number(document.getElementById('startConfSlider').value) || 0.25,
      iou: Number(document.getElementById('startIouSlider').value) || 0.45,
      agnosticNms: settings.agnosticNms,
      detectAnimal: document.getElementById('chkDetectAnimal').checked,
      detectPerson: document.getElementById('chkDetectPerson').checked,
      detectVehicle: document.getElementById('chkDetectVehicle').checked
    };

    // Sync global validation confidence with detection confidence
    state.confidenceSettings.global = jobOptions.confidence;
    const valSlider = document.getElementById('validationConfSlider');
    const valText = document.getElementById('validationConfVal');
    if (valSlider) valSlider.value = String(jobOptions.confidence);
    if (valText) valText.textContent = jobOptions.confidence.toFixed(2);

    const jobId = 'job_' + Date.now();
    const scopeLabel = state.selectedSubfolder === '__ALL__'
      ? state.rootFolderName
      : `${state.rootFolderName} / ${state.selectedSubfolder}`;

    const job = {
      id: jobId,
      inputDir: scopeLabel,
      modelName: state.customTfliteName || jobOptions.modelVariant,
      state: 'running',
      processed: 0,
      total: itemsToProcess.length,
      speed: 0,
      elapsedSeconds: 0,
      remainingSeconds: null,
      message: '正在初始化 MegaDetectorV6 LiteRT 推理管线...',
      options: jobOptions,
      targetPaths: itemsToProcess.map((i) => i.path)
    };

    state.jobs.unshift(job);
    state.activeJobId = jobId;
    renderJobsList();
    await persistJobs();
    await runJobLoop(job);
  }

  async function resumeJob(jobId) {
    const job = state.jobs.find((j) => j.id === jobId);
    if (!job) return;
    state.cancelRequestedJobIds.delete(jobId);
    job.state = 'running';
    job.message = '继续本地探测...';
    state.activeJobId = jobId;
    renderJobsList();
    await persistJobs();
    await runJobLoop(job);
  }

  function cancelJob(jobId) {
    state.cancelRequestedJobIds.add(jobId);
    const job = state.jobs.find((j) => j.id === jobId);
    if (job) {
      job.state = 'paused';
      job.message = '已暂停（随时可继续）';
      renderJobsList();
      persistJobs();
    }
  }

  async function runJobLoop(job) {
    const startTime = performance.now() - (job.elapsedSeconds || 0) * 1000;
    const startIndex = job.processed || 0;
    const paths = job.targetPaths || [];

    for (let idx = startIndex; idx < paths.length; idx++) {
      if (state.cancelRequestedJobIds.has(job.id)) {
        job.state = 'paused';
        job.message = `已暂停 (${job.processed}/${job.total})`;
        renderJobsList();
        await persistJobs();
        return;
      }

      const item = state.itemsByPath.get(paths[idx]);
      if (item) {
        job.message = `正在探测: ${item.filename}`;
        await processSingleMediaItem(item, job.options || settings);
      }

      job.processed = idx + 1;
      const elapsed = Math.max(0.05, (performance.now() - startTime) / 1000);
      job.elapsedSeconds = elapsed;
      job.speed = job.processed / elapsed;
      const remainCount = job.total - job.processed;
      job.remainingSeconds = job.speed > 0 ? remainCount / job.speed : 0;

      renderJobsList();
      updateHeroStats();

      // Persist every 4 items or on completion so accidental tab close loses nothing
      if (job.processed % 4 === 0 || job.processed === job.total) {
        await persistItemsBatch(state.items);
        await persistJobs();
      }

      // Yield to browser UI thread
      await new Promise((r) => setTimeout(r, 25));
    }

    job.state = 'completed';
    const detectedItems = paths.map((p) => state.itemsByPath.get(p)).filter((i) => i && i.detectionBoxes && i.detectionBoxes.length > 0).length;
    job.message = `处理完成 · 检出目标 ${detectedItems}/${job.total} 项`;
    state.activeJobId = null;
    await persistItemsBatch(state.items);
    await persistJobs();
    refreshAllUI();
    showSnackbar(`本地探测完成！共处理 ${job.total} 个文件，检出目标 ${detectedItems} 项`);
  }

  async function processSingleMediaItem(item, jobOptions) {
    try {
      if (item.isVideo) {
        await processSingleVideoItem(item, jobOptions);
      } else {
        const imgEl = await loadMediaImageElement(item);
        if (imgEl) {
          item.width = imgEl.naturalWidth || imgEl.width || 960;
          item.height = imgEl.naturalHeight || imgEl.height || 640;
          const boxes = await detectImageElement(imgEl, item, jobOptions);
          item.detectionBoxes = boxes;
          item.processed = true;

          // Cache thumbnail in IndexedDB if within limit
          if (settings.keepResultsOnClose && settings.previewCacheLimitMb > 0 && !state.thumbnailsByPath.has(item.path)) {
            const thumbUrl = generateThumbnailDataUrl(imgEl, 360);
            state.thumbnailsByPath.set(item.path, thumbUrl);
            await idbPut('thumbnails', { path: item.path, dataUrl: thumbUrl });
          }
        }
      }
    } catch (err) {
      console.warn('Error processing item:', item.path, err);
      item.processed = true;
    }
  }

  async function processSingleVideoItem(item, jobOptions) {
    const file = state.fileBlobsByPath.get(item.path);
    if (!file) {
      item.processed = true;
      return;
    }
    const url = URL.createObjectURL(file);
    try {
      const video = document.createElement('video');
      video.preload = 'auto';
      video.muted = true;
      video.src = url;
      await new Promise((resolve, reject) => {
        video.onloadedmetadata = resolve;
        video.onerror = reject;
        setTimeout(resolve, 2500);
      });

      const duration = Math.max(0.5, video.duration || 2);
      const sampleCount = Math.max(1, Math.min(15, Number(jobOptions.vidStride) || 5));
      const frameBoxes = [];
      const c = document.createElement('canvas');
      c.width = video.videoWidth || 640;
      c.height = video.videoHeight || 360;
      const ctx = c.getContext('2d');

      for (let s = 0; s < sampleCount; s++) {
        const t = ((s + 0.5) / sampleCount) * duration;
        video.currentTime = t;
        await new Promise((r) => {
          video.onseeked = r;
          setTimeout(r, 400);
        });
        ctx.drawImage(video, 0, 0, c.width, c.height);
        if (s === 0 && !state.thumbnailsByPath.has(item.path)) {
          const thumbUrl = generateThumbnailDataUrl(c, 360);
          state.thumbnailsByPath.set(item.path, thumbUrl);
          await idbPut('thumbnails', { path: item.path, dataUrl: thumbUrl });
        }
        const boxes = await detectImageElement(c, item, jobOptions);
        for (const b of boxes) {
          frameBoxes.push({ ...b, frameIndex: s, timestamp: Number(t.toFixed(2)) });
        }
      }

      item.totalFramesProcessed = sampleCount;
      item.detectionBoxes = frameBoxes;
      item.processed = true;
    } catch (_) {
      item.processed = true;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function loadMediaImageElement(item) {
    return new Promise((resolve) => {
      const file = state.fileBlobsByPath.get(item.path);
      const thumb = state.thumbnailsByPath.get(item.path);
      if (!file && !thumb) {
        resolve(null);
        return;
      }
      const img = new Image();
      let objectUrl = null;
      img.onload = () => {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        resolve(img);
      };
      img.onerror = () => {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        resolve(null);
      };
      if (file) {
        objectUrl = URL.createObjectURL(file);
        img.src = objectUrl;
      } else {
        img.src = thumb;
      }
    });
  }

  // ========================================================================
  // 10. SPECIES BUCKETS, CONFIDENCE FILTERING & AUTO-GROUPING ENGINE
  // ========================================================================
  function getConfidenceThresholdForSpecies(species) {
    if (species && state.confidenceSettings[species] !== undefined) {
      return state.confidenceSettings[species];
    }
    return state.confidenceSettings.global ?? settings.confidence ?? 0.25;
  }

  function getFilteredBoxesForItem(item) {
    const boxes = item.detectionBoxes || [];
    return boxes.filter((b) => {
      const thresh = getConfidenceThresholdForSpecies(b.species);
      return (b.confidence ?? 1) >= thresh;
    });
  }

  // Compute effective species & counts for an item under current confidence thresholds
  function getEffectiveItemSummary(item) {
    if (item.manualValidated) {
      const spStr = (item.manualSpecies || '空').trim() || '空';
      const cntStr = spStr === '空' ? '空' : (item.manualCounts || '1');
      const speciesList = spStr === '空' ? [] : spStr.split(',').map((s) => s.trim()).filter(Boolean);
      const typesSet = new Set();
      if (item.manualTypes) {
        item.manualTypes.split(',').map((t) => t.trim()).filter(Boolean).forEach((t) => typesSet.add(t));
      } else {
        for (const sp of speciesList) {
          const tax = lookupSpeciesTaxonomy(sp);
          if (tax && tax.物种类型) typesSet.add(tax.物种类型);
        }
      }
      return {
        speciesStr: spStr,
        countsStr: cntStr,
        typesStr: Array.from(typesSet).join(',') || (spStr === '空' ? '' : '兽/鸟'),
        minConfidenceStr: '人工校验',
        primaryBucketSpecies: speciesList[0] || '空',
        isValidated: true,
        speciesList,
        countsMap: parseSpeciesCountsMap(spStr, cntStr)
      };
    }

    const validBoxes = getFilteredBoxesForItem(item);
    if (validBoxes.length === 0) {
      return {
        speciesStr: '空',
        countsStr: '空',
        typesStr: '',
        minConfidenceStr: '',
        primaryBucketSpecies: '空',
        isValidated: false,
        speciesList: [],
        countsMap: new Map()
      };
    }

    const countsMap = new Map();
    let minConf = Infinity;
    for (const box of validBoxes) {
      const sp = box.species || '动物';
      countsMap.set(sp, (countsMap.get(sp) || 0) + 1);
      if (box.confidence !== undefined && box.confidence < minConf) {
        minConf = box.confidence;
      }
    }

    // For video items, scale count by frame count or take max per frame
    if (item.isVideo && item.totalFramesProcessed > 1) {
      const minFrames = Math.ceil(item.totalFramesProcessed * (settings.minFrameRatio || 0));
      for (const [sp, totalFrames] of countsMap.entries()) {
        if (totalFrames < minFrames) {
          countsMap.delete(sp);
        } else {
          countsMap.set(sp, Math.max(1, Math.round(totalFrames / item.totalFramesProcessed)));
        }
      }
      if (countsMap.size === 0) {
        return {
          speciesStr: '空', countsStr: '空', typesStr: '', minConfidenceStr: '',
          primaryBucketSpecies: '空', isValidated: false, speciesList: [], countsMap: new Map()
        };
      }
    }

    const speciesList = Array.from(countsMap.keys());
    const countsList = speciesList.map((s) => String(countsMap.get(s)));
    const typesSet = new Set();
    for (const sp of speciesList) {
      const tax = lookupSpeciesTaxonomy(sp);
      if (tax && tax.物种类型) typesSet.add(tax.物种类型);
    }

    return {
      speciesStr: speciesList.join(','),
      countsStr: countsList.join(','),
      typesStr: Array.from(typesSet).join(','),
      minConfidenceStr: isFinite(minConf) ? minConf.toFixed(3) : '',
      primaryBucketSpecies: speciesList[0] || '空',
      isValidated: false,
      speciesList,
      countsMap
    };
  }

  function parseSpeciesCountsMap(speciesStr, countsStr) {
    const map = new Map();
    if (!speciesStr || speciesStr === '空') return map;
    const spArr = speciesStr.split(',').map((s) => s.trim()).filter(Boolean);
    const cntArr = (countsStr || '').split(',').map((c) => parseInt(c.trim(), 10) || 1);
    spArr.forEach((sp, i) => map.set(sp, cntArr[i] || 1));
    return map;
  }

  function getScopedItems() {
    if (state.selectedSubfolder === '__ALL__') return state.items;
    return state.items.filter((i) => i.subfolder === state.selectedSubfolder);
  }

  // Build burst groups and species buckets matching Neri's SpeciesValidationScreen
  function rebuildBucketsAndRows() {
    let baseItems = getScopedItems();

    // Filter by status & search query
    if (state.studioStatusFilter === 'unvalidated') {
      baseItems = baseItems.filter((i) => !i.manualValidated);
    } else if (state.studioStatusFilter === 'validated') {
      baseItems = baseItems.filter((i) => i.manualValidated);
    } else if (state.studioStatusFilter === 'favorite') {
      baseItems = baseItems.filter((i) => i.favorite);
    }

    const q = state.studioSearchQuery.trim().toLowerCase();
    if (q) {
      baseItems = baseItems.filter((i) => {
        const sum = getEffectiveItemSummary(i);
        return (
          i.filename.toLowerCase().includes(q) ||
          i.subfolder.toLowerCase().includes(q) ||
          sum.speciesStr.toLowerCase().includes(q) ||
          (i.remark || '').toLowerCase().includes(q)
        );
      });
    }

    // Step 1: Group into consecutive burst events if autoGroup is enabled
    const groups = [];
    const maxGapMs = (Number(settings.gapSeconds) || 30) * 1000;
    const maxBurst = Number(settings.burstSize) || 3;

    if (!settings.autoGroup) {
      baseItems.forEach((item, idx) => {
        const sum = getEffectiveItemSummary(item);
        groups.push({
          signature: `single_${item.path}`,
          items: [item],
          species: sum.primaryBucketSpecies,
          isValidated: item.manualValidated,
          index: idx
        });
      });
    } else {
      let currentGroup = null;
      for (const item of baseItems) {
        if (!currentGroup) {
          currentGroup = [item];
          continue;
        }
        const prev = currentGroup[currentGroup.length - 1];
        const sameFolder = prev.subfolder === item.subfolder;
        const timeDiff = Math.abs((item.timestamp || 0) - (prev.timestamp || 0));
        const withinGap = timeDiff <= maxGapMs;
        const underBurst = currentGroup.length < maxBurst || (item.isVideo && withinGap);

        if (sameFolder && withinGap && underBurst) {
          currentGroup.push(item);
        } else {
          groups.push(finalizeMediaGroup(currentGroup, groups.length));
          currentGroup = [item];
        }
      }
      if (currentGroup && currentGroup.length > 0) {
        groups.push(finalizeMediaGroup(currentGroup, groups.length));
      }
    }

    // Step 2: Bucket groups by (species + isValidated status)
    const bucketMap = new Map();
    for (const grp of groups) {
      const key = `${grp.isValidated ? 'val' : 'raw'}::${grp.species}`;
      if (!bucketMap.has(key)) {
        bucketMap.set(key, {
          key,
          species: grp.species,
          isValidated: grp.isValidated,
          groups: [],
          items: []
        });
      }
      const b = bucketMap.get(key);
      b.groups.push(grp);
      b.items.push(...grp.items);
    }

    // Order buckets: unvalidated non-empty first, then unvalidated empty, then validated
    state.buckets = Array.from(bucketMap.values()).sort((a, b) => {
      if (a.isValidated !== b.isValidated) return a.isValidated ? 1 : -1;
      if ((a.species === '空') !== (b.species === '空')) return a.species === '空' ? 1 : -1;
      return b.items.length - a.items.length;
    });

    if (state.buckets.length === 0) {
      state.selectedBucketKey = null;
      state.visibleRows = [];
      return;
    }

    if (!state.selectedBucketKey || !state.buckets.some((b) => b.key === state.selectedBucketKey)) {
      state.selectedBucketKey = state.buckets[0].key;
    }

    const activeBucket = state.buckets.find((b) => b.key === state.selectedBucketKey) || state.buckets[0];
    const rows = [];
    for (const grp of activeBucket.groups) {
      if (settings.autoGroup && grp.items.length > 1) {
        const expanded = settings.collapseGroups
          ? state.expandedGroupSignatures.has(grp.signature)
          : !state.expandedGroupSignatures.has(grp.signature);
        rows.push({
          isGroupHeader: true,
          group: grp,
          item: grp.items[0],
          expanded
        });
        if (expanded) {
          for (const child of grp.items) {
            rows.push({
              isGroupHeader: false,
              isGroupChild: true,
              group: grp,
              item: child,
              expanded: true
            });
          }
        }
      } else {
        rows.push({
          isGroupHeader: false,
          isGroupChild: false,
          group: grp,
          item: grp.items[0],
          expanded: false
        });
      }
    }
    state.visibleRows = rows;

    // Ensure valid selection in active bucket
    const allVisibleItemPaths = new Set(activeBucket.items.map((i) => i.path));
    if (!state.selectedPath || !allVisibleItemPaths.has(state.selectedPath)) {
      const firstRow = rows[0];
      if (firstRow) {
        selectStudioRow(firstRow, false, false);
      }
    }
  }

  function finalizeMediaGroup(items, index) {
    // Dominant species across group items
    const votes = new Map();
    let allValidated = true;
    for (const item of items) {
      const sum = getEffectiveItemSummary(item);
      if (!item.manualValidated) allValidated = false;
      const sp = sum.primaryBucketSpecies;
      votes.set(sp, (votes.get(sp) || 0) + (sp === '空' ? 0.5 : 1));
    }
    let bestSp = '空';
    let bestScore = -1;
    for (const [sp, sc] of votes.entries()) {
      if (sc > bestScore) {
        bestScore = sc;
        bestSp = sp;
      }
    }
    return {
      signature: `grp_${items[0].path}`,
      items,
      species: bestSp,
      isValidated: allValidated,
      index
    };
  }

  // ========================================================================
  // 11. MANUAL MARKING, MULTI-SPECIES QUANTITY & UNDO ENGINE
  // ========================================================================
  function getTargetItemsForMarking() {
    const targets = [];
    const seen = new Set();

    // If a collapsed group header is selected, mark all items in that group
    for (const row of state.visibleRows) {
      if (row.isGroupHeader && !row.expanded && state.selectedPaths.has(row.item.path)) {
        for (const member of row.group.items) {
          if (!seen.has(member.path)) {
            seen.add(member.path);
            targets.push(member);
          }
        }
      }
    }

    for (const path of state.selectedPaths) {
      if (!seen.has(path)) {
        const it = state.itemsByPath.get(path);
        if (it) {
          seen.add(path);
          targets.push(it);
        }
      }
    }

    if (targets.length === 0 && state.selectedPath) {
      const it = state.itemsByPath.get(state.selectedPath);
      if (it) targets.push(it);
    }
    return targets;
  }

  function recordUndoSnapshot(items) {
    const snapshot = items.map((it) => ({
      path: it.path,
      manualValidated: it.manualValidated,
      manualSpecies: it.manualSpecies,
      manualCounts: it.manualCounts,
      manualTypes: it.manualTypes,
      remark: it.remark
    }));
    state.markHistory.push(snapshot);
    const maxUndo = Number(settings.undoSteps) || 50;
    if (state.markHistory.length > maxUndo) {
      state.markHistory.shift();
    }
    updateUndoButtonUI();
  }

  async function undoLastMark() {
    if (state.markHistory.length === 0) return;
    const snapshot = state.markHistory.pop();
    const modifiedItems = [];
    for (const prev of snapshot) {
      const item = state.itemsByPath.get(prev.path);
      if (item) {
        item.manualValidated = prev.manualValidated;
        item.manualSpecies = prev.manualSpecies;
        item.manualCounts = prev.manualCounts;
        item.manualTypes = prev.manualTypes;
        item.remark = prev.remark;
        modifiedItems.push(item);
      }
    }
    state.pendingItemPath = null;
    state.pendingSpeciesNames = [];
    await persistItemsBatch(modifiedItems);
    updateUndoButtonUI();
    refreshAllUI();
    showSnackbar(`已撤回上一步标记（恢复 ${modifiedItems.length} 个文件）`);
  }

  function updateUndoButtonUI() {
    const btn = document.getElementById('btnUndoMark');
    if (btn) {
      btn.disabled = state.markHistory.length === 0;
      btn.title = state.markHistory.length > 0
        ? `撤回上一步标记 (可撤回 ${state.markHistory.length} 步 · Ctrl+Z)`
        : '暂无可撤回的操作';
    }
  }

  // Mark 'correct' or 'empty'
  async function markSelectionAction(action) {
    const targets = getTargetItemsForMarking();
    if (targets.length === 0) return;
    recordUndoSnapshot(targets);

    for (const item of targets) {
      if (action === 'empty') {
        item.manualValidated = true;
        item.manualSpecies = '空';
        item.manualCounts = '空';
        item.manualTypes = '';
      } else if (action === 'correct') {
        const sum = getEffectiveItemSummary(item);
        item.manualValidated = true;
        item.manualSpecies = sum.speciesStr || '空';
        item.manualCounts = sum.countsStr || '空';
        item.manualTypes = sum.typesStr || '';
      }
    }

    state.pendingItemPath = null;
    state.pendingSpeciesNames = [];
    await persistItemsBatch(targets);
    refreshAllUI();
    showSnackbar(action === 'empty' ? `已标记 ${targets.length} 项为空拍` : `已确认 ${targets.length} 项检测结果正确`);
  }

  // Quick Species Button Click (supports multi-species composition on current selection!)
  async function markQuickSpecies(speciesName) {
    const targets = getTargetItemsForMarking();
    if (targets.length === 0) return;
    recordUndoSnapshot(targets);

    const selectionKey = targets.map((t) => t.path).join('|');
    const isSamePendingSelection = state.pendingItemPath === selectionKey;

    if (!isSamePendingSelection) {
      state.pendingItemPath = selectionKey;
      state.pendingSpeciesNames = [speciesName];
    } else {
      const idx = state.pendingSpeciesNames.indexOf(speciesName);
      if (idx >= 0 && state.pendingSpeciesNames.length > 1) {
        state.pendingSpeciesNames.splice(idx, 1);
      } else if (idx < 0) {
        state.pendingSpeciesNames.push(speciesName);
      }
    }

    // Record usage frequency for auto-sort
    settings.quickSpeciesUsage[speciesName] = (settings.quickSpeciesUsage[speciesName] || 0) + 1;
    saveSettings();

    for (const item of targets) {
      const existingMap = isSamePendingSelection
        ? parseSpeciesCountsMap(item.manualSpecies, item.manualCounts)
        : new Map();
      const finalCounts = state.pendingSpeciesNames.map((sp) => String(existingMap.get(sp) || 1));
      const typesSet = new Set();
      for (const sp of state.pendingSpeciesNames) {
        const tax = lookupSpeciesTaxonomy(sp);
        if (tax && tax.物种类型) typesSet.add(tax.物种类型);
      }
      item.manualValidated = true;
      item.manualSpecies = state.pendingSpeciesNames.join(',');
      item.manualCounts = finalCounts.join(',');
      item.manualTypes = Array.from(typesSet).join(',');
    }

    await persistItemsBatch(targets);
    refreshAllUI();
    showSnackbar(`已标记 ${targets.length} 项为：${state.pendingSpeciesNames.join('、')}`);
  }

  // Quantity Button Click (updates count of last marked species or current species)
  async function markQuantity(qtyStr) {
    const targets = getTargetItemsForMarking();
    if (targets.length === 0) return;
    recordUndoSnapshot(targets);

    for (const item of targets) {
      const sum = getEffectiveItemSummary(item);
      let spList = item.manualValidated && item.manualSpecies && item.manualSpecies !== '空'
        ? item.manualSpecies.split(',').map((s) => s.trim()).filter(Boolean)
        : [...sum.speciesList];

      if (spList.length === 0) spList = ['动物'];
      const cntList = (item.manualValidated ? item.manualCounts : sum.countsStr)
        .split(',')
        .map((c) => c.trim())
        .filter((c) => c && c !== '空');
      while (cntList.length < spList.length) cntList.push('1');
      cntList[cntList.length - 1] = qtyStr;

      item.manualValidated = true;
      item.manualSpecies = spList.join(',');
      item.manualCounts = cntList.slice(0, spList.length).join(',');
    }

    await persistItemsBatch(targets);
    refreshAllUI();
    showSnackbar(`已将 ${targets.length} 项的物种数量更新为 ${qtyStr}`);
  }

  // ========================================================================
  // 12. TAXONOMY ENRICHMENT, WORKING DAYS, INDEPENDENT DETECTION & CSV EXPORT
  // ========================================================================
  function buildEnrichedExportRows(itemsSubset) {
    if (!itemsSubset || itemsSubset.length === 0) return [];

    // 1. Sort chronologically
    const sorted = [...itemsSubset].sort(compareMediaItems);

    // 2. Find earliest shooting date in this directory scope for Working Days (工作天数)
    let earliestDateMidnight = null;
    for (const item of sorted) {
      if (item.dateTaken) {
        const d = new Date(`${item.dateTaken}T00:00:00`);
        if (!isNaN(d.getTime())) {
          if (earliestDateMidnight === null || d.getTime() < earliestDateMidnight) {
            earliestDateMidnight = d.getTime();
          }
        }
      }
    }

    // 3. Independent Detection Threshold (30 minutes default, matching system/data_processor.py)
    const indepThresholdMs = (Number(settings.independentMinutes) || 30) * 60 * 1000;
    const speciesLastDetected = new Map();

    const rows = [];
    for (const item of sorted) {
      const sum = getEffectiveItemSummary(item);
      if (settings.emptyExportPolicy === 'exclude_empty' && sum.speciesStr === '空') {
        continue;
      }

      // Calculate Working Days
      let workingDays = '';
      if (earliestDateMidnight !== null && item.dateTaken) {
        const curMidnight = new Date(`${item.dateTaken}T00:00:00`).getTime();
        if (!isNaN(curMidnight)) {
          workingDays = String(Math.round((curMidnight - earliestDateMidnight) / 86400000) + 1);
        }
      }

      // Calculate Independent Detection First Individual (独立探测首只)
      let indepFlag = '';
      const validSpecies = sum.speciesList.filter((s) => s && s !== '空' && s !== 'Unknown');
      if (item.timestamp && validSpecies.length > 0) {
        let isIndependent = false;
        for (const sp of validSpecies) {
          const lastTs = speciesLastDetected.get(sp);
          if (lastTs === undefined || (item.timestamp - lastTs) > indepThresholdMs) {
            isIndependent = true;
          }
          speciesLastDetected.set(sp, item.timestamp);
        }
        if (isIndependent) indepFlag = '1';
      }

      // Lookup 2026《中国生物物种名录》taxonomy fields
      const sciList = [], ordCnList = [], ordLatList = [], famCnList = [], famLatList = [], genCnList = [], genLatList = [];
      const typeSet = new Set();

      for (const sp of validSpecies) {
        const tax = lookupSpeciesTaxonomy(sp);
        if (tax) {
          sciList.push(tax.学名 || '');
          ordCnList.push(tax.目名 || '');
          ordLatList.push(tax.目拉丁名 || '');
          famCnList.push(tax.科名 || '');
          famLatList.push(tax.科拉丁名 || '');
          genCnList.push(tax.属名 || '');
          genLatList.push(tax.属拉丁名 || '');
          if (tax.物种类型) typeSet.add(tax.物种类型);
        } else {
          sciList.push(''); ordCnList.push(''); ordLatList.push('');
          famCnList.push(''); famLatList.push(''); genCnList.push(''); genLatList.push('');
        }
      }

      if (item.manualValidated && item.manualTypes) {
        item.manualTypes.split(',').map((t) => t.trim()).filter(Boolean).forEach((t) => typeSet.add(t));
      }

      rows.push({
        _subfolder: item.subfolder,
        '文件名': item.filename,
        '格式': item.ext,
        '拍摄日期': item.dateTaken || '',
        '拍摄时间': item.timeTaken || '',
        '工作天数': workingDays,
        '物种名称': sum.speciesStr,
        '学名': sciList.join(','),
        '目名': ordCnList.join(','),
        '目拉丁名': ordLatList.join(','),
        '科名': famCnList.join(','),
        '科拉丁名': famLatList.join(','),
        '属名': genCnList.join(','),
        '属拉丁名': genLatList.join(','),
        '物种类型': Array.from(typeSet).join(','),
        '物种数量': sum.countsStr,
        '最低置信度': sum.minConfidenceStr,
        '独立探测首只': indepFlag,
        '备注': item.remark || ''
      });
    }
    return rows;
  }

  function rowsToCsvString(rows, columns) {
    const cols = columns && columns.length ? columns : DEFAULT_EXPORT_COLUMNS;
    const escapeCsvCell = (val) => {
      const str = String(val ?? '');
      if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
        return '"' + str.replace(/"/g, '""') + '"';
      }
      return str;
    };
    const lines = [cols.map(escapeCsvCell).join(',')];
    for (const r of rows) {
      lines.push(cols.map((c) => escapeCsvCell(r[c] ?? '')).join(','));
    }
    // Prepend UTF-8 BOM (\uFEFF) so Microsoft Excel opens Chinese CSV cleanly
    return '\uFEFF' + lines.join('\r\n');
  }

  function triggerCsvDownload(filename, csvContent) {
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  }

  function exportSingleMergedCsv(customSubfolder) {
    const targetItems = customSubfolder
      ? state.items.filter((i) => i.subfolder === customSubfolder)
      : getScopedItems();
    if (targetItems.length === 0) {
      showSnackbar('当前没有可导出的媒体数据');
      return;
    }
    const rows = buildEnrichedExportRows(targetItems);
    const csv = rowsToCsvString(rows, settings.exportColumns);
    const scopeName = customSubfolder || (state.selectedSubfolder === '__ALL__' ? state.rootFolderName : state.selectedSubfolder) || 'Neri';
    const safeName = scopeName.replace(/[\\/:*?"<>|]/g, '_');
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    triggerCsvDownload(`物种检测信息_${safeName}_${stamp}.csv`, csv);
    showSnackbar(`已导出 CSV：共 ${rows.length} 条记录`);
  }

  async function exportPerSubfolderBatchCsv() {
    if (state.items.length === 0) {
      showSnackbar('当前没有可导出的媒体数据');
      return;
    }
    const byFolder = new Map();
    for (const item of state.items) {
      const folder = item.subfolder || '根目录';
      if (!byFolder.has(folder)) byFolder.set(folder, []);
      byFolder.get(folder).push(item);
    }

    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    let tableCount = 0;
    let totalRecords = 0;

    for (const [folder, folderItems] of byFolder.entries()) {
      const rows = buildEnrichedExportRows(folderItems);
      if (rows.length === 0) continue;
      const csv = rowsToCsvString(rows, settings.exportColumns);
      const safeFolder = folder.replace(/[\\/:*?"<>|]/g, '_');
      triggerCsvDownload(`物种检测信息_${safeFolder}_${stamp}.csv`, csv);
      tableCount++;
      totalRecords += rows.length;
      await new Promise((r) => setTimeout(r, 180));
    }

    showSnackbar(`批量导出完成：已按子目录生成 ${tableCount} 份 CSV 表格（合计 ${totalRecords} 条记录）`);
  }

  function handleExportButtonClick() {
    if (state.items.length === 0) {
      showSnackbar('请先导入文件夹或加载演示数据集');
      return;
    }
    const subfolders = new Set(state.items.map((i) => i.subfolder));
    if (settings.exportScopeMode === 'single' || subfolders.size <= 1) {
      exportSingleMergedCsv();
      return;
    }
    if (settings.exportScopeMode === 'per_folder') {
      exportPerSubfolderBatchCsv();
      return;
    }
    // Open Export Choice Modal
    const modalDesc1 = document.getElementById('modalCurrentScopeDesc');
    const modalDesc2 = document.getElementById('modalPerSubfolderDesc');
    if (modalDesc1) {
      modalDesc1.textContent = `导出当前范围（${getScopedItems().length} 个媒体文件）合并为 1 份 CSV 表格`;
    }
    if (modalDesc2) {
      modalDesc2.textContent = `检测到 ${subfolders.size} 个子文件夹，每个子目录独立计算工作天数与独立首只并分别生成 CSV`;
    }
    openModal('exportCsvModal');
  }

  // ========================================================================
  // 13. UI RENDERING & INTERACTIVE CANVAS VIEWER
  // ========================================================================
  function refreshAllUI() {
    updatePersistenceChip();
    updateModelStatusUI();
    updateTopSubfolderSelect();
    updateHeroStats();
    renderJobsList();
    renderSubfolderOverview();
    rebuildBucketsAndRows();
    renderStudioLeftLists();
    renderStudioQuickMarkButtons();
    renderStudioStage();
    if (state.activeView === 'export') {
      renderExportDataTable();
    }
  }

  function updateTopSubfolderSelect() {
    const rootLabel = document.getElementById('topRootFolderLabel');
    const select = document.getElementById('globalSubfolderSelect');
    if (rootLabel) {
      rootLabel.textContent = state.rootFolderName || '未选择文件夹';
    }
    if (!select) return;

    const folderCounts = new Map();
    for (const item of state.items) {
      folderCounts.set(item.subfolder, (folderCounts.get(item.subfolder) || 0) + 1);
    }

    const prevVal = state.selectedSubfolder;
    select.innerHTML = '';
    const allOpt = document.createElement('option');
    allOpt.value = '__ALL__';
    allOpt.textContent = `全部目录 (${state.items.length})`;
    select.appendChild(allOpt);

    for (const [folder, count] of folderCounts.entries()) {
      const opt = document.createElement('option');
      opt.value = folder;
      opt.textContent = `${folder} (${count})`;
      select.appendChild(opt);
    }

    if (prevVal !== '__ALL__' && folderCounts.has(prevVal)) {
      select.value = prevVal;
    } else {
      state.selectedSubfolder = '__ALL__';
      select.value = '__ALL__';
    }
  }

  function updateHeroStats() {
    const scoped = getScopedItems();
    let detected = 0, empty = 0, validated = 0;
    for (const item of scoped) {
      const sum = getEffectiveItemSummary(item);
      if (item.manualValidated) validated++;
      if (sum.speciesStr !== '空') detected++;
      else if (item.processed || item.manualValidated) empty++;
    }
    const setNum = (id, v) => {
      const el = document.getElementById(id);
      if (el) el.textContent = String(v);
    };
    setNum('heroStatTotal', scoped.length);
    setNum('heroStatDetected', detected);
    setNum('heroStatEmpty', empty);
    setNum('heroStatValidated', validated);

    // Update navrail pending badge
    const badge = document.getElementById('navPendingBadge');
    if (badge) {
      const unvalidated = scoped.length - validated;
      if (unvalidated > 0) {
        badge.style.display = 'inline-flex';
        badge.textContent = unvalidated > 99 ? '99+' : String(unvalidated);
      } else {
        badge.style.display = 'none';
      }
    }
  }

  function renderJobsList() {
    const container = document.getElementById('jobsListContainer');
    const clearBtn = document.getElementById('btnClearJobs');
    if (!container) return;
    if (clearBtn) clearBtn.style.display = state.jobs.length > 0 ? 'inline-flex' : 'none';

    if (state.jobs.length === 0) {
      container.innerHTML = `<div style="color:var(--md-on-surface-variant); font-size:13.5px; padding:8px 0;">创建任务后，本地推理进度与实时速度会显示在这里。</div>`;
      return;
    }

    container.innerHTML = state.jobs.map((job) => {
      const pct = job.total > 0 ? Math.round((job.processed / job.total) * 100) : 0;
      const speedText = job.speed > 0 ? `${job.speed.toFixed(1)} 项/秒` : '';
      const elapsedText = job.elapsedSeconds > 0 ? `已用 ${Math.round(job.elapsedSeconds)}s` : '';
      const remText = job.state === 'running' && job.remainingSeconds != null ? `剩余 ${Math.round(job.remainingSeconds)}s` : '';
      const metaParts = [job.modelName, speedText, elapsedText, remText].filter(Boolean).join(' · ');

      return `
        <div class="job-card">
          <div class="job-card-top">
            <div class="job-card-title">
              <span style="color:var(--md-primary)">●</span>
              <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(job.inputDir)}</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              <span class="status-chip" style="padding:3px 10px; font-size:11.5px;">${job.processed}/${job.total} (${pct}%)</span>
              ${job.state === 'running'
                ? `<button type="button" class="md3-btn md3-btn-tonal" data-job-cancel="${job.id}" style="min-height:28px; padding:2px 10px; font-size:11.5px;">暂停</button>`
                : job.state === 'paused'
                  ? `<button type="button" class="md3-btn md3-btn-filled" data-job-resume="${job.id}" style="min-height:28px; padding:2px 10px; font-size:11.5px;">继续</button>`
                  : `<button type="button" class="md3-btn md3-btn-tonal" data-job-validate="${job.id}" style="min-height:28px; padding:2px 10px; font-size:11.5px;">去标记</button>`
              }
            </div>
          </div>
          <div class="md3-progress-track">
            <div class="md3-progress-fill" style="width:${pct}%"></div>
          </div>
          <div class="job-card-meta">
            <span>${escapeHtml(job.message || '')}</span>
            <span>${escapeHtml(metaParts)}</span>
          </div>
        </div>
      `;
    }).join('');
  }

  function renderSubfolderOverview() {
    const container = document.getElementById('subfolderOverviewContainer');
    if (!container) return;
    if (state.items.length === 0) {
      container.innerHTML = `<div style="color:var(--md-on-surface-variant); font-size:13.5px; padding:8px 0;">尚未加载文件夹。</div>`;
      return;
    }

    const map = new Map();
    for (const item of state.items) {
      if (!map.has(item.subfolder)) {
        map.set(item.subfolder, { total: 0, detected: 0, validated: 0 });
      }
      const rec = map.get(item.subfolder);
      rec.total++;
      const sum = getEffectiveItemSummary(item);
      if (sum.speciesStr !== '空') rec.detected++;
      if (item.manualValidated) rec.validated++;
    }

    container.innerHTML = Array.from(map.entries()).map(([folder, st]) => `
      <div class="selectable-tile" style="background:var(--md-surface-container); padding:10px 14px; margin-bottom:8px;">
        <div class="tile-main">
          <div class="tile-text">
            <div class="tile-title">${escapeHtml(folder)}</div>
            <div class="tile-subtitle">共 ${st.total} 项 · 检出目标 ${st.detected} 项 · 已人工标记 ${st.validated} 项</div>
          </div>
        </div>
        <div style="display:flex; gap:6px;">
          <button type="button" class="md3-btn md3-btn-tonal" data-open-subfolder="${escapeHtml(folder)}" style="min-height:30px; padding:4px 12px; font-size:12px;">人工标记</button>
          <button type="button" class="md3-btn md3-btn-outlined" data-export-subfolder="${escapeHtml(folder)}" style="min-height:30px; padding:4px 12px; font-size:12px;">导出 CSV</button>
        </div>
      </div>
    `).join('');
  }

  function renderStudioLeftLists() {
    const bucketListEl = document.getElementById('speciesBucketList');
    const bucketBadge = document.getElementById('bucketCountBadge');
    const fileListEl = document.getElementById('validationFileList');
    const fileBadge = document.getElementById('fileCountBadge');
    const selChip = document.getElementById('studioSelectionChip');

    if (selChip) selChip.textContent = `已选 ${Math.max(1, state.selectedPaths.size)} 项`;
    if (bucketBadge) bucketBadge.textContent = `${state.buckets.length} 类`;
    if (fileBadge) fileBadge.textContent = `${state.visibleRows.length} 行`;

    if (!bucketListEl || !fileListEl) return;

    if (state.buckets.length === 0) {
      bucketListEl.innerHTML = `<div style="padding:12px; color:var(--md-on-surface-variant); font-size:12.5px;">暂无分组数据</div>`;
      fileListEl.innerHTML = `<div style="padding:12px; color:var(--md-on-surface-variant); font-size:12.5px;">暂无媒体文件</div>`;
      return;
    }

    bucketListEl.innerHTML = state.buckets.map((b) => {
      const selected = b.key === state.selectedBucketKey;
      const color = getSpeciesColor(b.species);
      const statusIcon = b.isValidated
        ? `<span style="color:var(--md-success); font-weight:900;" title="已人工校验">✓</span>`
        : `<span style="color:var(--md-warning); font-weight:900;" title="待人工校验">●</span>`;
      const subtitle = settings.autoGroup
        ? `${b.groups.length} 组 / ${b.items.length} 个文件`
        : `${b.items.length} 个文件`;

      return `
        <div class="selectable-tile ${selected ? 'selected' : ''}" data-bucket-key="${escapeHtml(b.key)}">
          <div class="tile-main">
            ${statusIcon}
            <span style="width:10px; height:10px; border-radius:50%; background:${color}; flex-shrink:0;"></span>
            <div class="tile-text">
              <div class="tile-title">${escapeHtml(b.species)} ${b.isValidated ? '(已校验)' : ''}</div>
              <div class="tile-subtitle">${subtitle}</div>
            </div>
          </div>
        </div>
      `;
    }).join('');

    fileListEl.innerHTML = state.visibleRows.map((row, idx) => {
      const isSel = row.isGroupHeader
        ? row.group.items.every((m) => state.selectedPaths.has(m.path)) || state.selectedPath === row.item.path
        : state.selectedPaths.has(row.item.path);
      const sum = getEffectiveItemSummary(row.item);

      if (row.isGroupHeader) {
        const g = row.group;
        const timeRange = `${g.items[0].timeTaken || ''} (${g.items.length} 连拍)`;
        return `
          <div class="selectable-tile group-header ${isSel ? 'selected' : ''}" data-row-index="${idx}">
            <div class="tile-main">
              <span style="font-size:14px;">📁</span>
              <div class="tile-text">
                <div class="tile-title">事件组 #${g.index + 1} · ${escapeHtml(g.species)}</div>
                <div class="tile-subtitle">${escapeHtml(timeRange)} · ${escapeHtml(g.items[0].subfolder)}</div>
              </div>
            </div>
            <button type="button" class="md3-icon-btn" data-toggle-group="${escapeHtml(g.signature)}" style="width:28px; height:28px;" title="${row.expanded ? '折叠连拍组' : '展开连拍组'}">
              ${row.expanded ? '▲' : '▼'}
            </button>
          </div>
        `;
      }

      const favIcon = row.item.favorite ? '★ ' : '';
      const valIcon = row.item.manualValidated ? '✓ ' : '';
      const sub = `${valIcon}${sum.speciesStr}${sum.countsStr && sum.countsStr !== '空' ? ' ×' + sum.countsStr : ''} · ${row.item.timeTaken || row.item.subfolder}`;

      return `
        <div class="selectable-tile ${row.isGroupChild ? 'group-child' : ''} ${isSel ? 'selected' : ''}" data-row-index="${idx}">
          <div class="tile-main">
            <span style="font-size:13px;">${row.item.isVideo ? '🎬' : '🖼️'}</span>
            <div class="tile-text">
              <div class="tile-title">${favIcon}${escapeHtml(row.item.filename)}</div>
              <div class="tile-subtitle">${escapeHtml(sub)}</div>
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  function selectStudioRow(row, ctrlKey, shiftKey) {
    if (!row) return;
    const targetPath = row.item.path;
    state.selectedPath = targetPath;
    state.selectedBoxIndex = -1;
    state.pendingItemPath = null;
    state.pendingSpeciesNames = [];

    if (shiftKey && state.selectionAnchorPath) {
      const paths = state.visibleRows.map((r) => r.item.path);
      const i1 = paths.indexOf(state.selectionAnchorPath);
      const i2 = paths.indexOf(targetPath);
      if (i1 >= 0 && i2 >= 0) {
        const [start, end] = i1 < i2 ? [i1, i2] : [i2, i1];
        state.selectedPaths.clear();
        for (let k = start; k <= end; k++) {
          const r = state.visibleRows[k];
          if (r.isGroupHeader && !r.expanded) {
            r.group.items.forEach((m) => state.selectedPaths.add(m.path));
          } else {
            state.selectedPaths.add(r.item.path);
          }
        }
      }
    } else if (ctrlKey) {
      if (state.selectedPaths.has(targetPath) && state.selectedPaths.size > 1) {
        state.selectedPaths.delete(targetPath);
      } else {
        state.selectedPaths.add(targetPath);
      }
      state.selectionAnchorPath = targetPath;
    } else {
      state.selectedPaths.clear();
      if (row.isGroupHeader && !row.expanded) {
        row.group.items.forEach((m) => state.selectedPaths.add(m.path));
      } else {
        state.selectedPaths.add(targetPath);
      }
      state.selectionAnchorPath = targetPath;
    }

    renderStudioLeftLists();
    renderStudioQuickMarkButtons();
    renderStudioStage();
  }

  function renderStudioQuickMarkButtons() {
    const spContainer = document.getElementById('quickSpeciesListContainer');
    const qtyContainer = document.getElementById('quantityButtonsContainer');
    if (!spContainer || !qtyContainer) return;

    let spList = [...settings.quickSpeciesList];
    if (settings.autoSortQuickMarks) {
      spList.sort((a, b) => (settings.quickSpeciesUsage[b] || 0) - (settings.quickSpeciesUsage[a] || 0));
    }

    const currentItem = state.selectedPath ? state.itemsByPath.get(state.selectedPath) : null;
    const activeSpeciesSet = new Set();
    if (currentItem) {
      const sum = getEffectiveItemSummary(currentItem);
      sum.speciesList.forEach((s) => activeSpeciesSet.add(s));
    }

    spContainer.innerHTML = spList.map((sp, i) => {
      const isSel = activeSpeciesSet.has(sp);
      const shortcutHint = i < 9 ? ` [${i + 1}]` : '';
      return `
        <button type="button" class="mark-btn ${isSel ? 'selected' : ''}" data-quick-species="${escapeHtml(sp)}" title="标记为 ${escapeHtml(sp)}${shortcutHint}">
          ${escapeHtml(sp)}
        </button>
      `;
    }).join('');

    qtyContainer.innerHTML = settings.quantityButtons.map((q) => `
      <button type="button" class="mark-btn" data-quick-qty="${escapeHtml(q)}">${escapeHtml(q)}</button>
    `).join('');
  }

  async function renderStudioStage() {
    const stageCard = document.getElementById('mediaStageCard');
    const gridView = document.getElementById('studioGridView');
    const emptyPrompt = document.getElementById('emptyStagePrompt');
    const canvas = document.getElementById('mediaCanvas');
    const video = document.getElementById('mediaVideo');
    const fnBadge = document.getElementById('stageFilenameBadge');
    const valBadge = document.getElementById('stageValidationBadge');
    const summaryLine = document.getElementById('summaryTextLine');
    const remarkInput = document.getElementById('quickRemarkInput');
    const favSvg = document.getElementById('favStarSvg');
    const delBoxBtn = document.getElementById('btnDeleteSelectedBox');

    if (delBoxBtn) {
      delBoxBtn.style.display = state.selectedBoxIndex >= 0 ? 'inline-flex' : 'none';
    }

    if (state.studioViewMode === 'grid') {
      if (stageCard) stageCard.style.display = 'none';
      if (gridView) {
        gridView.style.display = 'grid';
        renderStudioGridOverview(gridView);
      }
    } else {
      if (stageCard) stageCard.style.display = 'flex';
      if (gridView) gridView.style.display = 'none';
    }

    const item = state.selectedPath ? state.itemsByPath.get(state.selectedPath) : null;
    if (!item) {
      if (emptyPrompt) emptyPrompt.style.display = 'block';
      if (canvas) canvas.style.display = 'none';
      if (video) video.style.display = 'none';
      if (fnBadge) fnBadge.textContent = '未选择文件';
      if (summaryLine) summaryLine.textContent = '物种: -  |  数量: -  |  类型: -  |  置信度: -  |  拍摄时间: -';
      return;
    }

    if (emptyPrompt) emptyPrompt.style.display = 'none';
    if (fnBadge) fnBadge.textContent = `${item.subfolder} / ${item.filename}`;
    if (valBadge) {
      valBadge.className = item.manualValidated ? 'status-chip ready' : 'status-chip warning';
      valBadge.textContent = item.manualValidated ? '✓ 已人工校验' : '待校验';
    }
    if (favSvg) {
      favSvg.setAttribute('fill', item.favorite ? 'currentColor' : 'none');
      favSvg.style.color = item.favorite ? '#f59e0b' : 'currentColor';
    }
    if (remarkInput && document.activeElement !== remarkInput) {
      remarkInput.value = item.remark || '';
    }

    const sum = getEffectiveItemSummary(item);
    if (summaryLine) {
      summaryLine.textContent = `物种: ${sum.speciesStr}  |  数量: ${sum.countsStr}  |  类型: ${sum.typesStr || '-'}  |  置信度: ${sum.minConfidenceStr || '-'}  |  拍摄时间: ${item.dateTaken} ${item.timeTaken}`;
    }

    // Update species filter dropdown options
    updateStudioSpeciesFilterDropdown(item);

    // Render image + bounding boxes on Canvas
    if (canvas) canvas.style.display = 'block';
    if (video) video.style.display = 'none';
    const imgEl = await loadMediaImageElement(item);
    state.activeImageElement = imgEl;
    drawMediaCanvas(item, imgEl);
  }

  function drawMediaCanvas(item, imgEl) {
    const canvas = document.getElementById('mediaCanvas');
    if (!canvas || !item) return;
    const ctx = canvas.getContext('2d');
    const w = (imgEl && (imgEl.naturalWidth || imgEl.width)) || item.width || 960;
    const h = (imgEl && (imgEl.naturalHeight || imgEl.height)) || item.height || 640;
    canvas.width = w;
    canvas.height = h;
    canvas.style.transform = `scale(${state.zoom}) translate(${state.panX}px, ${state.panY}px)`;

    ctx.clearRect(0, 0, w, h);
    if (imgEl) {
      ctx.drawImage(imgEl, 0, 0, w, h);
    } else {
      ctx.fillStyle = '#25282a';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 20px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${item.filename} (重新关联本地原文件夹可查看高清原图)`, w / 2, h / 2);
      ctx.textAlign = 'left';
    }

    if (!state.showDetections) return;

    const visibleBoxes = getFilteredBoxesForItem(item);
    visibleBoxes.forEach((box, idx) => {
      if (state.filterSpecies !== 'global' && box.species !== state.filterSpecies) return;
      const [x1, y1, x2, y2] = box.bbox;
      const bx = x1 * w, by = y1 * h, bw = (x2 - x1) * w, bh = (y2 - y1) * h;
      const color = getSpeciesColor(box.species);
      const isSelectedBox = idx === state.selectedBoxIndex;

      ctx.save();
      ctx.strokeStyle = isSelectedBox ? '#ffffff' : color;
      ctx.lineWidth = Math.max(3, Math.round(w / 260));
      ctx.strokeRect(bx, by, bw, bh);

      // Label Pill
      const confPct = box.confidence != null ? ` ${(box.confidence * 100).toFixed(0)}%` : '';
      const label = `${box.species}${confPct}`;
      ctx.font = `bold ${Math.max(14, Math.round(w / 52))}px sans-serif`;
      const textW = ctx.measureText(label).width + 14;
      const textH = Math.max(24, Math.round(w / 34));
      const ly = Math.max(0, by - textH);

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(bx, ly, textW, textH, 6);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, bx + 7, ly + textH * 0.74);
      ctx.restore();
    });

    // Draw active rubberband box if user is hand-drawing a new detection box
    if (state.drawBoxMode && state.drawingStart && state.drawingCurrent) {
      const x1 = Math.min(state.drawingStart.x, state.drawingCurrent.x) * w;
      const y1 = Math.min(state.drawingStart.y, state.drawingCurrent.y) * h;
      const bw = Math.abs(state.drawingCurrent.x - state.drawingStart.x) * w;
      const bh = Math.abs(state.drawingCurrent.y - state.drawingStart.y) * h;
      ctx.save();
      ctx.strokeStyle = '#ff9800';
      ctx.lineWidth = 3;
      ctx.setLineDash([8, 4]);
      ctx.strokeRect(x1, y1, bw, bh);
      ctx.restore();
    }
  }

  function renderStudioGridOverview(gridEl) {
    const activeBucket = state.buckets.find((b) => b.key === state.selectedBucketKey);
    const items = activeBucket ? activeBucket.items : getScopedItems();
    gridEl.innerHTML = items.map((item) => {
      const thumb = state.thumbnailsByPath.get(item.path) || '';
      const sum = getEffectiveItemSummary(item);
      const isSel = state.selectedPaths.has(item.path);
      return `
        <div class="grid-photo-card ${isSel ? 'selected' : ''}" data-grid-path="${escapeHtml(item.path)}">
          <div class="grid-photo-thumb">
            ${thumb ? `<img src="${thumb}" alt="${escapeHtml(item.filename)}" loading="lazy" />` : `<span>${item.isVideo ? '🎬 视频' : '🖼️ 图像'}</span>`}
          </div>
          <div class="grid-photo-info">
            <div style="font-weight:700; font-size:12.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(item.filename)}</div>
            <div style="font-size:11.5px; color:var(--md-on-surface-variant); margin-top:2px;">
              ${item.manualValidated ? '✓ ' : ''}${escapeHtml(sum.speciesStr)} ${sum.countsStr !== '空' ? '×' + sum.countsStr : ''}
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  function updateStudioSpeciesFilterDropdown(item) {
    const select = document.getElementById('filterSpeciesSelect');
    if (!select) return;
    const speciesSet = new Set(['global']);
    if (item && item.detectionBoxes) {
      item.detectionBoxes.forEach((b) => {
        if (b.species) speciesSet.add(b.species);
      });
    }
    const prev = state.filterSpecies;
    select.innerHTML = '';
    for (const sp of speciesSet) {
      const opt = document.createElement('option');
      opt.value = sp;
      opt.textContent = sp === 'global' ? '全局设置' : sp;
      select.appendChild(opt);
    }
    select.value = speciesSet.has(prev) ? prev : 'global';
  }

  function renderExportDataTable() {
    const headEl = document.getElementById('exportTableHead');
    const bodyEl = document.getElementById('exportTableBody');
    const statsEl = document.getElementById('exportStatsRow');
    if (!headEl || !bodyEl) return;

    const scoped = getScopedItems();
    const rows = buildEnrichedExportRows(scoped);
    const cols = settings.exportColumns.length ? settings.exportColumns : DEFAULT_EXPORT_COLUMNS;

    if (statsEl) {
      const indepCount = rows.filter((r) => r['独立探测首只'] === '1').length;
      const uniqueSpecies = new Set(rows.map((r) => r['物种名称']).filter((s) => s && s !== '空'));
      statsEl.innerHTML = `
        <div class="status-chip ready">总导出行数：${rows.length}</div>
        <div class="status-chip">独立有效探测 (首只=1)：${indepCount}</div>
        <div class="status-chip">识别物种数：${uniqueSpecies.size}</div>
        <div class="status-chip">导出字段数：${cols.length} 列</div>
      `;
    }

    headEl.innerHTML = `<tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}</tr>`;
    if (rows.length === 0) {
      bodyEl.innerHTML = `<tr><td colspan="${cols.length}" style="text-align:center; padding:24px; color:var(--md-on-surface-variant);">暂无数据，请先在“开始”页面导入文件夹或加载演示数据集</td></tr>`;
      return;
    }
    bodyEl.innerHTML = rows.slice(0, 300).map((r) => `
      <tr>${cols.map((c) => `<td>${escapeHtml(r[c] ?? '')}</td>`).join('')}</tr>
    `).join('');
  }

  // ========================================================================
  // 14. SETTINGS & CACHE MANAGEMENT UI
  // ========================================================================
  function renderSeedPaletteSwatches() {
    const container = document.getElementById('seedPaletteContainer');
    if (!container) return;
    container.innerHTML = SEED_COLOR_OPTIONS.map((opt) => {
      const active = (settings.seedColor || '').toUpperCase() === opt.color.toUpperCase();
      return `
        <button type="button" class="seed-swatch ${active ? 'active' : ''}" data-seed-color="${opt.color}">
          <span class="seed-dot" style="background:${opt.color}"></span>
          <span>${escapeHtml(opt.label)}</span>
        </button>
      `;
    }).join('');
  }

  function renderSettingsControls() {
    const setChk = (id, val) => { const el = document.getElementById(id); if (el) el.checked = Boolean(val); };
    const setVal = (id, val) => { const el = document.getElementById(id); if (el) el.value = String(val); };

    setChk('setKeepResultsOnClose', settings.keepResultsOnClose);
    setVal('setPreviewCacheLimitMb', settings.previewCacheLimitMb);
    setChk('setAutoGroup', settings.autoGroup);
    setChk('setCollapseGroups', settings.collapseGroups);
    setVal('setBurstSize', settings.burstSize);
    setVal('setGapSeconds', settings.gapSeconds);
    setVal('setUndoSteps', settings.undoSteps);
    const undoValEl = document.getElementById('setUndoStepsVal');
    if (undoValEl) undoValEl.textContent = `${settings.undoSteps} 步`;
    setChk('setAutoSortQuickMarks', settings.autoSortQuickMarks);
    setChk('setAgnosticNms', settings.agnosticNms);
    setVal('setMinFrameRatio', settings.minFrameRatio);
    const mfrValEl = document.getElementById('setMinFrameRatioVal');
    if (mfrValEl) mfrValEl.textContent = `${Math.round(settings.minFrameRatio * 100)}%`;
    setVal('setIndependentMinutes', settings.independentMinutes);
    setVal('setExportScopeMode', settings.exportScopeMode);
    setVal('setEmptyExportPolicy', settings.emptyExportPolicy);
    setVal('setThemeMode', settings.themeMode);

    renderQuickSpeciesSettingsChips();
    renderExportColumnsCheckboxes();
    renderSeedPaletteSwatches();
    updateCacheBreakdownUI();
  }

  function renderQuickSpeciesSettingsChips() {
    const container = document.getElementById('settingsQuickSpeciesChips');
    if (!container) return;
    container.innerHTML = settings.quickSpeciesList.map((sp, idx) => `
      <span class="status-chip" style="cursor:pointer; background:var(--md-secondary-container); color:var(--md-on-secondary-container);" data-remove-quick-sp="${idx}" title="点击移除「${escapeHtml(sp)}」">
        ${escapeHtml(sp)} ✕
      </span>
    `).join('');
  }

  function renderExportColumnsCheckboxes() {
    const container = document.getElementById('exportColumnsCheckboxes');
    if (!container) return;
    const selectedSet = new Set(settings.exportColumns);
    container.innerHTML = DEFAULT_EXPORT_COLUMNS.map((col) => `
      <label style="display:flex; align-items:center; gap:6px; font-size:13px; cursor:pointer; padding:6px 10px; background:var(--md-surface-container); border-radius:var(--radius-sm);">
        <input type="checkbox" data-export-col="${escapeHtml(col)}" ${selectedSet.has(col) ? 'checked' : ''} />
        <span>${escapeHtml(col)}</span>
      </label>
    `).join('');
  }

  async function updateCacheBreakdownUI() {
    const totalEl = document.getElementById('cacheTotalSizeText');
    const linesEl = document.getElementById('cacheBreakdownLines');
    if (!totalEl || !linesEl) return;

    const [items, thumbs, jobs, models] = await Promise.all([
      idbGetAll('items'),
      idbGetAll('thumbnails'),
      idbGetAll('jobs'),
      idbGetAll('models')
    ]);

    const itemsBytes = new Blob([JSON.stringify(items)]).size;
    const thumbsBytes = thumbs.reduce((acc, t) => acc + ((t.dataUrl && t.dataUrl.length) || 0), 0);
    const jobsBytes = new Blob([JSON.stringify(jobs)]).size;
    const modelsBytes = models.reduce((acc, m) => acc + ((m.buffer && m.buffer.byteLength) || 0), 0);
    const totalBytes = itemsBytes + thumbsBytes + jobsBytes + modelsBytes;

    totalEl.textContent = `总计约 ${formatBytes(totalBytes)}`;
    linesEl.innerHTML = `
      <div>• 识别与人工标记记录：<strong>${items.length} 条 (${formatBytes(itemsBytes)})</strong></div>
      <div>• 预览缩略图缓存：<strong>${thumbs.length} 张 (${formatBytes(thumbsBytes)})</strong></div>
      <div>• 本地 LiteRT 模型缓存：<strong>${models.length} 个 (${formatBytes(modelsBytes)})</strong></div>
      <div>• 任务队列与日志：<strong>${jobs.length} 条 (${formatBytes(jobsBytes)})</strong></div>
    `;
  }

  async function executeSelectiveClearCache(opts) {
    let clearedParts = [];
    if (opts.thumbnails) {
      await idbClearStore('thumbnails');
      state.thumbnailsByPath.clear();
      clearedParts.push('缩略图缓存');
    }
    if (opts.jobsLogs) {
      await idbClearStore('jobs');
      state.jobs = [];
      clearedParts.push('任务队列与日志');
    }
    if (opts.models) {
      await idbClearStore('models');
      state.customTfliteName = null;
      state.customTfliteBuffer = null;
      state.tfliteModelInstance = null;
      const el = document.getElementById('loadedTfliteName');
      if (el) el.textContent = '使用内置探测引擎 (可点击右侧加载您的 MegaDetectorV6 .tflite 权重文件)';
      clearedParts.push('本地模型缓存');
    }
    if (opts.recognitionResults) {
      await idbClearStore('items');
      await idbClearStore('meta');
      state.items = [];
      state.itemsByPath.clear();
      state.fileBlobsByPath.clear();
      state.rootFolderName = '';
      state.selectedSubfolder = '__ALL__';
      state.selectedPath = null;
      state.selectedPaths.clear();
      state.markHistory = [];
      const banner = document.getElementById('sessionRestoreBanner');
      if (banner) banner.style.display = 'none';
      clearedParts.push('识别结果与人工标记');
    }

    await updateCacheBreakdownUI();
    refreshAllUI();
    showSnackbar(`已清理：${clearedParts.join('、')}`);
  }

  // ========================================================================
  // 15. UTILITIES, MODALS & EVENT LISTENERS
  // ========================================================================
  function formatBytes(bytes) {
    if (!bytes || bytes <= 0) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(2)} MB`;
  }

  function escapeHtml(str) {
    return String(str ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function showSnackbar(message) {
    const container = document.getElementById('snackbarContainer');
    if (!container) return;
    const el = document.createElement('div');
    el.className = 'md3-snackbar';
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transition = 'opacity 0.25s';
      setTimeout(() => el.remove(), 260);
    }, 3200);
  }

  function openModal(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('open');
  }

  function closeModal(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('open');
  }

  function switchView(viewName) {
    state.activeView = viewName;
    document.querySelectorAll('.nav-rail-item').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.view === viewName);
    });
    document.querySelectorAll('.view-panel').forEach((panel) => {
      panel.classList.toggle('active', panel.id === `view-${viewName}`);
    });
    refreshAllUI();
    if (viewName === 'settings') {
      renderSettingsControls();
    }
  }

  function moveSelectedMedia(delta) {
    if (state.visibleRows.length === 0) return;
    const curIdx = state.visibleRows.findIndex((r) => r.item.path === state.selectedPath);
    const nextIdx = Math.max(0, Math.min(state.visibleRows.length - 1, (curIdx < 0 ? 0 : curIdx) + delta));
    selectStudioRow(state.visibleRows[nextIdx], false, false);
  }

  function renderOtherSpeciesAutocomplete(query) {
    const listEl = document.getElementById('speciesAutocompleteList');
    if (!listEl) return;
    const q = (query || '').trim().toLowerCase();
    const matches = q
      ? taxonomyList.filter((t) =>
          t.中文名.toLowerCase().includes(q) ||
          t.学名.toLowerCase().includes(q) ||
          t.科名.toLowerCase().includes(q) ||
          t.目名.toLowerCase().includes(q)
        ).slice(0, 40)
      : taxonomyList.slice(0, 25);

    if (matches.length === 0) {
      listEl.innerHTML = `<div style="padding:10px 12px; font-size:12.5px; color:var(--md-on-surface-variant);">名录中未找到匹配物种，直接点击“确认标记”可使用自定义物种名</div>`;
      return;
    }

    listEl.innerHTML = matches.map((m) => `
      <div class="selectable-tile" data-pick-taxonomy="${escapeHtml(m.中文名)}" data-pick-type="${escapeHtml(m.物种类型)}" style="padding:7px 12px;">
        <div class="tile-main">
          <div class="tile-text">
            <div class="tile-title">${escapeHtml(m.中文名)} <span style="font-weight:400; font-style:italic; color:var(--md-on-surface-variant); font-size:12px;">${escapeHtml(m.学名)}</span></div>
            <div class="tile-subtitle">${escapeHtml(m.物种类型)} · ${escapeHtml(m.目名)} · ${escapeHtml(m.科名)}</div>
          </div>
        </div>
      </div>
    `).join('');
  }

  function bindEvents() {
    // Navigation Rail
    document.querySelectorAll('.nav-rail-item').forEach((btn) => {
      btn.addEventListener('click', () => switchView(btn.dataset.view));
    });
    document.getElementById('brandHomeBtn')?.addEventListener('click', () => switchView('start'));

    // Quick Theme & Shortcuts
    document.getElementById('quickShortcutsBtn')?.addEventListener('click', () => openModal('shortcutsModal'));
    document.getElementById('quickThemeToggleBtn')?.addEventListener('click', () => {
      const isDark = document.documentElement.classList.contains('dark');
      settings.themeMode = isDark ? 'light' : 'dark';
      saveSettings();
      applyTheme();
    });

    // Modal Close Buttons
    document.querySelectorAll('[data-close-modal]').forEach((btn) => {
      btn.addEventListener('click', () => closeModal(btn.dataset.closeModal));
    });

    // Folder & File Inputs
    const folderInput = document.getElementById('folderInput');
    const filesInput = document.getElementById('filesInput');
    const tfliteInput = document.getElementById('tfliteModelInput');
    const backupInput = document.getElementById('backupJsonInput');

    const triggerFolderPick = () => folderInput?.click();
    document.getElementById('topPickFolderBtn')?.addEventListener('click', triggerFolderPick);
    document.getElementById('btnSelectFolderMain')?.addEventListener('click', (e) => {
      e.stopPropagation();
      triggerFolderPick();
    });
    document.getElementById('restoreRelinkFolderBtn')?.addEventListener('click', triggerFolderPick);
    document.getElementById('restoreGoValidateBtn')?.addEventListener('click', () => switchView('validate'));

    document.getElementById('btnSelectFilesMain')?.addEventListener('click', (e) => {
      e.stopPropagation();
      filesInput?.click();
    });
    document.getElementById('btnLoadDemoDataset')?.addEventListener('click', (e) => {
      e.stopPropagation();
      loadDemoCameraTrapDataset();
    });

    folderInput?.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        importFileList(e.target.files);
      }
    });
    filesInput?.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        importFileList(e.target.files, '所选媒体文件');
      }
    });

    // Drag & Drop on Dropzone
    const dropzone = document.getElementById('folderDropzone');
    if (dropzone) {
      dropzone.addEventListener('click', (e) => {
        if (e.target === dropzone || e.target.id === 'dropzoneMainText' || e.target.id === 'dropzoneSubText') {
          triggerFolderPick();
        }
      });
      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
      dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          importFileList(e.dataTransfer.files);
        }
      });
    }

    // LiteRT (.tflite) Model Upload & Caching
    const triggerTflitePick = () => tfliteInput?.click();
    document.getElementById('btnUploadTflite')?.addEventListener('click', triggerTflitePick);
    document.getElementById('modelStatusChip')?.addEventListener('click', triggerTflitePick);
    document.getElementById('selectLiteRtModel')?.addEventListener('change', (e) => {
      settings.modelVariant = e.target.value;
      saveSettings();
      updateModelStatusUI();
    });

    tfliteInput?.addEventListener('change', async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const buffer = await file.arrayBuffer();
      state.customTfliteName = file.name;
      state.customTfliteBuffer = buffer;
      state.customTfliteMeta = inspectTfliteFlatBuffer(buffer);
      if (state.customTfliteMeta && state.customTfliteMeta.detectedInputSize) {
        settings.inputSize = state.customTfliteMeta.detectedInputSize;
        const sizeSel = document.getElementById('selectInputSize');
        if (sizeSel) sizeSel.value = String(settings.inputSize);
      }
      await idbPut('models', { id: 'custom_litert_model', name: file.name, buffer, updatedAt: Date.now() });
      await tryInitTfliteRuntime(buffer);
      const modelSel = document.getElementById('selectLiteRtModel');
      if (modelSel) modelSel.value = 'custom-litert';
      settings.modelVariant = 'custom-litert';
      saveSettings();
      const metaStr = state.customTfliteMeta ? ` · ${state.customTfliteMeta.summary}` : '';
      const el = document.getElementById('loadedTfliteName');
      if (el) el.textContent = `${file.name} (${formatBytes(buffer.byteLength)}${metaStr} · 已缓存至浏览器 IndexedDB)`;
      updateModelStatusUI();
      updateCacheBreakdownUI();
      showSnackbar(`已加载并缓存 LiteRT 模型：${file.name}`);
    });

    // Global Subfolder Scope Selector
    document.getElementById('globalSubfolderSelect')?.addEventListener('change', (e) => {
      state.selectedSubfolder = e.target.value;
      refreshAllUI();
    });

    // Start Screen Sliders & Start Job Button
    document.getElementById('startConfSlider')?.addEventListener('input', (e) => {
      const v = Number(e.target.value);
      settings.confidence = v;
      saveSettings();
      const lbl = document.getElementById('startConfVal');
      if (lbl) lbl.textContent = v.toFixed(2);
    });
    document.getElementById('startIouSlider')?.addEventListener('input', (e) => {
      const v = Number(e.target.value);
      settings.iou = v;
      saveSettings();
      const lbl = document.getElementById('startIouVal');
      if (lbl) lbl.textContent = v.toFixed(2);
    });
    document.getElementById('btnStartDetectionJob')?.addEventListener('click', () => startLocalDetectionJob());

    // Job Card Delegation
    document.getElementById('jobsListContainer')?.addEventListener('click', (e) => {
      const cancelBtn = e.target.closest('[data-job-cancel]');
      if (cancelBtn) {
        cancelJob(cancelBtn.dataset.jobCancel);
        return;
      }
      const resumeBtn = e.target.closest('[data-job-resume]');
      if (resumeBtn) {
        resumeJob(resumeBtn.dataset.jobResume);
        return;
      }
      const valBtn = e.target.closest('[data-job-validate]');
      if (valBtn) {
        switchView('validate');
      }
    });
    document.getElementById('btnClearJobs')?.addEventListener('click', async () => {
      state.jobs = [];
      await idbClearStore('jobs');
      renderJobsList();
    });

    // Subfolder Overview Delegation
    document.getElementById('subfolderOverviewContainer')?.addEventListener('click', (e) => {
      const openBtn = e.target.closest('[data-open-subfolder]');
      if (openBtn) {
        state.selectedSubfolder = openBtn.dataset.openSubfolder;
        switchView('validate');
        return;
      }
      const expBtn = e.target.closest('[data-export-subfolder]');
      if (expBtn) {
        exportSingleMergedCsv(expBtn.dataset.exportSubfolder);
      }
    });

    // Validation Studio Left Lists
    document.getElementById('speciesBucketList')?.addEventListener('click', (e) => {
      const tile = e.target.closest('[data-bucket-key]');
      if (!tile) return;
      state.selectedBucketKey = tile.dataset.bucketKey;
      rebuildBucketsAndRows();
      if (state.visibleRows.length > 0) {
        selectStudioRow(state.visibleRows[0], false, false);
      } else {
        refreshAllUI();
      }
    });

    document.getElementById('validationFileList')?.addEventListener('click', (e) => {
      const toggleBtn = e.target.closest('[data-toggle-group]');
      if (toggleBtn) {
        e.stopPropagation();
        const sig = toggleBtn.dataset.toggleGroup;
        if (state.expandedGroupSignatures.has(sig)) state.expandedGroupSignatures.delete(sig);
        else state.expandedGroupSignatures.add(sig);
        rebuildBucketsAndRows();
        renderStudioLeftLists();
        return;
      }
      const tile = e.target.closest('[data-row-index]');
      if (!tile) return;
      const row = state.visibleRows[Number(tile.dataset.rowIndex)];
      if (row) selectStudioRow(row, e.ctrlKey || e.metaKey, e.shiftKey);
    });

    // Studio Grid View Click
    document.getElementById('studioGridView')?.addEventListener('click', (e) => {
      const card = e.target.closest('[data-grid-path]');
      if (!card) return;
      const path = card.dataset.gridPath;
      state.selectedPath = path;
      if (!e.ctrlKey && !e.metaKey) state.selectedPaths.clear();
      state.selectedPaths.add(path);
      renderStudioLeftLists();
      renderStudioStage();
    });
    document.getElementById('studioGridView')?.addEventListener('dblclick', (e) => {
      const card = e.target.closest('[data-grid-path]');
      if (!card) return;
      state.studioViewMode = 'split';
      renderStudioStage();
    });

    // Studio Toolbar Controls
    document.getElementById('btnViewSplitMode')?.addEventListener('click', () => {
      state.studioViewMode = 'split';
      document.getElementById('btnViewSplitMode').className = 'md3-btn md3-btn-tonal';
      document.getElementById('btnViewGridMode').className = 'md3-btn md3-btn-text';
      renderStudioStage();
    });
    document.getElementById('btnViewGridMode')?.addEventListener('click', () => {
      state.studioViewMode = 'grid';
      document.getElementById('btnViewGridMode').className = 'md3-btn md3-btn-tonal';
      document.getElementById('btnViewSplitMode').className = 'md3-btn md3-btn-text';
      renderStudioStage();
    });
    document.getElementById('studioStatusFilter')?.addEventListener('change', (e) => {
      state.studioStatusFilter = e.target.value;
      refreshAllUI();
    });
    document.getElementById('studioSearchInput')?.addEventListener('input', (e) => {
      state.studioSearchQuery = e.target.value;
      refreshAllUI();
    });
    document.getElementById('btnSelectAllInBucket')?.addEventListener('click', () => {
      const activeBucket = state.buckets.find((b) => b.key === state.selectedBucketKey);
      if (!activeBucket) return;
      state.selectedPaths.clear();
      activeBucket.items.forEach((i) => state.selectedPaths.add(i.path));
      renderStudioLeftLists();
      showSnackbar(`已全选当前分组内全部 ${activeBucket.items.length} 个文件`);
    });
    document.getElementById('btnToggleExpandGroups')?.addEventListener('click', () => {
      const activeBucket = state.buckets.find((b) => b.key === state.selectedBucketKey);
      if (!activeBucket) return;
      const allExpanded = activeBucket.groups.every((g) => state.expandedGroupSignatures.has(g.signature));
      activeBucket.groups.forEach((g) => {
        if (allExpanded) state.expandedGroupSignatures.delete(g.signature);
        else state.expandedGroupSignatures.add(g.signature);
      });
      rebuildBucketsAndRows();
      renderStudioLeftLists();
    });
    document.getElementById('btnRedetectSelected')?.addEventListener('click', () => {
      const targets = getTargetItemsForMarking();
      if (targets.length === 0) return;
      startLocalDetectionJob(targets);
    });

    // Stage Prev/Next/Zoom/Favorite/DrawBox
    document.getElementById('btnPrevMedia')?.addEventListener('click', () => moveSelectedMedia(-1));
    document.getElementById('btnNextMedia')?.addEventListener('click', () => moveSelectedMedia(1));
    document.getElementById('btnZoomIn')?.addEventListener('click', () => {
      state.zoom = Math.min(4, state.zoom + 0.25);
      const item = state.selectedPath ? state.itemsByPath.get(state.selectedPath) : null;
      drawMediaCanvas(item, state.activeImageElement);
    });
    document.getElementById('btnZoomOut')?.addEventListener('click', () => {
      state.zoom = Math.max(0.5, state.zoom - 0.25);
      const item = state.selectedPath ? state.itemsByPath.get(state.selectedPath) : null;
      drawMediaCanvas(item, state.activeImageElement);
    });
    document.getElementById('btnZoomReset')?.addEventListener('click', () => {
      state.zoom = 1;
      state.panX = 0;
      state.panY = 0;
      const item = state.selectedPath ? state.itemsByPath.get(state.selectedPath) : null;
      drawMediaCanvas(item, state.activeImageElement);
    });
    document.getElementById('btnToggleFavorite')?.addEventListener('click', async () => {
      const targets = getTargetItemsForMarking();
      if (targets.length === 0) return;
      const nextFav = !targets[0].favorite;
      targets.forEach((t) => { t.favorite = nextFav; });
      await persistItemsBatch(targets);
      renderStudioLeftLists();
      renderStudioStage();
      showSnackbar(nextFav ? '已添加至收藏媒体 ★' : '已取消收藏');
    });

    // Hand-draw Bounding Box & Click Box on Canvas
    const drawBoxBtn = document.getElementById('btnDrawBoxMode');
    drawBoxBtn?.addEventListener('click', () => {
      state.drawBoxMode = !state.drawBoxMode;
      drawBoxBtn.classList.toggle('active', state.drawBoxMode);
      showSnackbar(state.drawBoxMode ? '已进入手绘检测框模式：在图像上按住鼠标拖拽即可新增目标框' : '已退出手绘检测框模式');
    });
    document.getElementById('btnDeleteSelectedBox')?.addEventListener('click', async () => {
      const item = state.selectedPath ? state.itemsByPath.get(state.selectedPath) : null;
      if (!item || state.selectedBoxIndex < 0) return;
      const visible = getFilteredBoxesForItem(item);
      const targetBox = visible[state.selectedBoxIndex];
      if (targetBox) {
        const idx = item.detectionBoxes.indexOf(targetBox);
        if (idx >= 0) item.detectionBoxes.splice(idx, 1);
        state.selectedBoxIndex = -1;
        await persistItemsBatch([item]);
        refreshAllUI();
        showSnackbar('已删除选中的检测框');
      }
    });

    const mediaCanvas = document.getElementById('mediaCanvas');
    mediaCanvas?.addEventListener('mousedown', (e) => {
      const item = state.selectedPath ? state.itemsByPath.get(state.selectedPath) : null;
      if (!item) return;
      const rect = mediaCanvas.getBoundingClientRect();
      const nx = clamp01((e.clientX - rect.left) / rect.width);
      const ny = clamp01((e.clientY - rect.top) / rect.height);

      if (state.drawBoxMode) {
        state.drawingStart = { x: nx, y: ny };
        state.drawingCurrent = { x: nx, y: ny };
        return;
      }

      // Check if user clicked inside an existing bounding box
      const visible = getFilteredBoxesForItem(item);
      let hitIdx = -1;
      for (let i = visible.length - 1; i >= 0; i--) {
        const [x1, y1, x2, y2] = visible[i].bbox;
        if (nx >= x1 && nx <= x2 && ny >= y1 && ny <= y2) {
          hitIdx = i;
          break;
        }
      }
      state.selectedBoxIndex = hitIdx;
      renderStudioStage();
    });

    mediaCanvas?.addEventListener('mousemove', (e) => {
      if (!state.drawBoxMode || !state.drawingStart) return;
      const rect = mediaCanvas.getBoundingClientRect();
      state.drawingCurrent = {
        x: clamp01((e.clientX - rect.left) / rect.width),
        y: clamp01((e.clientY - rect.top) / rect.height)
      };
      const item = state.selectedPath ? state.itemsByPath.get(state.selectedPath) : null;
      drawMediaCanvas(item, state.activeImageElement);
    });

    window.addEventListener('mouseup', async () => {
      if (!state.drawBoxMode || !state.drawingStart || !state.drawingCurrent) return;
      const x1 = Math.min(state.drawingStart.x, state.drawingCurrent.x);
      const y1 = Math.min(state.drawingStart.y, state.drawingCurrent.y);
      const x2 = Math.max(state.drawingStart.x, state.drawingCurrent.x);
      const y2 = Math.max(state.drawingStart.y, state.drawingCurrent.y);
      state.drawingStart = null;
      state.drawingCurrent = null;

      if (x2 - x1 > 0.02 && y2 - y1 > 0.02 && state.selectedPath) {
        const item = state.itemsByPath.get(state.selectedPath);
        if (item) {
          const sum = getEffectiveItemSummary(item);
          const sp = sum.primaryBucketSpecies !== '空' ? sum.primaryBucketSpecies : '动物';
          item.detectionBoxes.push({
            species: sp,
            rawClass: 'animal',
            confidence: 0.99,
            bbox: [Number(x1.toFixed(4)), Number(y1.toFixed(4)), Number(x2.toFixed(4)), Number(y2.toFixed(4))]
          });
          await persistItemsBatch([item]);
          refreshAllUI();
          showSnackbar(`已新增人工检测框 (${sp})`);
        }
      }
    });

    // Remark input
    document.getElementById('quickRemarkInput')?.addEventListener('change', async (e) => {
      const targets = getTargetItemsForMarking();
      if (targets.length === 0) return;
      const val = e.target.value.trim();
      targets.forEach((t) => { t.remark = val; });
      await persistItemsBatch(targets);
      renderStudioLeftLists();
      showSnackbar('已保存备注');
    });

    // Confidence Filter Bar in Studio
    document.getElementById('chkShowDetections')?.addEventListener('change', (e) => {
      state.showDetections = e.target.checked;
      const item = state.selectedPath ? state.itemsByPath.get(state.selectedPath) : null;
      drawMediaCanvas(item, state.activeImageElement);
    });
    document.getElementById('filterSpeciesSelect')?.addEventListener('change', (e) => {
      state.filterSpecies = e.target.value;
      const conf = getConfidenceThresholdForSpecies(state.filterSpecies);
      const slider = document.getElementById('validationConfSlider');
      const valEl = document.getElementById('validationConfVal');
      if (slider) slider.value = String(conf);
      if (valEl) valEl.textContent = conf.toFixed(2);
      const item = state.selectedPath ? state.itemsByPath.get(state.selectedPath) : null;
      drawMediaCanvas(item, state.activeImageElement);
    });
    document.getElementById('validationConfSlider')?.addEventListener('input', (e) => {
      const v = Number(e.target.value);
      state.confidenceSettings[state.filterSpecies] = v;
      const valEl = document.getElementById('validationConfVal');
      if (valEl) valEl.textContent = v.toFixed(2);
      rebuildBucketsAndRows();
      renderStudioLeftLists();
      renderStudioStage();
    });

    // Quick Marking Dock Buttons
    document.getElementById('btnMarkCorrect')?.addEventListener('click', () => markSelectionAction('correct'));
    document.getElementById('btnMarkEmpty')?.addEventListener('click', () => markSelectionAction('empty'));
    document.getElementById('btnUndoMark')?.addEventListener('click', () => undoLastMark());

    document.getElementById('quickSpeciesListContainer')?.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-quick-species]');
      if (btn) markQuickSpecies(btn.dataset.quickSpecies);
    });
    document.getElementById('quantityButtonsContainer')?.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-quick-qty]');
      if (btn) markQuantity(btn.dataset.quickQty);
    });

    // Other Species Modal
    document.getElementById('btnMarkOther')?.addEventListener('click', () => {
      const input = document.getElementById('otherSpeciesInput');
      if (input) input.value = '';
      renderOtherSpeciesAutocomplete('');
      openModal('otherSpeciesModal');
      setTimeout(() => input?.focus(), 60);
    });
    document.getElementById('otherSpeciesInput')?.addEventListener('input', (e) => {
      renderOtherSpeciesAutocomplete(e.target.value);
    });
    document.getElementById('speciesAutocompleteList')?.addEventListener('click', (e) => {
      const tile = e.target.closest('[data-pick-taxonomy]');
      if (!tile) return;
      const input = document.getElementById('otherSpeciesInput');
      const typeSel = document.getElementById('otherSpeciesType');
      if (input) input.value = tile.dataset.pickTaxonomy;
      if (typeSel && tile.dataset.pickType) typeSel.value = tile.dataset.pickType;
    });
    document.getElementById('btnConfirmOtherSpecies')?.addEventListener('click', async () => {
      const sp = (document.getElementById('otherSpeciesInput')?.value || '').trim();
      if (!sp) {
        showSnackbar('请输入或选择物种名称');
        return;
      }
      const cnt = String(Math.max(1, parseInt(document.getElementById('otherSpeciesCount')?.value || '1', 10)));
      const explicitType = document.getElementById('otherSpeciesType')?.value || '';
      const remark = (document.getElementById('otherSpeciesRemark')?.value || '').trim();
      const addQuick = document.getElementById('chkAddOtherToQuick')?.checked;

      if (addQuick && !settings.quickSpeciesList.includes(sp)) {
        settings.quickSpeciesList.unshift(sp);
        saveSettings();
      }

      const targets = getTargetItemsForMarking();
      if (targets.length > 0) {
        recordUndoSnapshot(targets);
        const tax = lookupSpeciesTaxonomy(sp);
        const finalType = explicitType || (tax && tax.物种类型) || '';
        for (const item of targets) {
          item.manualValidated = true;
          item.manualSpecies = sp;
          item.manualCounts = cnt;
          item.manualTypes = finalType;
          if (remark) item.remark = remark;
        }
        await persistItemsBatch(targets);
      }
      closeModal('otherSpeciesModal');
      refreshAllUI();
      showSnackbar(`已标记为：${sp} ×${cnt}`);
    });

    // CSV Export Buttons & Modal
    document.getElementById('topExportCsvBtn')?.addEventListener('click', handleExportButtonClick);
    document.getElementById('btnStudioExportCsv')?.addEventListener('click', handleExportButtonClick);
    document.getElementById('btnExportMergedCsv')?.addEventListener('click', () => exportSingleMergedCsv());
    document.getElementById('btnExportPerFolderCsv')?.addEventListener('click', () => exportPerSubfolderBatchCsv());

    document.getElementById('btnModalExportCurrentScope')?.addEventListener('click', () => {
      if (document.getElementById('chkRememberExportChoice')?.checked) {
        settings.exportScopeMode = 'single';
        saveSettings();
      }
      closeModal('exportCsvModal');
      exportSingleMergedCsv();
    });
    document.getElementById('btnModalExportPerSubfolder')?.addEventListener('click', () => {
      if (document.getElementById('chkRememberExportChoice')?.checked) {
        settings.exportScopeMode = 'per_folder';
        saveSettings();
      }
      closeModal('exportCsvModal');
      exportPerSubfolderBatchCsv();
    });

    // Backup & Restore JSON
    document.getElementById('btnExportBackupJson')?.addEventListener('click', () => {
      const payload = {
        version: 1,
        rootFolderName: state.rootFolderName,
        exportedAt: new Date().toISOString(),
        items: state.items.map(serializeItemForStorage)
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Neri_Backup_${state.rootFolderName || 'project'}.json`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    document.getElementById('btnImportBackupJson')?.addEventListener('click', () => backupInput?.click());
    backupInput?.addEventListener('change', async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        if (Array.isArray(data.items)) {
          state.rootFolderName = data.rootFolderName || '恢复的项目';
          state.items = data.items.sort(compareMediaItems);
          state.itemsByPath.clear();
          state.items.forEach((it) => state.itemsByPath.set(it.path, it));
          await persistItemsBatch(state.items);
          refreshAllUI();
          showSnackbar(`已从备份恢复 ${state.items.length} 条记录`);
        }
      } catch (err) {
        showSnackbar('备份 JSON 文件解析失败');
      }
    });

    // Settings Navigation Drawer
    document.querySelectorAll('.settings-nav-item').forEach((btn) => {
      btn.addEventListener('click', () => {
        const sec = btn.dataset.sec;
        document.querySelectorAll('.settings-nav-item').forEach((b) => b.classList.toggle('active', b === btn));
        document.querySelectorAll('.settings-sec').forEach((panel) => {
          panel.style.display = panel.id === `sec-${sec}` ? 'block' : 'none';
        });
      });
    });

    // Settings Controls Listeners
    document.getElementById('setKeepResultsOnClose')?.addEventListener('change', (e) => {
      settings.keepResultsOnClose = e.target.checked;
      saveSettings();
      updatePersistenceChip();
      if (settings.keepResultsOnClose && state.items.length > 0) {
        persistItemsBatch(state.items);
      }
    });
    document.getElementById('setPreviewCacheLimitMb')?.addEventListener('change', (e) => {
      settings.previewCacheLimitMb = Number(e.target.value);
      saveSettings();
    });
    document.getElementById('btnOpenClearCacheModal')?.addEventListener('click', () => openModal('clearCacheModal'));
    document.getElementById('btnConfirmClearCache')?.addEventListener('click', async () => {
      await executeSelectiveClearCache({
        thumbnails: document.getElementById('chkClearThumbnails')?.checked,
        jobsLogs: document.getElementById('chkClearJobsLogs')?.checked,
        models: document.getElementById('chkClearModels')?.checked,
        recognitionResults: document.getElementById('chkClearRecognitionResults')?.checked
      });
      closeModal('clearCacheModal');
    });
    document.getElementById('btnClearAllCacheQuick')?.addEventListener('click', async () => {
      await executeSelectiveClearCache({
        thumbnails: true,
        jobsLogs: true,
        models: true,
        recognitionResults: true
      });
    });

    document.getElementById('setAutoGroup')?.addEventListener('change', (e) => {
      settings.autoGroup = e.target.checked;
      saveSettings();
      refreshAllUI();
    });
    document.getElementById('setCollapseGroups')?.addEventListener('change', (e) => {
      settings.collapseGroups = e.target.checked;
      saveSettings();
      refreshAllUI();
    });
    document.getElementById('setBurstSize')?.addEventListener('change', (e) => {
      settings.burstSize = Math.max(1, Math.min(20, Number(e.target.value) || 3));
      saveSettings();
      refreshAllUI();
    });
    document.getElementById('setGapSeconds')?.addEventListener('change', (e) => {
      settings.gapSeconds = Math.max(5, Math.min(3600, Number(e.target.value) || 30));
      saveSettings();
      refreshAllUI();
    });
    document.getElementById('setUndoSteps')?.addEventListener('input', (e) => {
      settings.undoSteps = Number(e.target.value) || 50;
      saveSettings();
      const lbl = document.getElementById('setUndoStepsVal');
      if (lbl) lbl.textContent = `${settings.undoSteps} 步`;
    });
    document.getElementById('setAutoSortQuickMarks')?.addEventListener('change', (e) => {
      settings.autoSortQuickMarks = e.target.checked;
      saveSettings();
      renderStudioQuickMarkButtons();
    });

    document.getElementById('settingsQuickSpeciesChips')?.addEventListener('click', (e) => {
      const chip = e.target.closest('[data-remove-quick-sp]');
      if (!chip) return;
      const idx = Number(chip.dataset.removeQuickSp);
      settings.quickSpeciesList.splice(idx, 1);
      saveSettings();
      renderQuickSpeciesSettingsChips();
      renderStudioQuickMarkButtons();
    });
    document.getElementById('btnAddQuickSpecies')?.addEventListener('click', () => {
      const input = document.getElementById('inputNewQuickSpecies');
      const val = (input?.value || '').trim();
      if (!val) return;
      if (!settings.quickSpeciesList.includes(val)) {
        settings.quickSpeciesList.push(val);
        saveSettings();
        renderQuickSpeciesSettingsChips();
        renderStudioQuickMarkButtons();
      }
      if (input) input.value = '';
    });
    document.getElementById('btnResetQuickSpecies')?.addEventListener('click', () => {
      settings.quickSpeciesList = [...DEFAULT_QUICK_SPECIES];
      saveSettings();
      renderQuickSpeciesSettingsChips();
      renderStudioQuickMarkButtons();
    });

    document.getElementById('setAgnosticNms')?.addEventListener('change', (e) => {
      settings.agnosticNms = e.target.checked;
      saveSettings();
    });
    document.getElementById('setMinFrameRatio')?.addEventListener('input', (e) => {
      settings.minFrameRatio = Number(e.target.value) || 0;
      saveSettings();
      const lbl = document.getElementById('setMinFrameRatioVal');
      if (lbl) lbl.textContent = `${Math.round(settings.minFrameRatio * 100)}%`;
    });
    document.getElementById('setIndependentMinutes')?.addEventListener('change', (e) => {
      settings.independentMinutes = Math.max(1, Number(e.target.value) || 30);
      saveSettings();
    });
    document.getElementById('setExportScopeMode')?.addEventListener('change', (e) => {
      settings.exportScopeMode = e.target.value;
      saveSettings();
    });
    document.getElementById('setEmptyExportPolicy')?.addEventListener('change', (e) => {
      settings.emptyExportPolicy = e.target.value;
      saveSettings();
    });
    document.getElementById('exportColumnsCheckboxes')?.addEventListener('change', () => {
      const checked = Array.from(document.querySelectorAll('[data-export-col]:checked')).map((el) => el.dataset.exportCol);
      settings.exportColumns = checked.length > 0 ? checked : [...DEFAULT_EXPORT_COLUMNS];
      saveSettings();
    });
    document.getElementById('btnSelectAllExportCols')?.addEventListener('click', () => {
      settings.exportColumns = [...DEFAULT_EXPORT_COLUMNS];
      saveSettings();
      renderExportColumnsCheckboxes();
    });
    document.getElementById('btnResetExportCols')?.addEventListener('click', () => {
      settings.exportColumns = [...DEFAULT_EXPORT_COLUMNS];
      saveSettings();
      renderExportColumnsCheckboxes();
    });

    document.getElementById('setThemeMode')?.addEventListener('change', (e) => {
      settings.themeMode = e.target.value;
      saveSettings();
      applyTheme();
    });
    document.getElementById('seedPaletteContainer')?.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-seed-color]');
      if (!btn) return;
      settings.seedColor = btn.dataset.seedColor;
      saveSettings();
      applyTheme();
    });

    // Global Keyboard Shortcuts in Validation Studio
    window.addEventListener('keydown', (e) => {
      const tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (document.querySelector('.modal-backdrop.open')) {
        if (e.key === 'Escape') {
          document.querySelectorAll('.modal-backdrop.open').forEach((m) => m.classList.remove('open'));
        }
        return;
      }

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        undoLastMark();
        return;
      }

      if (state.activeView !== 'validate') return;

      if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
        e.preventDefault();
        moveSelectedMedia(-1);
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
        e.preventDefault();
        moveSelectedMedia(1);
      } else if (e.key.toLowerCase() === 'c' || e.key === 'Enter') {
        e.preventDefault();
        markSelectionAction('correct');
      } else if (e.key.toLowerCase() === 'e' || e.key === ' ') {
        e.preventDefault();
        markSelectionAction('empty');
      } else if (e.key.toLowerCase() === 'o') {
        e.preventDefault();
        document.getElementById('btnMarkOther')?.click();
      } else if (e.key.toLowerCase() === 'f') {
        e.preventDefault();
        document.getElementById('btnToggleFavorite')?.click();
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        if (state.selectedBoxIndex >= 0) {
          e.preventDefault();
          document.getElementById('btnDeleteSelectedBox')?.click();
        }
      } else if (/^[1-9]$/.test(e.key)) {
        const idx = parseInt(e.key, 10) - 1;
        let spList = [...settings.quickSpeciesList];
        if (settings.autoSortQuickMarks) {
          spList.sort((a, b) => (settings.quickSpeciesUsage[b] || 0) - (settings.quickSpeciesUsage[a] || 0));
        }
        if (spList[idx]) {
          e.preventDefault();
          markQuickSpecies(spList[idx]);
        }
      }
    });
  }

  // ========================================================================
  // 16. INITIALIZATION
  // ========================================================================
  async function initApp() {
    applyTheme();
    bindEvents();
    renderSettingsControls();
    await restoreSessionFromIndexedDB();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }
})();
