const uploadForm = document.querySelector('#upload-form');
const fileInput = document.querySelector('#file-input');
const fileDropZone = document.querySelector('#file-drop-zone');
const selectedFilesList = document.querySelector('#selected-files');
const uploadStatus = document.querySelector('#upload-status');
const promptInput = document.querySelector('#prompt');
const topKInput = document.querySelector('#top-k');
const maxTokensInput = document.querySelector('#max-tokens');
const runCompareButton = document.querySelector('#run-compare');
const compareStatus = document.querySelector('#compare-status');
const resultsPanel = document.querySelector('#results');
const tokenPromptMetric = document.querySelector('#metric-prompt-tokens');
const tokenRagMetric = document.querySelector('#metric-rag-tokens');
const tokenDmlMetric = document.querySelector('#metric-dml-tokens');
const tokenDeltaMetric = document.querySelector('#metric-token-delta');
const dmlFidelityMetric = document.querySelector('#metric-dml-fidelity');
const ragDocsMetric = document.querySelector('#metric-rag-docs');
const dmlEntriesMetric = document.querySelector('#metric-dml-entries');
const baseOutput = document.querySelector('#base-output');
const dmlOutput = document.querySelector('#dml-output');
const baseUsage = document.querySelector('#base-usage');
const dmlUsage = document.querySelector('#dml-usage');
const ragAggregateOutput = document.querySelector('#rag-aggregate-output');
const ragAggregateUsage = document.querySelector('#rag-aggregate-usage');
const ragAggregateSource = document.querySelector('#rag-aggregate-source');
const ragContext = document.querySelector('#rag-context');
const dmlContext = document.querySelector('#dml-context');
const dmlSummaryList = document.querySelector('#dml-summary-list');
const insightCopy = document.querySelector('#insight-copy');
const ragDocumentsTable = document.querySelector('#rag-documents tbody');
const ragTokenGraph = document.querySelector('#rag-token-graph');
const ragResponseTabList = document.querySelector('#rag-response-tablist');
const ragResponsePanel = document.querySelector('#rag-response-panel');
const ragPanelPlaceholder = ragResponsePanel?.querySelector('.rag-panel-placeholder') || null;
const ragBackendView = document.querySelector('#rag-backend-view');
const ragBackendStatus = document.querySelector('#rag-backend-status');
const ragBackendTitle = document.querySelector('#rag-backend-title');
const ragBackendDescription = document.querySelector('#rag-backend-description');
const ragGradeLetter = document.querySelector('#rag-grade-letter');
const ragGradeScore = document.querySelector('#rag-grade-score');
const ragGradeExplanation = document.querySelector('#rag-grade-explanation');
const ragOutput = document.querySelector('#rag-output');
const ragUsage = document.querySelector('#rag-usage');
const ragRetrievalLatency = document.querySelector('#rag-retrieval-latency');
const ragGenerationLatency = document.querySelector('#rag-generation-latency');
const ragContextTokensMetric = document.querySelector('#rag-context-tokens');
const ragDocumentCountMetric = document.querySelector('#rag-document-count');
const dmlEntriesTable = document.querySelector('#dml-entries tbody');
const knowledgeStatus = document.querySelector('#knowledge-status');
const ragKnowledgeTable = document.querySelector('#rag-knowledge tbody');
const dmlKnowledgeTable = document.querySelector('#dml-knowledge tbody');
const ragKnowledgeCount = document.querySelector('#knowledge-rag-count');
const ragKnowledgeTokens = document.querySelector('#knowledge-rag-tokens');
const dmlKnowledgeCount = document.querySelector('#knowledge-dml-count');
const dmlKnowledgeTokens = document.querySelector('#knowledge-dml-tokens');
const ragBackendList = document.querySelector('#rag-backends');
const pipelineSteps = document.querySelector('#pipeline-steps');
const baseLatencyMetric = document.querySelector('#metric-base-latency');
const dmlRetrievalLatencyMetric = document.querySelector('#metric-dml-retrieval-latency');
const dmlGenerationLatencyMetric = document.querySelector('#metric-dml-generation-latency');
const ragLatenciesList = document.querySelector('#metric-rag-latencies');
const nimImageInput = document.querySelector('#nim-image');
const ngcApiKeyInput = document.querySelector('#ngc-api-key');
const configureNimButton = document.querySelector('#configure-nim');
const nimStatus = document.querySelector('#nim-status');
const nimDetails = document.querySelector('#nim-details');
const nimConfigSummary = document.querySelector('#nim-config-summary');
const startNimButton = document.querySelector('#start-nim');
const stopNimButton = document.querySelector('#stop-nim');
const nimRuntimeStatus = document.querySelector('#nim-runtime-status');

const API = {
  upload: '/upload',
  compare: '/rag/compare',
  nimOptions: '/nim/options',
  nimConfigure: '/nim/configure',
  nimStart: '/nim/start',
  nimStop: '/nim/stop',
  knowledge: '/knowledge',
};

let nimConfigured = false;
const state = {
  ragBackends: [],
  activeRagId: null,
  lastComparison: null,
  lastTokenBreakdown: [],
  lastRequest: null,
};

if (nimImageInput && configureNimButton && nimStatus) {
  loadNimStatus();
  configureNimButton.addEventListener('click', configureNimEndpoint);
}

if (startNimButton && stopNimButton) {
  startNimButton.disabled = true;
  stopNimButton.disabled = true;
  startNimButton.addEventListener('click', startNimContainer);
  stopNimButton.addEventListener('click', stopNimContainer);
}

if (fileInput) {
  updateSelectedFileList(Array.from(fileInput.files || []));
  fileInput.addEventListener('change', () => {
    updateSelectedFileList(Array.from(fileInput.files || []));
  });
}

if (fileDropZone) {
  ['dragenter', 'dragover'].forEach((eventName) => {
    fileDropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      setDropZoneHighlight(true);
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = 'copy';
      }
    });
  });
  ['dragleave', 'dragend', 'drop'].forEach((eventName) => {
    fileDropZone.addEventListener(eventName, () => setDropZoneHighlight(false));
  });
  fileDropZone.addEventListener('drop', handleFileDrop);
  fileDropZone.addEventListener('click', (event) => {
    if (event.target === fileDropZone) {
      fileInput?.click();
    }
  });
  fileDropZone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      fileInput?.click();
    }
  });
}

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const files = getSelectedUploadFiles();
  if (!files.length) {
    uploadStatus.textContent = 'Please choose at least one file or folder.';
    return;
  }
  const formData = new FormData();
  files.forEach((file) => {
    const relativePath = file.webkitRelativePath || file.name || 'document';
    formData.append('files', file, relativePath);
  });
  uploadStatus.textContent = `Uploading ${files.length} file${files.length === 1 ? '' : 's'}…`;
  try {
    const response = await fetch(API.upload, { method: 'POST', body: formData });
    if (!response.ok) {
      let message = 'Upload failed';
      try {
        const err = await response.json();
        message = err.detail || message;
      } catch (parseErr) {
        console.error('Failed to parse upload error payload', parseErr);
      }
      throw new Error(message);
    }
    const payload = await response.json();
    uploadStatus.textContent = formatUploadStatus(payload);
    fileInput.value = '';
    updateSelectedFileList([]);
    refreshKnowledge();
  } catch (err) {
    console.error(err);
    uploadStatus.textContent = `Error: ${err.message}`;
  }
});

runCompareButton.addEventListener('click', async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    compareStatus.textContent = 'Enter a prompt to compare.';
    return;
  }
  compareStatus.textContent = 'Fetching context and generating responses…';
  resultsPanel.classList.add('hidden');
  try {
    const body = {
      prompt,
      top_k: Number(topKInput.value) || 0,
      max_new_tokens: Number(maxTokensInput.value) || 512,
    };
    state.lastRequest = { ...body };
    const response = await fetch(API.compare, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Compare request failed');
    }
    const payload = await response.json();
    renderResults(payload);
    compareStatus.textContent = 'Done.';
  } catch (err) {
    console.error(err);
    compareStatus.textContent = `Error: ${err.message}`;
  }
});

function renderResults(payload) {
  resultsPanel.classList.remove('hidden');
  const nf = new Intl.NumberFormat('en-US');
  const promptTokens = payload.prompt_tokens_est ?? 0;

  state.lastComparison = payload;
  state.lastTokenBreakdown = Array.isArray(payload.rag_token_breakdown)
    ? payload.rag_token_breakdown
    : [];
  const ragBackends = Array.isArray(payload.rag_backends) ? payload.rag_backends.slice() : [];
  ragBackends.sort((a, b) => {
    const aSeq = Number.isFinite(a?.sequence) ? Number(a.sequence) : Number.MAX_SAFE_INTEGER;
    const bSeq = Number.isFinite(b?.sequence) ? Number(b.sequence) : Number.MAX_SAFE_INTEGER;
    return aSeq - bSeq;
  });
  state.ragBackends = ragBackends;
  if (
    !state.activeRagId ||
    !ragBackends.some((backend) => backend.id === state.activeRagId)
  ) {
    const firstAvailable = ragBackends.find((backend) => backend.available !== false);
    state.activeRagId = firstAvailable?.id ?? (ragBackends[0]?.id || null);
  }
  buildRagResponseTabs(ragBackends);
  renderRagLatencyMetrics(ragBackends);
  setActiveRagBackend(state.activeRagId);

  const activeBackend = getActiveRagBackend();
  const ragTokens = activeBackend?.context_tokens ?? 0;
  const dmlTokens = payload.dml?.context_tokens ?? 0;
  setMetricValue(tokenPromptMetric, promptTokens);
  setMetricValue(tokenRagMetric, ragTokens);
  setMetricValue(tokenDmlMetric, dmlTokens);
  const tokenDelta = dmlTokens && ragTokens ? ragTokens - dmlTokens : 0;
  tokenDeltaMetric.textContent = tokenDelta
    ? `${tokenDelta > 0 ? '−' : '+'}${nf.format(Math.abs(tokenDelta))} tokens`
    : '0';
  dmlFidelityMetric.textContent = formatFloat(payload.dml?.avg_fidelity);
  ragDocsMetric.textContent = nf.format(activeBackend?.documents?.length ?? 0);
  dmlEntriesMetric.textContent = nf.format(payload.dml?.entries?.length ?? 0);

  setLatency(baseLatencyMetric, payload.base?.generation_latency_ms);
  setLatency(dmlRetrievalLatencyMetric, payload.dml?.retrieval_latency_ms);
  setLatency(dmlGenerationLatencyMetric, payload.dml?.generation_latency_ms);

  if (baseOutput) baseOutput.textContent = payload.base?.response || '';
  if (dmlOutput) dmlOutput.textContent = payload.dml?.response || '';

  if (baseUsage) {
    const usageText = formatUsage(payload.base?.usage);
    baseUsage.textContent = usageText;
    baseUsage.classList.toggle('hidden', !usageText);
  }
  if (dmlUsage) {
    const usageText = formatUsage(payload.dml?.usage);
    dmlUsage.textContent = usageText;
    dmlUsage.classList.toggle('hidden', !usageText);
  }

  if (dmlContext)
    dmlContext.textContent = payload.dml?.context || 'No DML memories matched this prompt yet.';
  renderDmlEntries(payload.dml?.entries || []);
  renderDmlSummaries(payload.dml?.entries || []);

  if (insightCopy) {
    insightCopy.textContent = buildInsightCopy({
      promptTokens,
      ragTokens,
      dmlTokens,
      tokenDelta,
      avgFidelity: payload.dml?.avg_fidelity,
      ragCount: activeBackend?.documents?.length || 0,
      dmlCount: payload.dml?.entries?.length || 0,
    });
  }

  renderTokenGraph(state.lastTokenBreakdown);
  renderPipeline(Array.isArray(payload.pipeline_trace) ? payload.pipeline_trace : []);
  refreshKnowledge();
}

function getActiveRagBackend() {
  if (!state.activeRagId) {
    return state.ragBackends[0] || null;
  }
  return state.ragBackends.find((backend) => backend.id === state.activeRagId) || null;
}

function buildRagResponseTabs(backends) {
  if (!ragResponseTabList) {
    return;
  }
  ragResponseTabList.innerHTML = '';
  if (!Array.isArray(backends) || !backends.length) {
    if (ragPanelPlaceholder) {
      ragPanelPlaceholder.classList.remove('hidden');
    }
    if (ragBackendView) {
      ragBackendView.classList.add('hidden');
    }
    return;
  }
  backends.forEach((backend, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'tab-chip';
    if (backend.available === false) {
      button.classList.add('tab-chip--error');
    }
    button.dataset.backendId = backend.id;
    button.setAttribute('role', 'tab');
    const sequenceLabel = Number.isFinite(backend.sequence)
      ? `${Number(backend.sequence)}. `
      : '';
    button.textContent = `${sequenceLabel}${backend.label || backend.id || `Backend ${index + 1}`}`;
    if (backend.strategy) {
      button.title = backend.strategy;
    } else if (backend.error) {
      button.title = backend.error;
    }
    button.addEventListener('click', () => setActiveRagBackend(backend.id));
    ragResponseTabList.appendChild(button);
  });
}

function setActiveRagBackend(backendId) {
  if (!state.ragBackends.length) {
    state.activeRagId = null;
  } else if (backendId && state.ragBackends.some((backend) => backend.id === backendId)) {
    state.activeRagId = backendId;
  } else {
    const firstAvailable = state.ragBackends.find((backend) => backend.available !== false);
    state.activeRagId = firstAvailable?.id ?? state.ragBackends[0].id;
  }
  if (ragResponseTabList) {
    const buttons = Array.from(ragResponseTabList.querySelectorAll('[data-backend-id]'));
    buttons.forEach((button) => {
      const isActive = button.dataset.backendId === state.activeRagId;
      button.classList.toggle('active', isActive);
      button.setAttribute('aria-selected', isActive ? 'true' : 'false');
      button.setAttribute('tabindex', isActive ? '0' : '-1');
    });
  }
  renderActiveRagPanel();
  renderTokenGraph(state.lastTokenBreakdown || []);
  renderRagBackendList(state.ragBackends);
  renderRagLatencyMetrics(state.ragBackends);
  renderAggregateRagCard(getActiveRagBackend());
}

function renderActiveRagPanel() {
  if (!ragResponsePanel || !ragBackendView) {
    return;
  }
  const backend = getActiveRagBackend();
  if (!backend) {
    if (ragPanelPlaceholder) {
      ragPanelPlaceholder.classList.remove('hidden');
    }
    ragBackendView.classList.add('hidden');
    if (ragContext) {
      ragContext.textContent = 'No RAG backend selected.';
    }
    renderRagDocuments([]);
    return;
  }
  if (ragPanelPlaceholder) {
    ragPanelPlaceholder.classList.add('hidden');
  }
  ragBackendView.classList.remove('hidden');
  if (ragBackendTitle) {
    ragBackendTitle.textContent = backend.label || backend.id || 'Backend';
  }
  if (ragBackendDescription) {
    ragBackendDescription.textContent = backend.strategy || '';
    ragBackendDescription.classList.toggle('hidden', !backend.strategy);
  }
  if (backend.available === false) {
    if (ragBackendStatus) {
      ragBackendStatus.textContent = backend.error || 'Backend unavailable.';
      ragBackendStatus.classList.remove('hidden');
    }
    if (ragOutput) {
      ragOutput.textContent = 'No response generated. Backend is unavailable.';
    }
    if (ragUsage) {
      ragUsage.textContent = '';
      ragUsage.classList.add('hidden');
    }
    if (ragGradeLetter) {
      ragGradeLetter.textContent = '–';
    }
    if (ragGradeScore) {
      ragGradeScore.textContent = '–';
    }
    if (ragGradeExplanation) {
      ragGradeExplanation.textContent = backend.error || 'Backend unavailable.';
    }
    setLatency(ragRetrievalLatency, null);
    setLatency(ragGenerationLatency, null);
    setMetricValue(ragContextTokensMetric, 0);
    setMetricValue(ragDocumentCountMetric, 0);
    if (ragContext) {
      ragContext.textContent = 'No RAG context available.';
    }
    renderRagDocuments([]);
    return;
  }
  if (ragBackendStatus) {
    ragBackendStatus.textContent = '';
    ragBackendStatus.classList.add('hidden');
  }
  const usageText = formatUsage(backend.usage);
  if (ragUsage) {
    ragUsage.textContent = usageText;
    ragUsage.classList.toggle('hidden', !usageText);
  }
  if (ragOutput) {
    ragOutput.textContent = backend.response || 'No response generated yet.';
  }
  if (ragGradeLetter) {
    ragGradeLetter.textContent = backend.grade?.grade || '–';
    ragGradeLetter.classList.toggle('grade-na', backend.grade?.grade === 'N/A');
  }
  if (ragGradeScore) {
    const score = Number(backend.grade?.score);
    ragGradeScore.textContent = Number.isFinite(score) ? score.toFixed(2) : '–';
  }
  if (ragGradeExplanation) {
    ragGradeExplanation.textContent = backend.grade?.explanation || 'No evaluation available.';
  }
  setLatency(ragRetrievalLatency, backend.retrieval_latency_ms);
  setLatency(ragGenerationLatency, backend.generation_latency_ms);
  setMetricValue(ragContextTokensMetric, backend.context_tokens ?? 0);
  setMetricValue(ragDocumentCountMetric, backend.documents?.length ?? 0);
  if (ragContext) {
    ragContext.textContent = backend.context || 'No RAG context retrieved for this prompt.';
  }
  renderRagDocuments(backend.documents || []);
}

function renderAggregateRagCard(backend) {
  if (!ragAggregateOutput) {
    return;
  }
  if (!backend) {
    ragAggregateOutput.textContent = 'No RAG responses available yet.';
    if (ragAggregateUsage) {
      ragAggregateUsage.textContent = '';
      ragAggregateUsage.classList.add('hidden');
    }
    if (ragAggregateSource) {
      ragAggregateSource.textContent = 'Run a comparison and choose a backend to view its response here.';
      ragAggregateSource.classList.remove('hidden');
    }
    return;
  }
  if (backend.available === false) {
    ragAggregateOutput.textContent = 'Backend unavailable.';
    if (ragAggregateUsage) {
      ragAggregateUsage.textContent = '';
      ragAggregateUsage.classList.add('hidden');
    }
    if (ragAggregateSource) {
      ragAggregateSource.textContent = backend.error || 'No status details provided.';
      ragAggregateSource.classList.remove('hidden');
    }
    return;
  }
  ragAggregateOutput.textContent = backend.response || 'No response generated yet.';
  if (ragAggregateUsage) {
    const usageText = formatUsage(backend.usage);
    ragAggregateUsage.textContent = usageText;
    ragAggregateUsage.classList.toggle('hidden', !usageText);
  }
  if (ragAggregateSource) {
    const details = [];
    const backendLabel = backend.label || backend.id;
    if (backendLabel) {
      details.push(`Active backend: ${backendLabel}`);
    }
    if (backend.grade?.grade) {
      const score = Number(backend.grade.score);
      const formattedScore = Number.isFinite(score) ? score.toFixed(2) : '–';
      details.push(`Similarity vs DML: ${backend.grade.grade} (${formattedScore})`);
    }
    const note = details.join(' • ');
    ragAggregateSource.textContent = note;
    ragAggregateSource.classList.toggle('hidden', !note);
  }
}

function renderTokenGraph(breakdown) {
  if (!ragTokenGraph) {
    return;
  }
  state.lastTokenBreakdown = Array.isArray(breakdown) ? breakdown : [];
  const entries = state.lastTokenBreakdown
    .slice()
    .sort((a, b) => {
      const aSeq = Number.isFinite(a?.sequence) ? Number(a.sequence) : Number.MAX_SAFE_INTEGER;
      const bSeq = Number.isFinite(b?.sequence) ? Number(b.sequence) : Number.MAX_SAFE_INTEGER;
      if (aSeq !== bSeq) return aSeq - bSeq;
      return (Number(b?.tokens) || 0) - (Number(a?.tokens) || 0);
    });
  ragTokenGraph.innerHTML = '';
  if (!entries.length) {
    const emptyMessage = document.createElement('p');
    emptyMessage.className = 'empty-state';
    emptyMessage.textContent = 'No RAG retrievals yet.';
    ragTokenGraph.appendChild(emptyMessage);
    return;
  }
  const nf = new Intl.NumberFormat('en-US');
  const maxTokens = Math.max(...entries.map((entry) => Number(entry.tokens) || 0), 1);
  entries.forEach((entry) => {
    const bar = document.createElement('div');
    bar.className = 'token-graph-bar';
    if (entry.id === state.activeRagId) {
      bar.classList.add('active');
    }
    const label = document.createElement('div');
    label.className = 'token-graph-label';
    const title = document.createElement('strong');
    const sequenceLabel = Number.isFinite(entry.sequence)
      ? `${Number(entry.sequence)}. `
      : '';
    title.textContent = `${sequenceLabel}${entry.label || entry.id || 'Backend'}`;
    label.appendChild(title);
    if (entry.strategy) {
      const strategy = document.createElement('span');
      strategy.className = 'token-graph-strategy';
      strategy.textContent = entry.strategy;
      label.appendChild(strategy);
    }
    if (entry.strategy) {
      bar.title = entry.strategy;
    }
    const meter = document.createElement('div');
    meter.className = 'token-graph-meter';
    const fill = document.createElement('div');
    fill.className = 'token-graph-fill';
    const ratio = Math.min(100, ((Number(entry.tokens) || 0) / maxTokens) * 100);
    fill.style.width = `${ratio}%`;
    meter.appendChild(fill);
    const value = document.createElement('span');
    value.className = 'token-graph-value';
    value.textContent = `${nf.format(Number(entry.tokens) || 0)} tok`;
    bar.appendChild(label);
    bar.appendChild(meter);
    bar.appendChild(value);
    ragTokenGraph.appendChild(bar);
  });
}

function renderPipeline(steps) {
  if (!pipelineSteps) {
    return;
  }
  pipelineSteps.innerHTML = '';
  const entries = Array.isArray(steps) ? steps.slice() : [];
  if (!entries.length) {
    pipelineSteps.classList.add('hidden');
    return;
  }
  pipelineSteps.classList.remove('hidden');
  const stageLabels = {
    base: 'Base model',
    rag: 'RAG',
    dml: 'DML',
  };
  entries
    .sort((a, b) => {
      const aSeq = Number.isFinite(a?.sequence) ? Number(a.sequence) : Number(a?.step) || Number.MAX_SAFE_INTEGER;
      const bSeq = Number.isFinite(b?.sequence) ? Number(b.sequence) : Number(b?.step) || Number.MAX_SAFE_INTEGER;
      return aSeq - bSeq;
    })
    .forEach((entry, index) => {
      const item = document.createElement('li');
      item.className = 'pipeline-step';
      const badge = document.createElement('span');
      badge.className = 'pipeline-step__index';
      const stepNumber = Number.isFinite(entry.sequence)
        ? Number(entry.sequence)
        : Number.isFinite(entry.step)
        ? Number(entry.step)
        : index + 1;
      badge.textContent = String(stepNumber);
      const label = document.createElement('span');
      label.className = 'pipeline-step__label';
      const stage = entry.stage && stageLabels[entry.stage] ? stageLabels[entry.stage] : entry.stage;
      const detail = entry.label || entry.id;
      label.textContent = detail ? `${stage || 'Step'} · ${detail}` : stage || 'Step';
      item.appendChild(badge);
      item.appendChild(label);
      pipelineSteps.appendChild(item);
    });
}

function getSelectedUploadFiles() {
  if (!fileInput || !fileInput.files) {
    return [];
  }
  return Array.from(fileInput.files);
}

function setDropZoneHighlight(active) {
  if (!fileDropZone) {
    return;
  }
  fileDropZone.classList.toggle('drag-active', Boolean(active));
}

function handleFileDrop(event) {
  event.preventDefault();
  const transfer = event.dataTransfer;
  if (!transfer) {
    return;
  }
  const droppedFiles = Array.from(transfer.files || []);
  if (!droppedFiles.length && transfer.items) {
    const items = Array.from(transfer.items);
    items
      .filter((item) => item.kind === 'file')
      .map((item) => item.getAsFile())
      .filter(Boolean)
      .forEach((file) => droppedFiles.push(file));
  }
  if (!droppedFiles.length) {
    return;
  }
  setFileSelection(droppedFiles);
  setDropZoneHighlight(false);
}

function setFileSelection(files) {
  if (!fileInput) {
    return;
  }
  if (typeof DataTransfer === 'undefined') {
    console.warn('DataTransfer API not available in this browser; drag and drop selection skipped.');
    return;
  }
  const transfer = new DataTransfer();
  files.forEach((file) => {
    if (file) {
      transfer.items.add(file);
    }
  });
  fileInput.files = transfer.files;
  fileInput.dispatchEvent(new Event('change', { bubbles: true }));
}

function updateSelectedFileList(files) {
  if (!selectedFilesList) {
    return;
  }
  selectedFilesList.innerHTML = '';
  if (!files.length) {
    const empty = document.createElement('li');
    empty.className = 'selected-files-empty';
    empty.textContent = 'No files selected yet.';
    selectedFilesList.appendChild(empty);
    return;
  }
  files.forEach((file) => {
    const item = document.createElement('li');
    item.className = 'selected-file';
    if (file.webkitRelativePath || file.name) {
      const name = document.createElement('span');
      name.className = 'file-name';
      name.textContent = file.webkitRelativePath || file.name;
      item.appendChild(name);
    }
    const size = document.createElement('span');
    size.className = 'file-size';
    size.textContent = formatFileSize(file.size);
    item.appendChild(size);
    selectedFilesList.appendChild(item);
  });
}

function formatFileSize(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return '';
  }
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const precision = unitIndex === 0 ? 0 : size < 10 ? 1 : 0;
  return `${size.toFixed(precision)} ${units[unitIndex]}`;
}

function formatUploadStatus(payload) {
  if (!payload) {
    return 'Upload complete.';
  }
  const nf = new Intl.NumberFormat('en-US');
  const files = Number(payload.files_ingested ?? payload.files ?? 0);
  const documents = Number(payload.documents ?? 0);
  const chunks = Number(payload.chunks ?? 0);
  const tokens = Number(payload.tokens ?? 0);
  const skipped = Number(payload.skipped_files ?? payload.skipped ?? 0);
  const pieces = [];
  if (documents || files) {
    pieces.push(
      `Ingested ${nf.format(documents)} document${documents === 1 ? '' : 's'} from ${nf.format(files)} file${files === 1 ? '' : 's'}.`
    );
  }
  if (chunks) {
    pieces.push(
      `${nf.format(chunks)} chunk${chunks === 1 ? '' : 's'} (~${nf.format(tokens)} tokens) indexed across RAG and the DML.`
    );
  }
  if (skipped) {
    pieces.push(`${nf.format(skipped)} file${skipped === 1 ? ' was' : 's were'} skipped.`);
  }
  if (!pieces.length) {
    pieces.push('Upload complete.');
  }
  if (Array.isArray(payload.errors) && payload.errors.length) {
    const preview = payload.errors.slice(0, 2).join(' ');
    pieces.push(`Warnings: ${preview}${payload.errors.length > 2 ? '…' : ''}`);
    console.warn('Upload warnings:', payload.errors);
  }
  return pieces.join(' ');
}

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function setMetricValue(target, value) {
  if (!target) return;
  const nf = new Intl.NumberFormat('en-US');
  const num = Number(value || 0);
  target.textContent = nf.format(num);
}

function setLatency(target, value) {
  if (!target) return;
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    target.textContent = '–';
    return;
  }
  target.textContent = `${Math.round(numeric)} ms`;
}

function formatLatencyValue(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return '–';
  }
  return `${Math.round(numeric)} ms`;
}

function renderRagLatencyMetrics(backends) {
  if (!ragLatenciesList) {
    return;
  }
  ragLatenciesList.innerHTML = '';
  if (!Array.isArray(backends) || !backends.length) {
    const note = document.createElement('p');
    note.className = 'metric-note';
    note.textContent = 'No RAG responses yet.';
    ragLatenciesList.appendChild(note);
    return;
  }
  backends.forEach((backend) => {
    const item = document.createElement('div');
    item.className = 'metric-subitem';
    if (backend.id === state.activeRagId) {
      item.classList.add('active');
    }
    if (backend.available === false) {
      item.classList.add('unavailable');
    }
    const name = document.createElement('span');
    name.textContent = backend.label || backend.id || 'Backend';
    const value = document.createElement('span');
    if (backend.available === false) {
      value.textContent = backend.error ? `Offline (${backend.error})` : 'Offline';
    } else {
      const retrieval = formatLatencyValue(backend.retrieval_latency_ms);
      const generation = formatLatencyValue(backend.generation_latency_ms);
      value.textContent = `${retrieval} / ${generation}`;
    }
    item.appendChild(name);
    item.appendChild(value);
    ragLatenciesList.appendChild(item);
  });
}

function formatFloat(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '–';
  }
  return Number(value).toFixed(2);
}

function formatUsage(usage) {
  if (!usage) {
    return '';
  }
  const nf = new Intl.NumberFormat('en-US');
  const prompt = usage.prompt_tokens ?? usage.promptTokens;
  const completion = usage.completion_tokens ?? usage.completionTokens;
  const total = usage.total_tokens ?? usage.totalTokens;
  const pieces = [];
  if (prompt !== undefined) pieces.push(`Prompt: ${nf.format(prompt)}`);
  if (completion !== undefined) pieces.push(`Completion: ${nf.format(completion)}`);
  if (total !== undefined) pieces.push(`Total: ${nf.format(total)}`);
  return pieces.length ? pieces.join(' | ') : '';
}

function renderRagDocuments(documents) {
  if (!ragDocumentsTable) {
    return;
  }
  ragDocumentsTable.innerHTML = '';
  if (!documents.length) {
    const emptyRow = document.createElement('tr');
    emptyRow.innerHTML = '<td colspan="4">No matching RAG documents for this backend yet.</td>';
    ragDocumentsTable.appendChild(emptyRow);
    return;
  }
  documents.forEach((doc, idx) => {
    const row = document.createElement('tr');
    const source = doc.meta?.doc_path || doc.meta?.source || 'uploaded document';
    row.innerHTML = `
      <td>${idx + 1}</td>
      <td>${Number(doc.score ?? 0).toFixed(3)}</td>
      <td>${doc.tokens ?? 0}</td>
      <td>${escapeHtml(source)}</td>
    `;
    ragDocumentsTable.appendChild(row);
  });
}

function renderDmlEntries(entries) {
  if (!dmlEntriesTable) {
    return;
  }
  dmlEntriesTable.innerHTML = '';
  if (!entries.length) {
    const emptyRow = document.createElement('tr');
    emptyRow.innerHTML = '<td colspan="5">No DML memories retrieved.</td>';
    dmlEntriesTable.appendChild(emptyRow);
    return;
  }
  entries.forEach((entry) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${entry.id}</td>
      <td>L${entry.level}</td>
      <td>${Number(entry.fidelity ?? 0).toFixed(2)}</td>
      <td>${entry.tokens ?? 0}</td>
      <td>${escapeHtml(entry.summary ?? '')}</td>
    `;
    dmlEntriesTable.appendChild(row);
  });
}

function renderDmlSummaries(entries) {
  if (!dmlSummaryList) {
    return;
  }
  dmlSummaryList.innerHTML = '';
  if (!entries.length) {
    const emptyItem = document.createElement('li');
    emptyItem.className = 'summary-empty';
    emptyItem.textContent = 'No DML memories retrieved.';
    dmlSummaryList.appendChild(emptyItem);
    return;
  }
  entries.forEach((entry) => {
    const item = document.createElement('li');
    const meta = document.createElement('div');
    meta.className = 'summary-meta';
    const details = [];
    details.push(entry.level !== undefined && entry.level !== null ? `L${entry.level}` : 'L?');
    if (entry.fidelity !== undefined && entry.fidelity !== null) {
      details.push(`f=${Number(entry.fidelity).toFixed(2)}`);
    }
    if (entry.tokens !== undefined && entry.tokens !== null) {
      details.push(`${entry.tokens} tok`);
    }
    meta.textContent = details.join(' • ');
    const summary = document.createElement('p');
    summary.textContent = entry.summary || 'No summary available.';
    item.appendChild(meta);
    item.appendChild(summary);
    dmlSummaryList.appendChild(item);
  });
}

async function refreshKnowledge() {
  if (!knowledgeStatus) {
    return;
  }
  knowledgeStatus.textContent = 'Refreshing knowledge summaries…';
  try {
    const response = await fetch(API.knowledge);
    if (!response.ok) {
      throw new Error('Failed to load knowledge summaries');
    }
    const payload = await response.json();
    renderKnowledge(payload);
    const ragCount = payload.rag?.count || 0;
    const dmlCount = payload.dml?.count || 0;
    if (!ragCount && !dmlCount) {
      knowledgeStatus.textContent = 'No documents have been ingested into the knowledge bases yet.';
    } else if (payload.dml?.truncated) {
      const shown = payload.dml?.entries?.length || 0;
      const limit = payload.dml?.display_limit || shown;
      knowledgeStatus.textContent = `Showing ${shown} of ${dmlCount} DML memories (limited to ${limit} for performance).`;
    } else {
      knowledgeStatus.textContent = '';
    }
  } catch (err) {
    console.error(err);
    knowledgeStatus.textContent = `Error: ${err.message}`;
  }
}

function renderKnowledge(payload) {
  if (!payload) {
    return;
  }
  setMetricValue(ragKnowledgeCount, payload.rag?.count ?? 0);
  setMetricValue(ragKnowledgeTokens, payload.rag?.total_tokens ?? 0);
  setMetricValue(dmlKnowledgeCount, payload.dml?.count ?? 0);
  setMetricValue(dmlKnowledgeTokens, payload.dml?.total_tokens ?? 0);
  renderRagKnowledge(payload.rag?.documents || []);
  renderDmlKnowledge(payload.dml);
  renderRagBackendList(payload.rag?.backends || []);
}

function renderRagKnowledge(documents) {
  if (!ragKnowledgeTable) {
    return;
  }
  ragKnowledgeTable.innerHTML = '';
  if (!documents.length) {
    const row = document.createElement('tr');
    row.innerHTML = '<td colspan="3">No documents ingested yet.</td>';
    ragKnowledgeTable.appendChild(row);
    return;
  }
  documents.forEach((doc) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${doc.index}</td>
      <td>${doc.tokens ?? 0}</td>
      <td>${escapeHtml(doc.source || 'uploaded document')}</td>
    `;
    ragKnowledgeTable.appendChild(row);
  });
}

function renderDmlKnowledge(dml) {
  if (!dmlKnowledgeTable) {
    return;
  }
  dmlKnowledgeTable.innerHTML = '';
  const entries = dml?.entries || [];
  if (!entries.length) {
    const row = document.createElement('tr');
    row.innerHTML = '<td colspan="5">No DML memories stored yet.</td>';
    dmlKnowledgeTable.appendChild(row);
    return;
  }
  entries.forEach((entry) => {
    const summary = truncateText(entry.summary || '');
    const fidelity = entry.fidelity !== undefined && entry.fidelity !== null
      ? Number(entry.fidelity).toFixed(2)
      : '–';
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${entry.id}</td>
      <td>L${entry.level ?? '?'}</td>
      <td>${fidelity}</td>
      <td>${entry.tokens ?? 0}</td>
      <td>${escapeHtml(summary || 'No summary available.')}</td>
    `;
    dmlKnowledgeTable.appendChild(row);
  });
  if (dml?.truncated && (dml.count ?? 0) > entries.length) {
    const infoRow = document.createElement('tr');
    infoRow.className = 'table-note';
    const remaining = (dml.count ?? 0) - entries.length;
    infoRow.innerHTML = `
      <td colspan="5">${remaining} additional memories are hidden for performance. Use retrieval reports for full context.</td>
    `;
    dmlKnowledgeTable.appendChild(infoRow);
  }
}

function renderRagBackendList(backends) {
  if (!ragBackendList) {
    return;
  }
  ragBackendList.innerHTML = '';
  if (!Array.isArray(backends) || !backends.length) {
    const empty = document.createElement('li');
    empty.className = 'rag-backend-empty';
    empty.textContent = 'No RAG backends are configured.';
    ragBackendList.appendChild(empty);
    return;
  }
  backends.forEach((backend) => {
    const item = document.createElement('li');
    item.className = 'rag-backend-item';
    if (backend.id === state.activeRagId) {
      item.classList.add('active');
    }
    if (backend.available === false) {
      item.classList.add('unavailable');
    }
    const title = document.createElement('div');
    title.className = 'rag-backend-title';
    title.textContent = backend.label || backend.id || 'Backend';
    const status = document.createElement('span');
    status.className = 'rag-backend-status';
    status.textContent = backend.available === false ? 'Offline' : 'Ready';
    title.appendChild(status);
    item.appendChild(title);
    if (backend.strategy) {
      const description = document.createElement('p');
      description.className = 'rag-backend-description';
      description.textContent = backend.strategy;
      item.appendChild(description);
    }
    if (backend.available === false && backend.error) {
      const error = document.createElement('p');
      error.className = 'rag-backend-error';
      error.textContent = backend.error;
      item.appendChild(error);
    }
    ragBackendList.appendChild(item);
  });
}

function truncateText(value, limit = 160) {
  if (!value) {
    return '';
  }
  const normalized = value.replace(/\s+/g, ' ').trim();
  if (normalized.length <= limit) {
    return normalized;
  }
  return `${normalized.slice(0, limit - 1)}…`;
}

function buildInsightCopy({ promptTokens, ragTokens, dmlTokens, tokenDelta, avgFidelity, ragCount, dmlCount }) {
  if (!ragTokens && !dmlTokens) {
    return 'No retrieval context has been generated yet. Upload documents to populate RAG and the DML.';
  }
  const nf = new Intl.NumberFormat('en-US');
  const parts = [];
  if (ragCount) {
    parts.push(`RAG contributed ${nf.format(ragCount)} document chunk${ragCount === 1 ? '' : 's'} totalling ${nf.format(ragTokens)} tokens.`);
  }
  if (dmlCount) {
    const fidelityText = avgFidelity !== undefined && avgFidelity !== null ? ` with an average fidelity of ${Number(avgFidelity).toFixed(2)}` : '';
    parts.push(`The Daystrom Memory Lattice surfaced ${nf.format(dmlCount)} memory node${dmlCount === 1 ? '' : 's'}${fidelityText} and ${nf.format(dmlTokens)} contextual tokens.`);
  }
  if (tokenDelta) {
    const direction = tokenDelta > 0 ? 'fewer' : 'more';
    parts.push(`Compared to RAG alone, the DML context uses ${nf.format(Math.abs(tokenDelta))} ${direction} tokens, highlighting the distinct retrieval category.`);
  }
  parts.push(`The user prompt spans approximately ${nf.format(promptTokens)} tokens.`);
  return parts.join(' ');
}

async function loadNimStatus() {
  nimStatus.textContent = 'Checking NVIDIA NIM configuration…';
  try {
    const response = await fetch(API.nimOptions);
    if (!response.ok) {
      throw new Error('Failed to load NIM status');
    }
    const payload = await response.json();
    nimConfigured = Boolean(payload.current);
    if (nimImageInput && !nimImageInput.value && payload.default?.image) {
      nimImageInput.value = payload.default.image;
    }
    updateRuntimeStatus(payload.runtime, nimConfigured);
    if (payload.current) {
      renderNimSummary(payload.current, 'Using previously configured NIM.', payload);
    } else {
      nimStatus.textContent = 'Input the NVIDIA NIM image and provide your NGC API key to begin.';
    }
  } catch (err) {
    console.error(err);
    nimStatus.textContent = `Error: ${err.message}`;
  }
}

async function configureNimEndpoint() {
  const nimImage = (nimImageInput.value || '').trim();
  const apiKey = (ngcApiKeyInput.value || '').trim();
  if (!apiKey) {
    nimStatus.textContent = 'Enter your NGC API key to continue.';
    return;
  }
  configureNimButton.disabled = true;
  nimStatus.textContent = 'Pulling NIM container and configuring service…';
  try {
    const response = await fetch(API.nimConfigure, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nim_image: nimImage, api_key: apiKey }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to configure NIM');
    }
    const payload = await response.json();
    ngcApiKeyInput.value = '';
    const message = buildStatusMessage(payload);
    nimConfigured = true;
    renderNimSummary(payload.nim, message, payload);
  } catch (err) {
    console.error(err);
    nimStatus.textContent = `Error: ${err.message}`;
  } finally {
    configureNimButton.disabled = false;
  }
}

function buildStatusMessage(payload) {
  if (!payload) {
    return 'Configured.';
  }
  if (payload.pull_status === 'ok') {
    return `Configured ${payload.nim.label}. Docker image pulled successfully.`;
  }
  if (payload.pull_status === 'skipped') {
    return `Configured ${payload.nim.label}. Docker image pull skipped: ${payload.logs?.[0] || 'Docker unavailable.'}`;
  }
  return `Configured ${payload.nim.label} with warnings.`;
}

function renderNimSummary(nim, message, payload) {
  nimStatus.textContent = message;
  if (!nim) {
    nimDetails.classList.add('hidden');
    nimConfigSummary.textContent = '';
    updateRuntimeStatus(payload?.runtime, nimConfigured, message || payload?.message, payload?.logs);
    return;
  }
  const summary = {
    id: nim.id,
    label: nim.label,
    model_name: nim.model_name,
    api_base: nim.api_base,
    image: nim.image,
    pull_status: payload?.pull_status,
  };
  if (Array.isArray(payload?.logs) && payload.logs.length) {
    summary.logs = payload.logs;
  }
  nimConfigSummary.textContent = JSON.stringify(summary, null, 2);
  nimDetails.classList.remove('hidden');
  updateRuntimeStatus(payload?.runtime, nimConfigured, message || payload?.message, payload?.logs);
}

function updateRuntimeStatus(runtime, isConfigured, message, logs) {
  if (!nimRuntimeStatus) {
    return;
  }
  const lines = [];
  if (message) {
    lines.push(message);
  }
  if (!isConfigured) {
    lines.push('Configure a NIM to enable runtime controls.');
    if (startNimButton) startNimButton.disabled = true;
    if (stopNimButton) stopNimButton.disabled = true;
    if (Array.isArray(logs) && logs.length) {
      lines.push(...logs);
    }
    nimRuntimeStatus.textContent = lines.join('\n');
    return;
  }
  if (!runtime) {
    lines.push('Runtime status unavailable.');
    if (startNimButton) startNimButton.disabled = false;
    if (stopNimButton) stopNimButton.disabled = true;
    if (Array.isArray(logs) && logs.length) {
      lines.push(...logs);
    }
    nimRuntimeStatus.textContent = lines.join('\n');
    return;
  }
  if (runtime.docker_available === false) {
    lines.push('Docker is not available on this server.');
  }
  if (runtime.running) {
    lines.push(runtime.healthy ? 'NIM container is running.' : 'NIM container is starting…');
  } else {
    lines.push('NIM container is stopped.');
  }
  if (runtime.container_id) {
    const containerIdStr = String(runtime.container_id);
    const shortId = containerIdStr.slice(0, 12);
    const truncated = containerIdStr.length > shortId.length ? '…' : '';
    lines.push(`Container ID: ${shortId}${truncated}`);
  }
  if (Array.isArray(logs) && logs.length) {
    lines.push(...logs);
  }
  const dockerMissing = runtime.docker_available === false;
  if (startNimButton) {
    startNimButton.disabled = !isConfigured || runtime.running || dockerMissing;
  }
  if (stopNimButton) {
    stopNimButton.disabled = !isConfigured || !runtime.running;
  }
  nimRuntimeStatus.textContent = lines.join('\n');
}

async function startNimContainer() {
  if (!nimConfigured) {
    updateRuntimeStatus(null, false, 'Configure a NIM before starting it.');
    return;
  }
  updateRuntimeStatus({ running: false }, true, 'Starting NIM…');
  if (startNimButton) startNimButton.disabled = true;
  if (stopNimButton) stopNimButton.disabled = true;
  try {
    const response = await fetch(API.nimStart, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to start NIM');
    }
    const payload = await response.json();
    updateRuntimeStatus(payload.runtime, nimConfigured, payload.message, payload.logs);
  } catch (err) {
    console.error(err);
    updateRuntimeStatus(null, nimConfigured, `Error: ${err.message}`);
  }
}

async function stopNimContainer() {
  updateRuntimeStatus({ running: true }, nimConfigured, 'Stopping NIM…');
  if (startNimButton) startNimButton.disabled = true;
  if (stopNimButton) stopNimButton.disabled = true;
  try {
    const response = await fetch(API.nimStop, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to stop NIM');
    }
    const payload = await response.json();
    updateRuntimeStatus(payload.runtime, nimConfigured, payload.message, payload.logs);
  } catch (err) {
    console.error(err);
    updateRuntimeStatus(null, nimConfigured, `Error: ${err.message}`);
  }
}

refreshKnowledge();
