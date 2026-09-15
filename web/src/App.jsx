import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

// axios reports every failure the same way, so a 500 that explains itself used
// to surface as "cannot reach the API". Separate the two cases.
function describeApiError(error, fallback) {
  if (error.response) {
    return error.response.data?.detail ?? `${fallback} (HTTP ${error.response.status})`;
  }
  if (error.request) {
    return `Cannot reach the SpeechScribe API at ${API_BASE}. Is it running?`;
  }
  return error.message || fallback;
}

const featureCards = [
  {
    id: 'transcription',
    title: 'Transcription',
    description: 'Transcribe audio files or live recordings.',
    icon: 'description'
  },
  {
    id: 'live-microphone',
    title: 'Live Microphone',
    description: 'Real-time speech-to-text from microphone.',
    icon: 'mic'
  },
  {
    id: 'meeting-mode',
    title: 'Meeting Mode',
    description: 'Capture and transcribe meetings with speaker diarization.',
    icon: 'groups'
  },
  {
    id: 'voice-synthesis',
    title: 'Voice Synthesis',
    description: 'Generate speech from text using TTS models.',
    icon: 'volume_up'
  },
  {
    id: 'chat',
    title: 'AI Chat',
    description: 'Chat with a local LLM about your transcripts.',
    icon: 'chat'
  }
];

const modelOptions = ['whisper-large', 'whisper-medium', 'whisper-small'];

function App() {
  const [selectedPanel, setSelectedPanel] = useState('transcription');
  const [theme, setTheme] = useState('light');
  const [selectedModel, setSelectedModel] = useState(modelOptions[0]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [transcriptionResult, setTranscriptionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState('Drop audio files here or use the controls.');
  const [recording, setRecording] = useState(false);
  const [modelStatus, setModelStatus] = useState({});
  const [gpuUsage, setGpuUsage] = useState(0);
  const [plugins, setPlugins] = useState([]);
  const [waveformData, setWaveformData] = useState([]);
  const [showPluginPanel, setShowPluginPanel] = useState(false);
  const [apiReachable, setApiReachable] = useState(true);
  const fileInputRef = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // Each probe is independent: an endpoint that is not implemented yet must
  // not stop the others from reporting.
  const refreshPlugins = useCallback(async () => {
    try {
      const { data } = await axios.get(`${API_BASE}/plugins`);
      setPlugins(Array.isArray(data) ? data : []);
      setApiReachable(true);
      return Array.isArray(data) ? data : [];
    } catch (error) {
      console.error('Failed to fetch plugins:', error);
      setApiReachable(false);
      return [];
    }
  }, []);

  useEffect(() => {
    const fetchStatus = async () => {
      await refreshPlugins();

      try {
        const { data } = await axios.get(`${API_BASE}/models/status`);
        setModelStatus(data);
      } catch {
        setModelStatus({});
      }

      try {
        const { data } = await axios.get(`${API_BASE}/gpu/usage`);
        setGpuUsage(data.usage ?? 0);
      } catch {
        setGpuUsage(0);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, [refreshPlugins]);

  const dropHint = useMemo(() => (selectedFile ? selectedFile.name : 'Drop audio here or use the buttons to record/upload.'), [selectedFile]);

  const handleDrop = useCallback((event) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      setSelectedFile(file);
      setFeedback(`Ready to transcribe ${file.name}`);
    }
  }, []);

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
  }, []);

  const handleFileChange = useCallback((event) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setFeedback(`Ready to transcribe ${file.name}`);
    }
  }, []);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleTranscribe = async () => {
    if (!selectedFile) {
      setFeedback('Select a file before running transcription.');
      return;
    }

    setLoading(true);
    setFeedback('Transcribing audio...');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model', selectedModel);

      const response = await axios.post(`${API_BASE}/transcribe`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setTranscriptionResult(response.data);
      setFeedback('Transcription complete.');
    } catch (error) {
      console.error(error);
      setFeedback(describeApiError(error, 'Transcription failed.'));
    } finally {
      setLoading(false);
    }
  };

  const handleRecord = () => {
    setRecording((prev) => {
      const newRecording = !prev;
      if (newRecording) {
        // Generate sample waveform data
        const sampleData = Array.from({ length: 100 }, () => Math.random() * 255);
        setWaveformData(sampleData);
      }
      setFeedback((state) => (state.includes('Recording') ? 'Drop audio files here or use the controls.' : 'Recording... click stop when finished.'));
      return newRecording;
    });
  };

  const handlePanelContent = useMemo(() => {
    if (selectedPanel === 'transcription') {
      return (
        <TranscribePanel
          dropHint={dropHint}
          feedback={feedback}
          handleDrop={handleDrop}
          handleDragOver={handleDragOver}
          handleFileChange={handleFileChange}
          handleUploadClick={handleUploadClick}
          handleTranscribe={handleTranscribe}
          handleRecord={handleRecord}
          loading={loading}
          recording={recording}
          selectedModel={selectedModel}
          setSelectedModel={setSelectedModel}
          fileInputRef={fileInputRef}
          transcriptionResult={transcriptionResult}
          waveformData={waveformData}
          setWaveformData={setWaveformData}
        />
      );
    }

    if (selectedPanel === 'live-microphone') {
      return <LiveMicrophonePanel recording={recording} setRecording={setRecording} waveformData={waveformData} setWaveformData={setWaveformData} />;
    }

    if (selectedPanel === 'meeting-mode') {
      return <MeetingModePanel />;
    }

    if (selectedPanel === 'voice-synthesis') {
      return <VoiceSynthesisPanel />;
    }

    if (selectedPanel === 'chat') {
      return <ChatPanel />;
    }

    return (
      <div className="card shadow-sm border-0">
        <div className="card-body">
          <h4 className="card-title text-capitalize">{selectedPanel.replace('-', ' ')}</h4>
          <p className="card-text text-muted">
            Feature coming soon.
          </p>
        </div>
      </div>
    );
  }, [selectedPanel, dropHint, feedback, handleDrop, handleDragOver, handleFileChange, handleUploadClick, handleTranscribe, loading, recording, selectedModel, transcriptionResult, fileInputRef, waveformData]);

  return (
    <div className="app-shell">
      <nav className="navbar navbar-expand-lg navbar-dark bg-dark px-4">
        <div className="container-fluid">
          <span className="navbar-brand fs-4">SpeechScribe</span>
          <div className="d-flex align-items-center gap-3">
            <div className="d-flex align-items-center gap-2">
              <span
                className={`badge ${apiReachable ? 'bg-success' : 'bg-danger'}`}
                title={apiReachable ? `${plugins.length} plugin(s) loaded` : `Cannot reach the API at ${API_BASE}`}
              >
                <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>
                  {apiReachable ? 'check_circle' : 'error'}
                </span>
                {apiReachable ? 'Models' : 'API offline'}
              </span>
              <span className="badge bg-info" title={`GPU Usage: ${gpuUsage}%`}>
                <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>memory</span>
                {gpuUsage}%
              </span>
              <button className="btn btn-outline-light btn-sm" title="Plugin Management" onClick={() => setShowPluginPanel(!showPluginPanel)}>
                <span className="material-symbols-outlined">extension</span>
              </button>
            </div>
            <select
              className="form-select form-select-sm"
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
            >
              {modelOptions.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
            <button
              className="btn btn-outline-light btn-sm"
              onClick={() => setTheme((prev) => (prev === 'light' ? 'dark' : 'light'))}
            >
              <span className="material-symbols-outlined me-1">{theme === 'light' ? 'dark_mode' : 'light_mode'}</span>
              {theme === 'light' ? 'Dark' : 'Light'}
            </button>
          </div>
        </div>
      </nav>

      <div className="d-flex flex-grow-1">
        <aside className="sidebar bg-white shadow-sm">
          <div className="p-3">
            <h6 className="text-uppercase text-muted fs-7">Quick actions</h6>
            <div className="row gy-3 mt-2">
              {featureCards.map((card) => (
                <div className="col-12" key={card.id}>
                  <button
                    className={`card border-0 shadow-sm w-100 text-start ${selectedPanel === card.id ? 'card-active' : ''}`}
                    onClick={() => setSelectedPanel(card.id)}
                  >
                    <div className="card-body p-3 d-flex align-items-center gap-3">
                      <span className="material-symbols-outlined fs-4 text-primary">{card.icon}</span>
                      <div>
                        <p className="mb-1 fw-semibold">{card.title}</p>
                        <p className="mb-0 text-muted small">{card.description}</p>
                      </div>
                    </div>
                  </button>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <main className="flex-grow-1 p-4">
          <div className="container-fluid">
            {handlePanelContent}
          </div>
        </main>
      </div>

      {showPluginPanel && (
        <div className="plugin-panel-overlay" onClick={() => setShowPluginPanel(false)}>
          <div className="plugin-panel" onClick={(e) => e.stopPropagation()}>
            <div className="plugin-panel-header">
              <h5>Plugin Management</h5>
              <button className="btn-close" onClick={() => setShowPluginPanel(false)} aria-label="Close plugin panel"></button>
            </div>
            <div className="plugin-panel-body">
              {!apiReachable && (
                <div className="alert alert-danger">
                  Cannot reach the SpeechScribe API at <code>{API_BASE}</code>. Start it with{' '}
                  <code>python src/speechscribe/api/main.py</code>.
                </div>
              )}
              {plugins.length > 0 ? (
                plugins.map((plugin) => (
                  <PluginCard key={plugin.plugin_id} plugin={plugin} onSaved={refreshPlugins} />
                ))
              ) : (
                <p className="text-muted">
                  {apiReachable ? 'No plugins discovered in speechscribe/plugins.' : 'Plugin list unavailable.'}
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function PluginCard({ plugin, onSaved }) {
  const [expanded, setExpanded] = useState(false);
  const [values, setValues] = useState(plugin.settings ?? {});
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null);
  const [health, setHealth] = useState(null);

  const schema = plugin.settings_schema ?? [];

  useEffect(() => {
    setValues(plugin.settings ?? {});
  }, [plugin.settings]);

  const checkHealth = useCallback(async () => {
    setHealth({ checking: true });
    try {
      const { data } = await axios.get(`${API_BASE}/plugins/${plugin.plugin_id}/health`);
      setHealth(data);
    } catch (error) {
      setHealth({ available: false, detail: error.response?.data?.detail ?? 'Health check failed' });
    }
  }, [plugin.plugin_id]);

  const handleChange = (key, value) => {
    setValues((prev) => ({ ...prev, [key]: value }));
    setStatus(null);
  };

  const handleSave = async () => {
    setSaving(true);
    setStatus(null);
    try {
      await axios.put(`${API_BASE}/plugins/${plugin.plugin_id}/settings`, { settings: values });
      setStatus({ type: 'success', text: 'Settings saved.' });
      await onSaved?.();
    } catch (error) {
      setStatus({ type: 'danger', text: error.response?.data?.detail ?? 'Failed to save settings.' });
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    setSaving(true);
    setStatus(null);
    try {
      const { data } = await axios.delete(`${API_BASE}/plugins/${plugin.plugin_id}/settings`);
      setValues(data.settings ?? {});
      setStatus({ type: 'success', text: 'Reverted to defaults.' });
      await onSaved?.();
    } catch (error) {
      setStatus({ type: 'danger', text: error.response?.data?.detail ?? 'Failed to reset settings.' });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="plugin-item">
      <div className="d-flex justify-content-between align-items-start gap-3">
        <div>
          <h6 className="mb-1">{plugin.name}</h6>
          <p className="mb-1 text-muted small">
            {plugin.type} • v{plugin.version} • <code>{plugin.plugin_id}</code>
          </p>
          {plugin.description && <p className="mb-1 small">{plugin.description}</p>}
          {plugin.capabilities?.length > 0 && (
            <div className="d-flex flex-wrap gap-1 mt-1">
              {plugin.capabilities.map((capability) => (
                <span key={capability} className="badge bg-secondary-subtle text-secondary-emphasis">
                  {capability}
                </span>
              ))}
            </div>
          )}
        </div>
        <div className="text-end">
          <span className="badge bg-success d-block mb-2">Active</span>
          <button
            type="button"
            className="btn btn-sm btn-outline-secondary"
            onClick={() => setExpanded((prev) => !prev)}
            aria-expanded={expanded}
          >
            <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>
              {expanded ? 'expand_less' : 'tune'}
            </span>
            <span className="ms-1">Settings</span>
          </button>
        </div>
      </div>

      {expanded && (
        <div className="mt-3 border-top pt-3">
          {schema.length === 0 ? (
            <p className="text-muted small mb-0">This plugin exposes no configurable settings.</p>
          ) : (
            <>
              {schema.map((field) => (
                <div className="mb-2" key={field.key}>
                  <label className="form-label small mb-1" htmlFor={`${plugin.plugin_id}-${field.key}`}>
                    {field.label}
                  </label>
                  {field.type === 'select' ? (
                    <select
                      id={`${plugin.plugin_id}-${field.key}`}
                      className="form-select form-select-sm"
                      value={values[field.key] ?? ''}
                      onChange={(e) => handleChange(field.key, e.target.value)}
                    >
                      {(field.options ?? []).map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  ) : field.type === 'boolean' ? (
                    <div className="form-check">
                      <input
                        id={`${plugin.plugin_id}-${field.key}`}
                        type="checkbox"
                        className="form-check-input"
                        checked={Boolean(values[field.key])}
                        onChange={(e) => handleChange(field.key, e.target.checked)}
                      />
                    </div>
                  ) : (
                    <input
                      id={`${plugin.plugin_id}-${field.key}`}
                      type={field.type === 'integer' || field.type === 'number' ? 'number' : 'text'}
                      className="form-control form-control-sm"
                      value={values[field.key] ?? ''}
                      onChange={(e) => handleChange(field.key, e.target.value)}
                    />
                  )}
                  {field.description && <div className="form-text small">{field.description}</div>}
                </div>
              ))}

              <div className="d-flex flex-wrap gap-2 mt-3">
                <button type="button" className="btn btn-sm btn-primary" onClick={handleSave} disabled={saving}>
                  {saving ? 'Saving...' : 'Save settings'}
                </button>
                <button type="button" className="btn btn-sm btn-outline-secondary" onClick={handleReset} disabled={saving}>
                  Reset to defaults
                </button>
                <button type="button" className="btn btn-sm btn-outline-secondary" onClick={checkHealth}>
                  Test connection
                </button>
              </div>
            </>
          )}

          {status && <div className={`alert alert-${status.type} py-2 px-3 mt-3 mb-0 small`}>{status.text}</div>}
          {health && !health.checking && (
            <div className={`alert alert-${health.available ? 'success' : 'warning'} py-2 px-3 mt-2 mb-0 small`}>
              {health.detail}
              {health.models?.length > 0 && <div className="mt-1">Available models: {health.models.join(', ')}</div>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TranscribePanel({
  dropHint,
  feedback,
  handleDrop,
  handleDragOver,
  handleFileChange,
  handleUploadClick,
  handleTranscribe,
  handleRecord,
  loading,
  recording,
  transcriptionResult,
  selectedModel,
  setSelectedModel,
  fileInputRef,
  waveformData,
  setWaveformData
}) {
  return (
    <>
      <div className="card shadow-sm border-0 mb-4">
        <div className="card-body">
          <div className="d-flex flex-column flex-md-row justify-content-between gap-3">
            <div>
              <h4 className="card-title mb-1">Transcribe audio</h4>
              <p className="text-muted mb-0">Send any supported audio file to the SpeechScribe pipeline.</p>
            </div>
            <div>
              <select
                className="form-select form-select-sm"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
              >
                {modelOptions.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div
            className="drop-zone mt-4"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            role="region"
            aria-label="Audio file drop zone"
            tabIndex={0}
          >
            <p className="mb-2 text-center fw-semibold">{dropHint}</p>
            <p className="text-muted small text-center">Try dragging in a file or use the controls below.</p>
          </div>
          {waveformData.length > 0 && (
            <div className="mt-3">
              <WaveformVisualization data={waveformData} />
            </div>
          )}
          <div className="d-flex flex-wrap gap-2 mt-3">
            <button type="button" className={`btn ${recording ? 'btn-danger' : 'btn-outline-primary'}`} onClick={handleRecord} aria-label={recording ? 'Stop recording' : 'Start recording'}>
              {recording && <span className="recording-pulse me-2" aria-hidden="true"></span>}
              <span className="material-symbols-outlined" aria-hidden="true">{recording ? 'stop' : 'mic'}</span>
              <span className="ms-2">{recording ? 'Stop Recording' : 'Record'}</span>
            </button>
            <button type="button" className="btn btn-outline-secondary" onClick={handleUploadClick} aria-label="Upload audio file">
              <span className="material-symbols-outlined" aria-hidden="true">upload</span>
              <span className="ms-2">Upload</span>
            </button>
            <button type="button" className="btn btn-primary" onClick={handleTranscribe} disabled={loading} aria-label="Transcribe audio">
              {loading ? (
                <>
                  <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                  Processing
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined" aria-hidden="true">play_arrow</span>
                  <span className="ms-2">Run</span>
                </>
              )}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              className="d-none"
              onChange={handleFileChange}
            />
          </div>
          <div className="alert alert-secondary mt-3 mb-0">{feedback}</div>
        </div>
      </div>

      <div className="row g-3">
        <div className="col-12 col-md-6">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-body">
              <div className="d-flex align-items-start justify-content-between mb-2">
                <h5 className="card-title mb-0">Transcript</h5>
                <button type="button" className="btn btn-sm btn-outline-secondary">
                  <span className="material-symbols-outlined">download</span>
                </button>
              </div>
              <p className="text-muted small mb-2">Auto-generated text aligned to audio.</p>
              <p className="card-text" style={{ minHeight: '180px' }}>
                {transcriptionResult?.transcript || 'Awaiting transcription output.'}
              </p>
            </div>
          </div>
        </div>
        <div className="col-12 col-md-6">
          <div className="card shadow-sm border-0 h-100">
            <div className="card-body">
              <h5 className="card-title">Speaker segmentation</h5>
              <p className="text-muted small">Each segment shows speaker labels + timestamps.</p>
              {transcriptionResult?.segments?.length ? (
                <ul className="list-unstyled mb-0">
                  {transcriptionResult.segments.map((segment) => (
                    <li key={segment.id ?? `${segment.start}-${segment.end}`} className="border-bottom pb-2 mb-2">
                      <div className="d-flex justify-content-between">
                        <strong>Speaker {segment.speaker ?? 'A'}</strong>
                        <span className="text-muted small">
                          {segment.start?.toFixed?.(2) ?? segment.start}:{' '}
                          {segment.end?.toFixed?.(2) ?? segment.end}s
                        </span>
                      </div>
                      <p className="mb-0">{segment.text}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-muted">No segments yet.</p>
              )}
            </div>
          </div>
        </div>
        <div className="col-12">
          <div className="card shadow-sm border-0">
            <div className="card-body">
              <div className="d-flex align-items-start justify-content-between mb-2">
                <h5 className="card-title mb-0">Summary</h5>
                <span className="material-symbols-outlined text-secondary">play_arrow</span>
              </div>
              <p className="card-text">
                {transcriptionResult?.summary || 'Summaries show key points from the transcription.'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function downloadBlob(contents, filename, mimeType) {
  const blob = new Blob([contents], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function chatToMarkdown(messages, model) {
  const header = `# SpeechScribe chat\n\n- Model: ${model || 'unknown'}\n- Exported: ${new Date().toISOString()}\n\n---\n`;
  const body = messages
    .map((msg) => `\n## ${msg.role === 'user' ? 'You' : 'Assistant'}\n\n${msg.content}\n`)
    .join('');
  return header + body;
}

// Parses the `event:`/`data:` frames the /chat/stream endpoint emits.
function parseSseFrames(buffer) {
  const frames = [];
  const parts = buffer.split('\n\n');
  const remainder = parts.pop() ?? '';

  for (const part of parts) {
    let event = 'message';
    const dataLines = [];
    for (const line of part.split('\n')) {
      if (line.startsWith('event:')) {
        event = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim());
      }
    }
    if (dataLines.length === 0) continue;
    try {
      frames.push({ event, data: JSON.parse(dataLines.join('\n')) });
    } catch {
      // Ignore a partial or malformed frame rather than breaking the stream.
    }
  }

  return { frames, remainder };
}

function ChatPanel() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [plugin, setPlugin] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [health, setHealth] = useState(null);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);

  const chatEndRef = useRef(null);
  const abortRef = useRef(null);
  const recorderRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Discover the LLM plugin and the models it can actually serve, so the
  // dropdown reflects what is installed rather than a hardcoded list.
  const loadPlugin = useCallback(async () => {
    try {
      const { data } = await axios.get(`${API_BASE}/plugins`, { params: { type: 'llm' } });
      const llm = Array.isArray(data) ? data[0] : null;

      if (!llm) {
        setPlugin(null);
        setHealth({ available: false, detail: 'No LLM plugin is installed.' });
        return;
      }

      setPlugin(llm);
      const fallback = llm.supported_models ?? [];
      const configured = llm.settings?.model ?? llm.default_model;

      try {
        const { data: status } = await axios.get(`${API_BASE}/plugins/${llm.plugin_id}/health`);
        setHealth(status);
        const available = status.models?.length ? status.models : fallback;
        setModels(available);
        setSelectedModel((prev) => {
          if (prev && available.includes(prev)) return prev;
          if (configured && available.includes(configured)) return configured;
          return available[0] ?? configured ?? '';
        });
      } catch {
        setModels(fallback);
        setSelectedModel((prev) => prev || configured || fallback[0] || '');
        setHealth({ available: false, detail: 'Could not check the LLM backend.' });
      }
    } catch {
      setPlugin(null);
      setHealth({ available: false, detail: `Cannot reach the SpeechScribe API at ${API_BASE}.` });
    }
  }, []);

  useEffect(() => {
    loadPlugin();
  }, [loadPlugin]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const streamReply = useCallback(
    async (conversation) => {
      setStreaming(true);
      setError(null);

      const controller = new AbortController();
      abortRef.current = controller;

      // Placeholder the tokens stream into.
      setMessages((prev) => [...prev, { role: 'assistant', content: '', streaming: true }]);

      const appendToLast = (chunk) =>
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === 'assistant') {
            next[next.length - 1] = { ...last, content: last.content + chunk };
          }
          return next;
        });

      const finishLast = () =>
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === 'assistant') {
            next[next.length - 1] = { ...last, streaming: false };
          }
          return next;
        });

      const dropEmptyLast = () =>
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.role === 'assistant' && !last.content) return prev.slice(0, -1);
          return prev;
        });

      try {
        const response = await fetch(`${API_BASE}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: conversation.map(({ role, content }) => ({ role, content })),
            model: selectedModel || undefined,
            plugin: plugin?.plugin_id
          }),
          signal: controller.signal
        });

        if (!response.ok) {
          let detail = `Request failed with status ${response.status}`;
          try {
            const payload = await response.json();
            detail = payload.detail ?? detail;
          } catch {
            // Response had no JSON body; keep the status-based message.
          }
          throw new Error(detail);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error('Streaming is not supported by this browser.');

        const decoder = new TextDecoder();
        let buffer = '';
        let failed = null;

        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const { frames, remainder } = parseSseFrames(buffer);
          buffer = remainder;

          for (const frame of frames) {
            if (frame.event === 'token') {
              appendToLast(frame.data.content ?? '');
            } else if (frame.event === 'error') {
              failed = frame.data.detail ?? 'The model backend reported an error.';
            }
          }

          if (failed) break;
        }

        if (failed) {
          dropEmptyLast();
          finishLast();
          setError(failed);
          loadPlugin(); // Refresh the backend badge after a failure.
        } else {
          finishLast();
        }
      } catch (err) {
        finishLast();
        dropEmptyLast();
        if (err.name === 'AbortError') {
          setError(null);
        } else {
          setError(err.message || 'Chat request failed.');
          loadPlugin();
        }
      } finally {
        abortRef.current = null;
        setStreaming(false);
      }
    },
    [plugin, selectedModel, loadPlugin]
  );

  const handleSend = useCallback(() => {
    const text = inputValue.trim();
    if (!text || streaming) return;

    const conversation = [...messages, { role: 'user', content: text }];
    setMessages(conversation);
    setInputValue('');
    streamReply(conversation);
  }, [inputValue, streaming, messages, streamReply]);

  const handleRetry = useCallback(() => {
    // Re-send the conversation up to and including the last user message.
    const lastUserIndex = messages.map((m) => m.role).lastIndexOf('user');
    if (lastUserIndex === -1) return;
    const conversation = messages.slice(0, lastUserIndex + 1);
    setMessages(conversation);
    setError(null);
    streamReply(conversation);
  }, [messages, streamReply]);

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  // Voice input: record from the microphone, transcribe it with the ASR
  // plugin, then drop the text into the composer for review before sending.
  const startVoiceInput = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setRecording(false);

        if (chunks.length === 0) return;
        setTranscribing(true);
        try {
          const formData = new FormData();
          formData.append('file', new Blob(chunks, { type: recorder.mimeType }), 'voice-input.webm');
          const { data } = await axios.post(`${API_BASE}/transcribe`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          const transcript = (data.transcript ?? '').trim();
          if (transcript) {
            setInputValue((prev) => (prev ? `${prev} ${transcript}` : transcript));
          } else {
            setError('No speech was recognized in the recording.');
          }
        } catch (err) {
          setError(err.response?.data?.detail ?? 'Voice transcription failed.');
        } finally {
          setTranscribing(false);
        }
      };

      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch (err) {
      console.error('Microphone unavailable:', err);
      setError('Unable to access the microphone. Check the browser permission.');
    }
  };

  const stopVoiceInput = () => {
    recorderRef.current?.stop();
    recorderRef.current = null;
  };

  const exportChat = (format) => {
    setShowExportMenu(false);
    if (messages.length === 0) return;

    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    if (format === 'json') {
      const payload = {
        exported_at: new Date().toISOString(),
        model: selectedModel,
        plugin: plugin?.plugin_id ?? null,
        messages: messages.map(({ role, content }) => ({ role, content }))
      };
      downloadBlob(JSON.stringify(payload, null, 2), `speechscribe-chat-${stamp}.json`, 'application/json');
    } else {
      downloadBlob(chatToMarkdown(messages, selectedModel), `speechscribe-chat-${stamp}.md`, 'text/markdown');
    }
  };

  const clearChat = () => {
    abortRef.current?.abort();
    setMessages([]);
    setError(null);
  };

  const backendDown = health && health.available === false;

  return (
    <div className="card shadow-sm border-0 h-100">
      <div className="card-header bg-white border-bottom">
        <div className="d-flex flex-wrap justify-content-between align-items-center gap-2">
          <div className="d-flex align-items-center gap-2">
            <h4 className="mb-0">AI Chat</h4>
            <span
              className={`badge ${backendDown ? 'bg-danger' : 'bg-success'}`}
              title={health?.detail ?? 'Checking backend...'}
            >
              {backendDown ? 'Offline' : 'Ready'}
            </span>
          </div>
          <div className="d-flex align-items-center gap-2">
            <select
              className="form-select form-select-sm w-auto"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={models.length === 0}
              aria-label="Model"
            >
              {models.length === 0 ? (
                <option value="">No models available</option>
              ) : (
                models.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))
              )}
            </select>

            <div className="position-relative">
              <button
                type="button"
                className="btn btn-sm btn-outline-secondary"
                onClick={() => setShowExportMenu((prev) => !prev)}
                disabled={messages.length === 0}
                aria-expanded={showExportMenu}
                title="Export chat history"
              >
                <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>download</span>
              </button>
              {showExportMenu && (
                <div className="dropdown-menu show" style={{ position: 'absolute', right: 0, top: '110%' }}>
                  <button type="button" className="dropdown-item" onClick={() => exportChat('markdown')}>
                    Export as Markdown
                  </button>
                  <button type="button" className="dropdown-item" onClick={() => exportChat('json')}>
                    Export as JSON
                  </button>
                </div>
              )}
            </div>

            <button
              type="button"
              className="btn btn-sm btn-outline-secondary"
              onClick={clearChat}
              disabled={messages.length === 0}
              title="Clear conversation"
            >
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>delete</span>
            </button>
          </div>
        </div>
      </div>

      <div className="card-body d-flex flex-column" style={{ height: 'calc(100vh - 280px)', overflow: 'hidden' }}>
        {backendDown && (
          <div className="alert alert-warning d-flex justify-content-between align-items-center py-2">
            <span className="small">{health.detail}</span>
            <button type="button" className="btn btn-sm btn-outline-dark" onClick={loadPlugin}>
              Retry
            </button>
          </div>
        )}

        {error && (
          <div className="alert alert-danger d-flex justify-content-between align-items-start gap-2 py-2">
            <span className="small">{error}</span>
            <span className="d-flex gap-2">
              <button type="button" className="btn btn-sm btn-outline-dark" onClick={handleRetry} disabled={streaming}>
                Retry
              </button>
              <button type="button" className="btn-close" onClick={() => setError(null)} aria-label="Dismiss error" />
            </span>
          </div>
        )}

        <div className="flex-grow-1 overflow-auto mb-3" style={{ minHeight: 0 }}>
          {messages.length === 0 ? (
            <div className="text-center text-muted mt-5">
              <span className="material-symbols-outlined" style={{ fontSize: '48px' }}>chat</span>
              <p className="mt-2">Start a conversation with the AI assistant</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`d-flex mb-3 ${msg.role === 'user' ? 'justify-content-end' : 'justify-content-start'}`}
              >
                <div
                  className={`rounded-3 p-3 ${msg.role === 'user' ? 'bg-primary text-white' : 'bg-light'}`}
                  style={{ maxWidth: '75%' }}
                >
                  {msg.role === 'assistant' ? (
                    <>
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                      {msg.streaming && !msg.content && (
                        <span className="text-muted small">Thinking...</span>
                      )}
                      {msg.streaming && msg.content && <span className="typing-caret" aria-hidden="true" />}
                    </>
                  ) : (
                    <p className="mb-0">{msg.content}</p>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="mt-auto">
          <div className="input-group">
            <button
              className={`btn ${recording ? 'btn-danger' : 'btn-outline-secondary'}`}
              type="button"
              onClick={recording ? stopVoiceInput : startVoiceInput}
              disabled={transcribing || streaming}
              title={recording ? 'Stop recording' : 'Dictate a message'}
              aria-label={recording ? 'Stop recording' : 'Dictate a message'}
            >
              {transcribing ? (
                <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true" />
              ) : (
                <span className="material-symbols-outlined">{recording ? 'stop' : 'mic'}</span>
              )}
            </button>
            <textarea
              className="form-control"
              placeholder={recording ? 'Listening...' : 'Type your message, or use the mic to dictate...'}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={2}
              disabled={streaming}
            />
            {streaming ? (
              <button className="btn btn-outline-danger" type="button" onClick={handleStop} title="Stop generating">
                <span className="material-symbols-outlined">stop_circle</span>
              </button>
            ) : (
              <button
                className="btn btn-primary"
                type="button"
                onClick={handleSend}
                disabled={!inputValue.trim()}
                title="Send"
              >
                <span className="material-symbols-outlined">send</span>
              </button>
            )}
          </div>
          <p className="text-muted small mt-2 mb-0">
            Enter sends, Shift+Enter adds a newline. Responses stream as they are generated.
          </p>
        </div>
      </div>
    </div>
  );
}


function WaveformVisualization({ data }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.length) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = 'var(--bs-primary)';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const sliceWidth = width / data.length;
    let x = 0;

    for (let i = 0; i < data.length; i++) {
      const v = data[i] / 128.0;
      const y = v * height / 2;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    ctx.stroke();
  }, [data]);

  return (
    <div className="waveform-container">
      <canvas ref={canvasRef} width={800} height={100} className="w-100 border rounded" />
    </div>
  );
}

function LiveMicrophonePanel({ recording, setRecording, waveformData, setWaveformData }) {
  const [realTimeTranscript, setRealTimeTranscript] = useState('');
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioChunks, setAudioChunks] = useState([]);
  const [transcriptionResult, setTranscriptionResult] = useState(null);

  useEffect(() => {
    if (recording && mediaRecorder) {
      mediaRecorder.start();
      const interval = setInterval(() => {
        // Simulate waveform
        setWaveformData(prev => {
          const newData = [...prev];
          newData.push(Math.random() * 255);
          if (newData.length > 200) newData.shift();
          return newData;
        });
      }, 100);
      return () => clearInterval(interval);
    } else if (mediaRecorder) {
      mediaRecorder.stop();
    }
  }, [recording, mediaRecorder, setWaveformData]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];

      recorder.ondataavailable = (event) => {
        chunks.push(event.data);
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: 'audio/wav' });
        setAudioChunks(chunks);
        
        // Send to API for transcription
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        
        try {
          const response = await axios.post(`${API_BASE}/transcribe`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          setTranscriptionResult(response.data);
          setRealTimeTranscript(response.data.transcript);
        } catch (error) {
          console.error('Transcription failed:', error);
          setRealTimeTranscript(describeApiError(error, 'Transcription failed.'));
        }
      };

      setMediaRecorder(recorder);
      setRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Unable to access microphone');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    setRecording(false);
  };

  return (
    <div className="card shadow-sm border-0">
      <div className="card-body">
        <h4 className="card-title">Live Microphone</h4>
        <p className="text-muted">Real-time speech-to-text from your microphone.</p>
        <div className="d-flex gap-2 mb-3">
          <button
            className={`btn ${recording ? 'btn-danger' : 'btn-success'}`}
            onClick={recording ? stopRecording : startRecording}
          >
            {recording && <span className="recording-pulse me-2"></span>}
            <span className="material-symbols-outlined">{recording ? 'stop' : 'mic'}</span>
            {recording ? 'Stop' : 'Start'} Recording
          </button>
        </div>
        {waveformData.length > 0 && <WaveformVisualization data={waveformData} />}
        <div className="mt-3">
          <h5>Real-time Transcript</h5>
          <div className="border rounded p-3 bg-light" style={{ minHeight: '100px' }}>
            {realTimeTranscript || 'Start recording to see live transcription...'}
          </div>
        </div>
        {transcriptionResult && (
          <div className="mt-3">
            <h5>Full Transcript</h5>
            <div className="border rounded p-3">
              <p>{transcriptionResult.transcript}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function MeetingModePanel() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [transcriptionResult, setTranscriptionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState('Drop meeting audio files here or upload.');
  const fileInputRef = useRef(null);

  const handleDrop = useCallback((event) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      setSelectedFile(file);
      setFeedback(`Ready to transcribe meeting: ${file.name}`);
    }
  }, []);

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
  }, []);

  const handleFileChange = useCallback((event) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setFeedback(`Ready to transcribe meeting: ${file.name}`);
    }
  }, []);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleTranscribe = async () => {
    if (!selectedFile) {
      setFeedback('Select a file before running transcription.');
      return;
    }

    setLoading(true);
    setFeedback('Transcribing meeting...');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('profile', 'meeting_diarization'); // Assuming a profile for meetings

      const response = await axios.post(`${API_BASE}/transcribe`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setTranscriptionResult(response.data);
      setFeedback('Meeting transcription complete.');
    } catch (error) {
      console.error(error);
      setFeedback(describeApiError(error, 'Meeting transcription failed.'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card shadow-sm border-0">
      <div className="card-body">
        <h4 className="card-title">Meeting Mode</h4>
        <p className="text-muted">Capture and transcribe meetings with speaker diarization.</p>
        
        <div
          className="drop-zone mb-3"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          role="region"
          aria-label="Meeting audio file drop zone"
        >
          <p className="mb-2 text-center fw-semibold">{selectedFile ? selectedFile.name : 'Drop meeting audio here'}</p>
          <p className="text-muted small text-center">Upload meeting recordings for transcription with speaker identification.</p>
        </div>

        <div className="d-flex gap-2 mb-3">
          <button type="button" className="btn btn-outline-secondary" onClick={handleUploadClick}>
            <span className="material-symbols-outlined">upload</span>
            Upload
          </button>
          <button type="button" className="btn btn-primary" onClick={handleTranscribe} disabled={loading || !selectedFile}>
            {loading ? (
              <>
                <span className="spinner-border spinner-border-sm me-2"></span>
                Processing
              </>
            ) : (
              <>
                <span className="material-symbols-outlined">play_arrow</span>
                Transcribe Meeting
              </>
            )}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            className="d-none"
            onChange={handleFileChange}
          />
        </div>

        <div className="alert alert-secondary">{feedback}</div>

        {transcriptionResult && (
          <div className="mt-3">
            <h5>Meeting Transcript</h5>
            <div className="border rounded p-3" style={{ maxHeight: '400px', overflowY: 'auto' }}>
              <p><strong>Speakers:</strong> {transcriptionResult.speakers.join(', ')}</p>
              <p>{transcriptionResult.transcript}</p>
              {transcriptionResult.segments.map((segment, idx) => (
                <div key={idx} className="mb-2">
                  <strong>{segment.speaker_label || `Speaker ${segment.speaker_id}`}:</strong> {segment.text}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function VoiceSynthesisPanel() {
  const [text, setText] = useState('');
  const [voice, setVoice] = useState('default');
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);

  const handleSynthesize = async () => {
    if (!text.trim()) return;

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/tts`, {
        text: text,
        voice: voice,
        engine: 'coqui_tts', // or whatever default
      });

      // Decode base64 audio
      const audioBlob = new Blob([Uint8Array.from(atob(response.data.audio_base64), c => c.charCodeAt(0))], { type: 'audio/wav' });
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
    } catch (error) {
      console.error('TTS failed:', error);
      alert(describeApiError(error, 'Failed to synthesize speech.'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card shadow-sm border-0">
      <div className="card-body">
        <h4 className="card-title">Voice Synthesis</h4>
        <p className="text-muted">Generate speech from text using TTS models.</p>
        <div className="mb-3">
          <label className="form-label">Text to Synthesize</label>
          <textarea
            className="form-control"
            rows={4}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to convert to speech..."
          />
        </div>
        <div className="mb-3">
          <label className="form-label">Voice</label>
          <select className="form-select" value={voice} onChange={(e) => setVoice(e.target.value)}>
            <option value="default">Default Voice</option>
            <option value="male">Male Voice</option>
            <option value="female">Female Voice</option>
          </select>
        </div>
        <button className="btn btn-primary me-2" onClick={handleSynthesize} disabled={!text.trim() || loading}>
          {loading ? (
            <>
              <span className="spinner-border spinner-border-sm me-2"></span>
              Synthesizing...
            </>
          ) : (
            <>
              <span className="material-symbols-outlined me-2">volume_up</span>
              Synthesize Speech
            </>
          )}
        </button>
        {audioUrl && (
          <audio controls className="mt-3">
            <source src={audioUrl} type="audio/wav" />
            Your browser does not support the audio element.
          </audio>
        )}
      </div>
    </div>
  );
}

export default App;
