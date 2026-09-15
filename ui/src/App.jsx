import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

function App() {
  const [activeTab, setActiveTab] = useState('transcribe');
  const [file, setFile] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatModel, setChatModel] = useState('llama3.2:latest');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState('');
  const chatMessagesRef = useRef(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError('');
      setTranscript('');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file to transcribe.');
      return;
    }

    setIsLoading(true);
    setError('');
    setTranscript('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/transcribe`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setTranscript(response.data.transcript || '');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to transcribe the file. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setTranscript('');
    setError('');
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;

    const userMessage = { role: 'user', content: chatInput.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setChatInput('');
    setChatError('');
    setChatLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: userMessage.content,
        model: chatModel,
      });

      const assistantMessage = { role: 'assistant', content: response.data.response };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setChatError(err.response?.data?.detail || 'Failed to send message. Please try again.');
    } finally {
      setChatLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setChatError('');
  };

  const handleChatKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages, chatLoading]);

  const renderTranscribePage = () => (
    <div className="container-fluid py-4">
      <div className="row">
        <div className="col-12">
          <h1 className="mb-2">
            <span className="material-icons me-2">mic</span>
            Transcribe Audio/Video
          </h1>
          <p className="text-muted mb-4">Upload an audio or video file to get a text transcription</p>
        </div>
      </div>

      <div className="row">
        <div className="col-lg-6 mb-4">
          <div className="card h-100 shadow-sm">
            <div className="card-header">
              <h5 className="card-title mb-0">
                <span className="material-icons me-2">upload_file</span>
                Upload File
              </h5>
            </div>
            <div className="card-body">
              <div className="mb-3">
                <input
                  type="file"
                  className="form-control"
                  id="fileInput"
                  accept="audio/*,video/*"
                  onChange={handleFileChange}
                  disabled={isLoading}
                />
                <div className="form-text">
                  Supported formats: MP3, WAV, MP4, AVI, MOV, and other common audio/video files
                </div>
              </div>

              {file && (
                <div className="alert alert-info mb-3">
                  <span className="material-icons me-2">audio_file</span>
                  Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </div>
              )}

              <div className="d-flex gap-2">
                <button
                  className="btn btn-primary"
                  onClick={handleUpload}
                  disabled={!file || isLoading}
                >
                  {isLoading ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                      Transcribing...
                    </>
                  ) : (
                    <>
                      <span className="material-icons me-2">play_arrow</span>
                      Transcribe
                    </>
                  )}
                </button>

                <button
                  className="btn btn-outline-secondary"
                  onClick={handleClear}
                  disabled={isLoading}
                >
                  <span className="material-icons me-2">refresh</span>
                  Clear
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="col-lg-6 mb-4">
          <div className="card h-100 shadow-sm">
            <div className="card-header">
              <h5 className="card-title mb-0">
                <span className="material-icons me-2">description</span>
                Transcription Result
              </h5>
            </div>
            <div className="card-body">
              {error && (
                <div className="alert alert-danger">
                  <span className="material-icons me-2">error</span>
                  {error}
                </div>
              )}

              {transcript && (
                <div className="alert alert-success">
                  <span className="material-icons me-2">check_circle</span>
                  Transcription completed successfully!
                </div>
              )}

              {transcript ? (
                <div className="border rounded p-3 bg-light">
                  <pre className="mb-0" style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
                    {transcript}
                  </pre>
                </div>
              ) : (
                <div className="text-center text-muted py-5">
                  <span className="material-icons" style={{ fontSize: '4rem', opacity: 0.3 }}>
                    description
                  </span>
                  <p className="mt-3">Transcription will appear here after upload</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderChatPage = () => (
    <div className="container-fluid py-4">
      <div className="row">
        <div className="col-12">
          <h1 className="mb-2">
            <span className="material-icons me-2">chat</span>
            AI Chat
          </h1>
          <p className="text-muted mb-4">Chat with AI models powered by SpeechScribe plugins</p>
        </div>
      </div>

      <div className="row">
        <div className="col-12">
          <div className="card shadow-sm">
            <div className="card-header d-flex justify-content-between align-items-center">
              <h5 className="card-title mb-0">
                <span className="material-icons me-2">smart_toy</span>
                Conversation
              </h5>
              <button
                className="btn btn-outline-danger btn-sm"
                onClick={handleClearChat}
                disabled={chatLoading}
              >
                <span className="material-icons me-1">delete</span>
                Clear Chat
              </button>
            </div>

            <div
              ref={chatMessagesRef}
              className="card-body"
              style={{ height: '400px', overflowY: 'auto', backgroundColor: '#f8f9fa' }}
            >
              {messages.length === 0 ? (
                <div className="text-center text-muted py-5">
                  <span className="material-icons" style={{ fontSize: '4rem', opacity: 0.3 }}>chat</span>
                  <p className="mt-3">Start a conversation by typing a message below</p>
                </div>
              ) : (
                messages.map((message, index) => (
                  <div
                    key={`${message.role}-${index}`}
                    className={`d-flex mb-3 ${message.role === 'user' ? 'justify-content-end' : 'justify-content-start'}`}
                  >
                    <div className={`d-flex align-items-start ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                      <div
                        className={`rounded-circle d-flex align-items-center justify-content-center me-2 ${message.role === 'user' ? 'ms-2 me-0' : ''}`}
                        style={{
                          width: '40px',
                          height: '40px',
                          backgroundColor: message.role === 'user' ? '#0d6efd' : '#198754',
                          color: 'white',
                          flexShrink: 0,
                        }}
                      >
                        <span className="material-icons" style={{ fontSize: '20px' }}>
                          {message.role === 'user' ? 'person' : 'smart_toy'}
                        </span>
                      </div>

                      <div
                        className={`px-3 py-2 rounded ${message.role === 'user' ? 'bg-primary text-white' : 'bg-light border'}`}
                        style={{ maxWidth: '70%', wordBreak: 'break-word' }}
                      >
                        {message.content}
                      </div>
                    </div>
                  </div>
                ))
              )}

              {chatLoading && (
                <div className="d-flex justify-content-start mb-3">
                  <div className="d-flex align-items-start">
                    <div
                      className="rounded-circle d-flex align-items-center justify-content-center me-2"
                      style={{
                        width: '40px',
                        height: '40px',
                        backgroundColor: '#198754',
                        color: 'white',
                        flexShrink: 0,
                      }}
                    >
                      <span className="material-icons" style={{ fontSize: '20px' }}>smart_toy</span>
                    </div>
                    <div className="px-3 py-2 rounded bg-light border">
                      <div className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></div>
                      Thinking...
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="card-footer">
              {chatError && (
                <div className="alert alert-danger mb-3">
                  <span className="material-icons me-2">error</span>
                  {chatError}
                </div>
              )}

              <div className="row">
                <div className="col-md-9 mb-2 mb-md-0">
                  <div className="input-group">
                    <input
                      type="text"
                      className="form-control"
                      placeholder="Type your message..."
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      onKeyDown={handleChatKeyDown}
                      disabled={chatLoading}
                    />
                    <button
                      className="btn btn-primary"
                      onClick={handleSendMessage}
                      disabled={!chatInput.trim() || chatLoading}
                    >
                      <span className="material-icons">send</span>
                    </button>
                  </div>
                </div>

                <div className="col-md-3">
                  <input
                    type="text"
                    className="form-control"
                    placeholder="Model (optional)"
                    value={chatModel}
                    onChange={(e) => setChatModel(e.target.value)}
                    disabled={chatLoading}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderComingSoonPage = (title) => (
    <div className="container-fluid py-4">
      <div className="row">
        <div className="col-12">
          <div className="text-center py-5">
            <span className="material-icons" style={{ fontSize: '6rem', opacity: 0.3 }}>construction</span>
            <h2 className="mt-3">{title}</h2>
            <p className="text-muted">This feature is coming soon.</p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'transcribe':
        return renderTranscribePage();
      case 'chat':
        return renderChatPage();
      case 'tts':
        return renderComingSoonPage('Text-to-Speech');
      case 'plugins':
        return renderComingSoonPage('Plugin Management');
      case 'translate':
        return renderComingSoonPage('Translation');
      case 'voice-assistant':
        return renderComingSoonPage('Voice Assistant');
      case 'dubbing':
        return renderComingSoonPage('Audio Dubbing');
      default:
        return renderTranscribePage();
    }
  };

  return (
    <div className="d-flex" style={{ minHeight: '100vh' }}>
      <div className="bg-dark text-white" style={{ width: '250px', minHeight: '100vh' }}>
        <div className="p-3">
          <h4 className="mb-4">
            <span className="material-icons me-2">mic</span>
            SpeechScribe
          </h4>

          <nav className="nav flex-column">
            <button
              className={`nav-link text-start text-white ${activeTab === 'transcribe' ? 'active' : ''}`}
              onClick={() => setActiveTab('transcribe')}
            >
              <span className="material-icons me-2">mic</span>
              Transcribe
            </button>

            <button
              className={`nav-link text-start text-white ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              <span className="material-icons me-2">chat</span>
              Chat
            </button>

            <button
              className={`nav-link text-start text-white ${activeTab === 'tts' ? 'active' : ''}`}
              onClick={() => setActiveTab('tts')}
            >
              <span className="material-icons me-2">volume_up</span>
              TTS
            </button>

            <button
              className={`nav-link text-start text-white ${activeTab === 'plugins' ? 'active' : ''}`}
              onClick={() => setActiveTab('plugins')}
            >
              <span className="material-icons me-2">extension</span>
              Plugins
            </button>

            <button
              className={`nav-link text-start text-white ${activeTab === 'translate' ? 'active' : ''}`}
              onClick={() => setActiveTab('translate')}
            >
              <span className="material-icons me-2">translate</span>
              Translate
            </button>

            <button
              className={`nav-link text-start text-white ${activeTab === 'voice-assistant' ? 'active' : ''}`}
              onClick={() => setActiveTab('voice-assistant')}
            >
              <span className="material-icons me-2">smart_toy</span>
              Voice Assistant
            </button>

            <button
              className={`nav-link text-start text-white ${activeTab === 'dubbing' ? 'active' : ''}`}
              onClick={() => setActiveTab('dubbing')}
            >
              <span className="material-icons me-2">movie</span>
              Dubbing
            </button>
          </nav>
        </div>
      </div>

      <div className="flex-grow-1 bg-light">
        {renderContent()}
      </div>
    </div>
  );
}

export default App;