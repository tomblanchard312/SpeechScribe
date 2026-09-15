# Plugin System & Ollama Integration - Implementation Complete ✅

## What's Been Built

### 1. Plugin Architecture
- **Plugin Loader** (`src/speechscribe/plugins/loader.py`)
  - Dynamic plugin discovery from `src/speechscribe/plugins/builtin/`
  - Hot-reload support (monitors file changes every 2.5 seconds)
  - Validates `plugin.json` schema and entry points
  - Supports 4 plugin types: asr, tts, translation, summarization

### 2. Built-in Plugins

#### Ollama LLM (`src/speechscribe/plugins/builtin/ollama_llm/`)
- **Type**: summarization
- **Capabilities**: generate, chat, summarize, generate_meeting_notes
- **Models**: qwen2.5, llama3, deepseek, llama3.1, mistral
- **Features**:
  - Basic text generation
  - Multi-turn chat conversations
  - Streaming support
  - Meeting notes with structured output
  - Connects to local Ollama on port 11434

#### Whisper ASR (`src/speechscribe/plugins/builtin/whisper/`)
- **Type**: asr
- **Capabilities**: transcribe, translate, detect_language
- **Models**: tiny, base, small, medium, large-v2, large-v3
- **Features**:
  - Fast transcription with faster-whisper
  - Language detection
  - Translation to English

#### SpeechT5 TTS (`src/speechscribe/plugins/builtin/speecht5/`)
- **Type**: tts
- **Capabilities**: synthesize, text_to_speech
- **Models**: microsoft/speecht5_tts
- **Features**:
  - HuggingFace-based TTS
  - Multiple voice embeddings
  - High-quality speech synthesis

### 3. API Integration

**New Endpoints in `src/speechscribe/api/main.py`:**

- `GET /plugins` - List all loaded plugins (optional `?type=llm|asr|tts`)
  ```json
  [
    {
      "plugin_id": "ollama_llm",
      "name": "ollama_llm",
      "type": "llm",
      "version": "1.1.0",
      "description": "Local Ollama LLM plugin for chat, summarization and meeting notes",
      "capabilities": ["generate", "chat", "stream_chat", "summarize"],
      "supported_models": ["qwen2.5", "llama3", "deepseek"],
      "default_model": "qwen2.5",
      "settings_schema": [{"key": "base_url", "type": "string", "label": "Ollama endpoint", "default": "http://127.0.0.1:11434"}],
      "settings": {"base_url": "http://127.0.0.1:11434", "model": "qwen2.5"}
    }
  ]
  ```

- `GET /plugins/{plugin_id}` - One plugin descriptor
- `PUT /plugins/{plugin_id}/settings` - Save settings (`{"settings": {...}}`); values are
  validated against the plugin's `settings_schema` and persisted to
  `config/plugin_settings.json`
- `DELETE /plugins/{plugin_id}/settings` - Revert to manifest defaults
- `GET /plugins/{plugin_id}/health` - Whether the plugin's backend is reachable, plus the
  models it can actually serve

- `POST /chat` - Chat with LLM plugins (blocking)
  ```json
  {
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "model": "qwen2.5"
  }
  ```
  Responds with `{"response": "...", "model": "qwen2.5"}`.

- `POST /chat/stream` - Same request body, streamed back as server-sent events:
  ```
  event: token
  data: {"content": "Hel"}

  event: done
  data: {"model": "qwen2.5"}
  ```
  Failures arrive as an `error` event (the response has already started, so they
  cannot use an HTTP status code).

### 4. Web UI (`web/`)

**New Features:**
- **Chat Panel**: reachable from the "AI Chat" card in the sidebar
- **Streaming**: replies render token by token, with a Stop button to abort
- **Model Selector**: populated from the models Ollama actually has pulled, falling back to
  the manifest's `supported_models` when the backend is down
- **Connection state**: a Ready/Offline badge plus a dismissible error alert with Retry
- **Voice input**: dictate with the mic; the transcript lands in the composer for review
- **Export**: download the conversation as Markdown or JSON; Clear starts over
- **Message Bubbles**: User (blue) and assistant (gray) messages
- **Markdown Support**: Code blocks, lists, tables rendered with react-markdown
- **Auto-scroll**: Automatically scrolls to latest message
- **Plugin panel**: lists every discovered plugin with version, capabilities and an inline
  settings form (Save / Reset to defaults / Test connection)

### 5. Documentation

- `QUICKSTART.md` - Complete setup and testing guide
- `src/speechscribe/plugins/builtin/README.md` - Plugin system documentation
- `src/speechscribe/plugins/builtin/ollama_llm/README.md` - Ollama plugin guide
- `examples/ollama_usage.py` - Programmatic usage examples
- `examples/plugin_integration.py` - Full pipeline with transcription + summarization
- `test_plugins.py` - Test script to verify plugin system

### 6. Dependencies Added

Updated `requirements.txt`:
- `requests>=2.31.0` - For Ollama HTTP client
- `transformers>=4.30.0` - For SpeechT5 plugin
- `datasets>=2.14.0` - For SpeechT5 embeddings

Updated `web/package.json`:
- `react-markdown` - For markdown rendering in chat

## Testing Checklist

### ✅ Pre-flight Checks

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Ollama**
   - Download from https://ollama.ai
   - Pull a model: `ollama pull qwen2.5`
   - Start server: `ollama serve`

3. **Install Frontend Dependencies**
   ```bash
   cd web
   npm install
   ```

### 🧪 Test Plan

#### Test 1: Plugin Loader
```bash
python test_plugins.py
```
**Expected**: See all 3 plugins listed (ollama_llm, whisper, speecht5)

#### Test 2: API Backend
```bash
python src/speechscribe/api/main.py
```

**Test 2a: List Plugins**
```bash
curl http://localhost:8000/plugins
```
**Expected**: JSON array with 3 plugins

**Test 2b: Chat API**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "model": "qwen2.5"
  }'
```
**Expected**: JSON response with assistant message

**Test 2c: Streaming Chat**
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "model": "qwen2.5"}'
```
**Expected**: a sequence of `event: token` frames, then `event: done`

**Test 2d: Plugin Health**
```bash
curl http://localhost:8000/plugins/ollama_llm/health
```
**Expected**: `available: true` and the models Ollama has pulled

#### Test 3: Web UI
```bash
cd web
npm run dev
```
Open http://localhost:5173

**Test 3a: Navigate to Chat**
- Click "AI Chat" in sidebar
- Should see empty chat interface with model selector

**Test 3b: Send Message**
- Type "Hello! What can you do?"
- Press Enter
- Should see:
  - Your message on right (blue bubble)
  - AI response on left (gray bubble)
  - Markdown rendered properly

**Test 3c: Test Markdown**
- Try: "Show me a Python hello world function"
- Should see code block with syntax highlighting

**Test 3d: Streaming, export and voice**
- The reply should appear progressively, with a Stop button while it streams
- The download icon offers Markdown and JSON exports of the conversation
- The mic button records, transcribes through `/transcribe`, and drops the text into the
  composer (needs an ASR plugin installed, e.g. `pip install faster-whisper`)

**Test 3e: Ollama down**
- Stop Ollama, then send a message
- Should see an "Offline" badge and a red alert with a Retry button, not a silent failure

**Test 3f: Plugin settings**
- Click the extension icon in the navbar
- Expand a plugin's Settings, change a value, Save, then Test connection
- Values persist to `config/plugin_settings.json` and apply on the next call

#### Test 4: Hot Reload
```bash
# Edit src/speechscribe/plugins/builtin/ollama_llm/plugin.json
# Change version from "1.0.0" to "1.0.1"
# Wait 3 seconds
curl http://localhost:8000/plugins | grep version
```
**Expected**: See version "1.0.1"

#### Test 5: Programmatic Usage

**Test 5a: Ollama Examples**
```bash
python examples/ollama_usage.py
```
**Expected**: See 5 examples run (chat, summarize, meeting notes, streaming, multi-turn)

**Test 5b: Full Pipeline** (requires audio file)
```bash
python examples/plugin_integration.py your_audio.mp3
```
**Expected**: Audio transcribed, then summarized

### 🐛 Troubleshooting

#### Plugin Not Loading
- Check `src/speechscribe/plugins/builtin/<plugin_name>/` exists
- Verify `plugin.json` is valid JSON
- Ensure `plugin.py` has the correct class name
- Run `python test_plugins.py` for detailed errors

#### Ollama Connection Error
- Verify Ollama is running: `curl http://localhost:11434/api/version`
- Check model is pulled: `ollama list`
- Try `ollama serve` if not running

#### Frontend Issues
- Missing react-markdown: `cd web && npm install`
- CORS errors: Ensure backend is on port 8000
- UI not loading: Check `npm run dev` output for port conflicts

#### Import Errors
- The API imports plugins using relative paths
- If you see import errors, ensure the package structure matches:
  ```
  speechscribe/
  ├── core/
  │   └── plugins/
  └── plugins/
      ├── ollama_llm/
      ├── whisper/
      └── speecht5/
  ```

## Layout

Everything lives under the single canonical package, matching the scope-boundary refactor
that removed the duplicate top-level `speechscribe/` package:

```
src/speechscribe/plugins/          # loader + settings store
src/speechscribe/plugins/builtin/  # shipped plugins, discovered at runtime
src/speechscribe/api/              # FastAPI layer
web/                               # React console (Vite)
config/                            # llm_models.yaml, plugin_settings.json
```

Plugin settings are written at runtime, so they never land inside the installed package.
They go to `config/plugin_settings.json` in a source checkout, or the user config directory
when installed. `SPEECHSCRIBE_CONFIG_DIR` overrides the location.

## Known Issues

1. **Line length warnings**: Some lines exceed 79 characters (linting warnings, not errors)
2. **Type hints**: Some optional parameters have type warnings (non-critical)
4. **`localhost` vs `127.0.0.1`**: on Windows, resolving `localhost` tries IPv6 first and
   stalls ~2s per request before falling back. The Ollama plugin defaults to
   `http://127.0.0.1:11434` for that reason.

## Next Steps

### Immediate Testing
1. Run `python test_plugins.py` to verify plugins load
2. Start backend: `cd src && uvicorn speechscribe.api.main:app --reload`
3. Start frontend: `cd web && npm run dev`
4. Test chat in browser at http://localhost:5173

### Future Enhancements
- [x] Streaming responses in UI (`POST /chat/stream` emits SSE; the panel renders tokens as they arrive, with a Stop button)
- [x] Error handling in ChatPanel for Ollama connection failures (backend badge from `/plugins/{id}/health`, dismissible error alert with Retry)
- [x] Plugin settings UI (per-plugin form driven by `settings_schema`, saved via `PUT /plugins/{id}/settings`)
- [x] Voice input in chat (microphone → `/transcribe` → text lands in the composer for review before sending)
- [x] Export chat history (Markdown or JSON download, plus Clear conversation)
- [x] Plugin marketplace/discovery (`GET /plugins` returns full descriptors; the plugin panel lists capabilities, versions and connection state)
- [ ] Remote plugin registry (install plugins from a catalog rather than the local `plugins/` folder)

## Documentation Links

- [Quick Start Guide](./QUICKSTART.md)
- [Plugin System Documentation](./src/speechscribe/plugins/builtin/README.md)
- [Ollama Plugin Guide](./src/speechscribe/plugins/builtin/ollama_llm/README.md)
- [API Documentation](http://localhost:8000/docs) (when running)

## Support

If you encounter issues:
1. Check logs (backend terminal and browser console)
2. Review error messages in `python test_plugins.py`
3. Verify all dependencies installed: `pip list | grep -E "(fastapi|requests|transformers)"`
4. Ensure Ollama is running: `ollama serve`

Happy coding! 🚀
