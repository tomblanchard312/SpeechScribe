# Quick Start Guide - Plugin System & Ollama Integration

This guide will help you test the new plugin system and Ollama chat integration in SpeechScribe.

## Prerequisites

1. **Python 3.10+**
2. **Node.js 18+** (for the web UI)
3. **Ollama** (for LLM chat)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `fastapi` - REST API framework
- `requests` - HTTP client for Ollama plugin
- `faster-whisper` - Whisper ASR plugin
- `transformers` - SpeechT5 TTS plugin

### 2. Install Ollama

Download from: https://ollama.ai

Pull a model:
```bash
ollama pull qwen2.5
```

Start Ollama:
```bash
ollama serve
```

### 3. Install Frontend Dependencies

```bash
cd web
npm install
```

This will install:
- React 18
- Vite (dev server)
- Bootstrap 5
- react-markdown
- Axios

## Testing the Plugin System

### Step 1: Test Plugin Loader

Run the test script:

```bash
python test_plugins.py
```

You should see:
```
Found 3 plugins:
  - Whisper ASR (asr) v1.0.0
  - SpeechT5 TTS (tts) v1.0.0
  - Ollama LLM (summarization) v1.0.0
```

### Step 2: Start the Backend

```bash
cd src
uvicorn speechscribe.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

### Step 3: Verify Plugins via API

List all plugins:
```bash
curl http://localhost:8000/plugins
```

Expected response:
```json
[
  {
    "plugin_id": "ollama_llm",
    "name": "Ollama LLM",
    "type": "summarization",
    "version": "1.0.0",
    "capabilities": ["generate", "chat", "summarize", "generate_meeting_notes"],
    "models": ["qwen2.5", "llama3", "deepseek"]
  },
  {
    "plugin_id": "whisper",
    "name": "Whisper ASR",
    ...
  }
]
```

### Step 4: Test Chat API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello! What can you do?"}
    ],
    "model": "qwen2.5",
    "stream": false
  }'
```

Expected response:
```json
{
  "role": "assistant",
  "content": "Hello! I can help you with...",
  "model": "qwen2.5"
}
```

## Testing the Web UI

### Step 1: Start the Frontend

```bash
cd web
npm run dev
```

Open http://localhost:5173

### Step 2: Navigate to Chat

Click **"AI Chat"** in the sidebar.

### Step 3: Test Chat Interface

1. Select a model from the dropdown (qwen2.5, llama3, etc.)
2. Type a message: "Hello! Can you summarize text?"
3. Press Enter or click Send
4. You should see:
   - Your message on the right (blue bubble)
   - AI response on the left (gray bubble) with markdown formatting

### Step 4: Test Markdown Rendering

Try these prompts to test markdown:

**Code blocks:**
```
Show me a Python function to calculate fibonacci numbers
```

**Lists:**
```
Give me 5 tips for better meetings
```

**Tables:**
```
Compare Python vs JavaScript in a table
```

## Plugin Hot Reload Testing

The plugin loader watches for changes and reloads automatically.

### Test Hot Reload

1. Edit a plugin file:
   ```bash
   # Edit src/speechscribe/plugins/builtin/ollama_llm/plugin.json
   # Change the version from "1.0.0" to "1.0.1"
   ```

2. Wait 2.5 seconds (reload interval)

3. Check the backend logs:
   ```
   INFO:speechscribe.core.plugins.loader:Reloading plugin: ollama_llm
   INFO:speechscribe.core.plugins.loader:Loaded plugin: Ollama LLM (summarization)
   ```

4. Verify via API:
   ```bash
   curl http://localhost:8000/plugins | grep version
   ```

## Troubleshooting

### Plugin Not Loading

**Check logs:**
```
WARNING:speechscribe.core.plugins.loader:Failed to load plugin whisper: ...
```

**Common fixes:**
- Install missing dependencies: `pip install faster-whisper transformers`
- Check `plugin.json` syntax
- Verify entry point matches class name

### Ollama Connection Error

**Error in UI:**
```
Failed to send message: Unable to reach Ollama
```

**Fixes:**
1. Check Ollama is running: `curl http://localhost:11434/api/version`
2. Pull the model: `ollama pull qwen2.5`
3. Check firewall settings

### Frontend Build Error

**Error:**
```
Module not found: react-markdown
```

**Fix:**
```bash
cd web
npm install react-markdown
```

### CORS Error

**Error in browser console:**
```
Access to XMLHttpRequest blocked by CORS policy
```

**Fix:**
Make sure backend is running on port 8000 and has CORS configured:
```python
# In src/speechscribe/api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    ...
)
```

## Next Steps

### Add Custom Plugins

See [Plugin System Documentation](./src/speechscribe/plugins/builtin/README.md)

### Integrate with Pipeline

Use plugins in the pipeline:

```python
from speechscribe.core.plugins import get_plugin_loader

loader = get_plugin_loader()

# Get ASR plugin
asr_plugins = loader.plugins_by_type("asr")
whisper = asr_plugins[0].entry_class()

# Transcribe
result = whisper.transcribe("audio.wav", language="en")
print(result["text"])
```

### Add More Models

Pull additional Ollama models:
```bash
ollama pull llama3
ollama pull deepseek
ollama pull mistral
```

Restart the backend to refresh the model list.

## API Documentation

Once the backend is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
SpeechScribe/
├── speechscribe/
│   ├── core/
│   │   └── plugins/          # Plugin loader
│   ├── plugins/
│   │   ├── ollama_llm/       # Ollama chat/summarization
│   │   ├── whisper/          # ASR transcription
│   │   └── speecht5/         # TTS synthesis
│   └── ui/
│       └── src/
│           └── App.jsx       # React UI with ChatPanel
├── src/
│   └── speechscribe/
│       └── api/
│           └── main.py       # FastAPI backend
└── test_plugins.py           # Plugin system test
```

## Support

For issues or questions:
1. Check the logs (backend and browser console)
2. Review plugin documentation
3. Test with the provided test script

Happy coding! 🚀
