# Ollama Plugin Testing Guide

## Prerequisites

1. Install Ollama from https://ollama.ai
2. Pull a model (e.g., qwen2.5):
   ```bash
   ollama pull qwen2.5
   ```

## Starting Ollama

Make sure Ollama is running:
```bash
ollama serve
```

This will start Ollama on `http://localhost:11434`

## Available Models

Pull any of these models:
```bash
ollama pull qwen2.5
ollama pull llama3
ollama pull deepseek
ollama pull llama3.1
ollama pull mistral
```

## Testing the Plugin

### 1. List Plugins

```bash
curl http://localhost:8000/plugins
```

You should see `ollama_llm` in the list.

### 2. Test Chat Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello! Can you help me summarize a meeting?"}
    ],
    "model": "qwen2.5",
    "stream": false
  }'
```

### 3. Test Meeting Summarization

```python
from speechscribe.plugins.ollama_llm.plugin import OllamaPlugin

plugin = OllamaPlugin()

transcript = """
Meeting started at 10:00 AM.
John: We need to finalize the Q4 roadmap.
Sarah: I agree. Let's prioritize the API improvements.
John: Action item: Sarah will lead the API project.
Meeting ended at 10:30 AM.
"""

notes = plugin.generate_meeting_notes(transcript)
print(notes)
```

## UI Testing

1. Start the backend:
   ```bash
   cd src
   uvicorn speechscribe.api.main:app --reload
   ```

2. Start the frontend:
   ```bash
   cd speechscribe/ui
   npm install
   npm run dev
   ```

3. Open http://localhost:5173
4. Click on "AI Chat" in the sidebar
5. Select a model from the dropdown
6. Type a message and press Enter

## Chat Features

- **Model Selection**: Choose from qwen2.5, llama3, deepseek, etc.
- **Markdown Rendering**: Code blocks, lists, and formatting are rendered
- **Chat Bubbles**: User messages appear on the right (blue), AI responses on the left (gray)
- **Auto-scroll**: Chat automatically scrolls to the latest message

## Troubleshooting

### Ollama Connection Error

If you see "Unable to reach Ollama", check:
1. Ollama is running: `ollama serve`
2. The model is pulled: `ollama list`
3. Port 11434 is accessible

### Model Not Found

Pull the model:
```bash
ollama pull qwen2.5
```

### Chat Not Responding

Check the browser console and backend logs for errors.
