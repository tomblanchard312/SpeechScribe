# SpeechScribe Plugin System

SpeechScribe supports a pluggable architecture for speech models. This allows you to add custom ASR, TTS, translation, and summarization models without modifying core code.

## Plugin Types

- **asr**: Automatic Speech Recognition (transcription)
- **tts**: Text-to-Speech (voice synthesis)
- **translation**: Language translation
- **summarization**: Text summarization and chat

## Plugin Structure

Each plugin is a directory in `speechscribe/plugins/` containing:

```
speechscribe/plugins/
├── my_plugin/
│   ├── plugin.json     # Metadata
│   └── plugin.py       # Implementation
```

### plugin.json

Defines plugin metadata:

```json
{
  "type": "asr",
  "name": "My ASR Plugin",
  "version": "1.0.0",
  "description": "A custom ASR model",
  "entry": "plugin.MyASRPlugin",
  "capabilities": [
    "transcribe",
    "translate"
  ],
  "supported_models": [
    "model-v1",
    "model-v2"
  ]
}
```

**Fields:**
- `type`: Plugin type (asr/tts/translation/summarization)
- `name`: Human-readable name
- `version`: Semantic version
- `description`: Brief description
- `entry`: Python entry point (`module.ClassName`)
- `capabilities`: List of methods the plugin provides
- `supported_models`: List of model names/variants

### plugin.py

Implements the plugin class:

```python
class MyASRPlugin:
    def __init__(self):
        self.model = None
    
    def transcribe(self, audio_path: str, **kwargs) -> dict:
        """Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional options (language, task, etc.)
        
        Returns:
            dict with keys: text, segments, language
        """
        # Your implementation here
        return {
            "text": "Transcribed text",
            "segments": [],
            "language": "en"
        }
```

## Built-in Plugins

### Whisper (ASR)
- **Location**: `speechscribe/plugins/whisper/`
- **Capabilities**: transcribe, translate, detect_language
- **Models**: tiny, base, small, medium, large-v2, large-v3
- **Dependencies**: faster-whisper

### SpeechT5 (TTS)
- **Location**: `speechscribe/plugins/speecht5/`
- **Capabilities**: synthesize, text_to_speech
- **Models**: microsoft/speecht5_tts
- **Dependencies**: transformers, datasets

### Ollama LLM (Summarization)
- **Location**: `speechscribe/plugins/ollama_llm/`
- **Capabilities**: generate, chat, summarize, meeting_notes
- **Models**: qwen2.5, llama3, deepseek, llama3.1, mistral
- **Dependencies**: requests, ollama (external)

## Creating a Custom Plugin

### 1. Create Plugin Directory

```bash
mkdir -p speechscribe/plugins/my_plugin
cd speechscribe/plugins/my_plugin
```

### 2. Create plugin.json

```json
{
  "type": "tts",
  "name": "My TTS Plugin",
  "version": "1.0.0",
  "description": "Custom TTS using MyModel",
  "entry": "plugin.MyTTSPlugin",
  "capabilities": ["synthesize"],
  "supported_models": ["custom-v1"]
}
```

### 3. Create plugin.py

```python
import logging

logger = logging.getLogger(__name__)

class MyTTSPlugin:
    def __init__(self):
        logger.info("Initializing MyTTS Plugin")
        # Load your model here
    
    def synthesize(self, text: str, output_path: str, **kwargs) -> str:
        """Convert text to speech.
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio
            **kwargs: Additional options (voice, speed, etc.)
        
        Returns:
            Path to generated audio file
        """
        # Your TTS implementation
        return output_path
```

### 4. Test Your Plugin

The plugin loader will automatically discover your plugin on startup. Check the logs:

```
INFO:speechscribe.core.plugins.loader:Loaded plugin: My TTS Plugin (tts)
```

Test via API:

```bash
curl http://localhost:8000/plugins
```

## Plugin Loader

The plugin loader runs in a background thread and:

1. Scans `speechscribe/plugins/` directory
2. Finds subdirectories with `plugin.json`
3. Validates metadata and entry point
4. Dynamically imports `plugin.py`
5. Tracks file modification times
6. Hot-reloads on changes (every 2.5 seconds)

### Usage in Code

```python
from speechscribe.core.plugins import get_plugin_loader

# Get loader instance
loader = get_plugin_loader()

# List all plugins
for descriptor in loader.plugins():
    print(f"{descriptor.name} ({descriptor.type})")

# Get plugins by type
asr_plugins = loader.plugins_by_type("asr")

# Instantiate a plugin
for desc in asr_plugins:
    if desc.plugin_id == "whisper":
        plugin_instance = desc.entry_class()
        result = plugin_instance.transcribe("audio.wav")
        print(result)
```

## Best Practices

### Error Handling

Always wrap plugin code in try/except:

```python
def transcribe(self, audio_path: str, **kwargs) -> dict:
    try:
        # Your code here
        return {"text": "..."}
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise
```

### Dependencies

Document dependencies in a `requirements.txt` inside your plugin:

```
speechscribe/plugins/my_plugin/
├── plugin.json
├── plugin.py
└── requirements.txt
```

### Logging

Use Python's logging module:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Processing audio...")
logger.warning("Model not loaded, using default")
logger.error(f"Failed to load audio: {e}")
```

### Resource Management

Load heavy models lazily:

```python
class MyPlugin:
    def __init__(self):
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            logger.info("Loading model...")
            self._model = load_heavy_model()
        return self._model
    
    def transcribe(self, audio_path: str, **kwargs) -> dict:
        model = self._load_model()
        # Use model...
```

## API Integration

Plugins are exposed via the REST API:

### GET /plugins

Lists all loaded plugins:

```json
[
  {
    "plugin_id": "whisper",
    "name": "Whisper ASR",
    "type": "asr",
    "version": "1.0.0",
    "description": "OpenAI Whisper transcription",
    "capabilities": ["transcribe", "translate", "detect_language"],
    "models": ["tiny", "base", "small", "medium", "large-v3"]
  }
]
```

### POST /chat (for summarization plugins)

Chat with LLM plugins:

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

## Troubleshooting

### Plugin Not Loading

Check logs for errors:
```
WARNING:speechscribe.core.plugins.loader:Failed to load plugin my_plugin: ...
```

Common issues:
- Invalid `plugin.json` syntax
- Entry point class not found
- Import errors in `plugin.py`
- Missing dependencies

### Hot Reload Not Working

- Check file timestamps are updating
- Ensure no syntax errors in modified files
- Look for import errors in logs

### Module Import Errors

If you see `ModuleNotFoundError`, ensure:
1. Plugin directory is in `speechscribe/plugins/`
2. `plugin.py` exists
3. Dependencies are installed

## Examples

See the built-in plugins for reference:
- [Whisper Plugin](./whisper/) - ASR with faster-whisper
- [SpeechT5 Plugin](./speecht5/) - TTS with HuggingFace
- [Ollama Plugin](./ollama_llm/) - LLM chat and summarization
