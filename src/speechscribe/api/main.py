import inspect
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Running this file directly needs src/ on the path so `speechscribe` resolves.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speechscribe.plugins.loader import get_plugin_loader
from speechscribe.plugins.settings import (
    PluginSettingsError,
    get_settings_store,
    normalize_schema,
    validate_settings,
)

logger = logging.getLogger("speechscribe.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

app = FastAPI(
    title="SpeechScribe API",
    description="Thin API layer for SpeechScribe plugins.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loader = get_plugin_loader()
settings_store = get_settings_store()


class TranscribeResponse(BaseModel):
    transcript: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # Either a single message or a full conversation may be sent. The UI sends
    # `messages` so the model keeps context across turns.
    message: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    model: Optional[str] = None
    plugin: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    model: Optional[str] = None


class PluginInfo(BaseModel):
    plugin_id: str
    name: str
    type: str
    version: str
    description: str
    capabilities: List[str]
    supported_models: List[str]
    default_model: Optional[str] = None
    settings_schema: List[Dict[str, Any]]
    settings: Dict[str, Any]


class PluginSettingsPayload(BaseModel):
    settings: Dict[str, Any]


class PluginHealth(BaseModel):
    plugin_id: str
    available: bool
    detail: str
    endpoint: Optional[str] = None
    version: Optional[str] = None
    models: List[str] = []


def describe_plugin(descriptor) -> PluginInfo:
    schema = normalize_schema(descriptor.metadata.get("settings_schema"))
    return PluginInfo(
        plugin_id=descriptor.plugin_id,
        name=descriptor.name,
        type=descriptor.type,
        version=descriptor.version,
        description=descriptor.description,
        capabilities=descriptor.capabilities,
        supported_models=descriptor.supported_models,
        default_model=descriptor.metadata.get("default_model"),
        settings_schema=schema,
        settings=settings_store.resolved(descriptor.plugin_id, schema),
    )


def find_descriptor(plugin_id: str):
    for descriptor in loader.plugins():
        if descriptor.plugin_id == plugin_id:
            return descriptor
    raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")


def instantiate(descriptor):
    """Build a plugin instance, applying the user's saved settings.

    Only settings the constructor actually accepts are passed, so a stale
    stored key cannot break instantiation.
    """
    entry_class = descriptor.entry_class
    schema = normalize_schema(descriptor.metadata.get("settings_schema"))
    resolved = settings_store.resolved(descriptor.plugin_id, schema)

    try:
        parameters = inspect.signature(entry_class).parameters
        kwargs = {
            key: value
            for key, value in resolved.items()
            if key in parameters and value is not None
        }
    except (TypeError, ValueError):
        kwargs = {}

    try:
        return entry_class(**kwargs)
    except Exception as e:
        logger.error(
            "Failed to instantiate %s plugin %s: %s",
            descriptor.type,
            descriptor.name,
            e,
        )
        raise HTTPException(status_code=500, detail=f"Plugin instantiation failed: {e}")


def get_plugin_by_type(plugin_type: str, name: Optional[str] = None):
    plugins = loader.plugins_by_type(plugin_type)
    if not plugins:
        raise HTTPException(
            status_code=500, detail=f"No {plugin_type} plugins available"
        )

    if name:
        for descriptor in plugins:
            if name in (descriptor.name, descriptor.plugin_id):
                return instantiate(descriptor)
        raise HTTPException(
            status_code=404, detail=f"{plugin_type} plugin '{name}' not found"
        )

    return instantiate(plugins[0])


def conversation_from(request: ChatRequest) -> List[Dict[str, str]]:
    """Normalize a chat request into an Ollama-style message list."""
    if request.messages:
        messages = [
            {"role": m.role, "content": m.content.strip()}
            for m in request.messages
            if m.content and m.content.strip()
        ]
        if messages:
            return messages

    if request.message and request.message.strip():
        return [{"role": "user", "content": request.message.strip()}]

    raise HTTPException(status_code=400, detail="Message cannot be empty")


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscribeResponse:
    logger.info("Received /transcribe request")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(file.filename).suffix or ".tmp"
    temp_path: Optional[Path] = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            content = await file.read()
            tmp.write(content)

        asr_plugin = get_plugin_by_type("asr")
        # Transcription is CPU-bound and blocking; keep it off the event loop.
        result = await run_in_threadpool(asr_plugin.transcribe, str(temp_path))

        if isinstance(result, dict):
            transcript = str(result.get("text", "")).strip()
        elif isinstance(result, str):
            transcript = result.strip()
        else:
            transcript = ""

        if not transcript:
            raise HTTPException(
                status_code=500, detail="Transcription failed: no text produced"
            )

        return TranscribeResponse(transcript=transcript)

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Invalid audio file")
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if temp_path:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to cleanup temp file %s: %s", temp_path, e)


# Declared sync on purpose: the plugin call blocks, so FastAPI runs it in a
# threadpool instead of stalling the event loop for every other request.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    logger.info("Received /chat request")

    messages = conversation_from(request)

    try:
        llm_plugin = get_plugin_by_type("llm", request.plugin)
        response = (
            llm_plugin.chat(messages, model=request.model)
            if request.model
            else llm_plugin.chat(messages)
        )

        if not response:
            raise HTTPException(
                status_code=500, detail="Chat failed: no response produced"
            )

        return ChatResponse(response=str(response), model=request.model)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


def sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream a chat completion to the UI as server-sent events.

    Errors are emitted as an `error` event rather than an HTTP status, because
    the response has already started by the time most failures surface.
    """
    logger.info("Received /chat/stream request")

    messages = conversation_from(request)
    llm_plugin = get_plugin_by_type("llm", request.plugin)

    if not hasattr(llm_plugin, "stream_chat"):
        raise HTTPException(
            status_code=501, detail="Selected LLM plugin does not support streaming"
        )

    def event_stream() -> Iterator[str]:
        produced = False
        try:
            for token in llm_plugin.stream_chat(messages, model=request.model):
                produced = True
                yield sse("token", {"content": token})
        except Exception as e:  # surfaced to the user in the chat panel
            logger.exception("Streaming chat failed")
            yield sse("error", {"detail": str(e)})
            return

        if not produced:
            yield sse("error", {"detail": "No response produced"})
            return

        yield sse("done", {"model": request.model})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/plugins", response_model=List[PluginInfo])
def list_plugins(type: Optional[str] = None) -> List[PluginInfo]:
    logger.info("Received /plugins request")

    try:
        descriptors = loader.plugins_by_type(type) if type else loader.plugins()
        return [describe_plugin(descriptor) for descriptor in descriptors]
    except Exception as e:
        logger.exception("Failed to list plugins")
        raise HTTPException(status_code=500, detail=f"Failed to list plugins: {str(e)}")


@app.get("/plugins/{plugin_id}", response_model=PluginInfo)
def get_plugin(plugin_id: str) -> PluginInfo:
    return describe_plugin(find_descriptor(plugin_id))


@app.put("/plugins/{plugin_id}/settings", response_model=PluginInfo)
def update_plugin_settings(
    plugin_id: str, payload: PluginSettingsPayload
) -> PluginInfo:
    """Persist plugin settings chosen in the UI."""
    descriptor = find_descriptor(plugin_id)
    schema = normalize_schema(descriptor.metadata.get("settings_schema"))

    if not schema:
        raise HTTPException(
            status_code=400, detail=f"Plugin '{plugin_id}' has no configurable settings"
        )

    try:
        validated = validate_settings(schema, payload.settings)
    except PluginSettingsError as e:
        raise HTTPException(status_code=400, detail=str(e))

    settings_store.set(plugin_id, validated)
    logger.info("Updated settings for plugin %s", plugin_id)
    return describe_plugin(descriptor)


@app.delete("/plugins/{plugin_id}/settings", response_model=PluginInfo)
def reset_plugin_settings(plugin_id: str) -> PluginInfo:
    """Drop saved overrides, reverting the plugin to its manifest defaults."""
    descriptor = find_descriptor(plugin_id)
    settings_store.clear(plugin_id)
    return describe_plugin(descriptor)


# Sync for the same reason as /chat: the health probe performs network I/O.
@app.get("/plugins/{plugin_id}/health", response_model=PluginHealth)
def plugin_health(plugin_id: str) -> PluginHealth:
    """Check whether a plugin's backing service is reachable.

    Used by the chat panel to explain an Ollama outage before the user types.
    """
    descriptor = find_descriptor(plugin_id)

    try:
        instance = instantiate(descriptor)
    except HTTPException as e:
        return PluginHealth(plugin_id=plugin_id, available=False, detail=str(e.detail))

    if not hasattr(instance, "health"):
        return PluginHealth(
            plugin_id=plugin_id,
            available=True,
            detail="Plugin loaded (no health check available)",
        )

    status = instance.health()
    models: List[str] = []
    if status.get("available") and hasattr(instance, "list_models"):
        models = instance.list_models()

    return PluginHealth(
        plugin_id=plugin_id,
        available=bool(status.get("available")),
        detail=str(status.get("detail", "")),
        endpoint=status.get("endpoint"),
        version=status.get("version"),
        models=models,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "SpeechScribe API is running", "version": "1.0.0"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response

    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
