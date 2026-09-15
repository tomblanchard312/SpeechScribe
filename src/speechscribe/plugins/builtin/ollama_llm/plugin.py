import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaPlugin:
    """Ollama LLM plugin for summarization, chat, and meeting notes."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def health(self) -> Dict[str, Any]:
        """Report whether the Ollama server is reachable.

        Never raises: the UI calls this to decide what to tell the user before
        they send a message.
        """
        url = f"{self.base_url}/api/version"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            version = response.json().get("version", "unknown")
            return {
                "available": True,
                "endpoint": self.base_url,
                "version": version,
                "detail": f"Ollama {version} is running",
            }
        except requests.exceptions.RequestException as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return {
                "available": False,
                "endpoint": self.base_url,
                "version": None,
                "detail": (
                    f"Cannot reach Ollama at {self.base_url}. "
                    "Start it with 'ollama serve'."
                ),
            }

    def list_models(self) -> List[str]:
        """Return the models actually pulled on the Ollama server.

        Falls back to an empty list when the server is unreachable so callers
        can degrade to their configured defaults.
        """
        url = f"{self.base_url}/api/tags"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            return sorted(
                {
                    m.get("name", "")
                    for m in models
                    if isinstance(m, dict) and m.get("name")
                }
            )
        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.warning("Unable to list Ollama models: %s", exc)
            return []

    def _make_request(
        self, url: str, payload: Dict[str, Any], stream: bool = False
    ) -> requests.Response:
        """Make HTTP request with retry logic."""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url, json=payload, timeout=self.timeout, stream=stream
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                last_exception = exc
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Ollama request failed (attempt {attempt + 1}/{self.max_retries}): {exc}"
                    )
                    time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff
                else:
                    logger.error(
                        f"Ollama request failed after {self.max_retries} attempts: {exc}"
                    )

        # The full exception is already logged above; keep the raised message
        # short enough to show a user directly.
        raise RuntimeError(
            f"Ollama at {self.base_url} is not reachable "
            f"(failed after {self.max_retries} attempts). "
            "Check that it is running with 'ollama serve'."
        )

    def generate(
        self, prompt: str, model: Optional[str] = None, system: Optional[str] = None
    ) -> str:
        """Generate text using Ollama API."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        response = self._make_request(url, payload, stream=False)
        result = response.json()
        return result.get("response", "")

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """Chat using Ollama API with conversation history."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
        }

        response = self._make_request(url, payload, stream=False)
        result = response.json()
        return result.get("message", {}).get("content", "")

    def stream_chat(
        self, messages: List[Dict[str, str]], model: Optional[str] = None
    ) -> Iterator[str]:
        """Stream a chat completion token by token."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
        }

        response = self._make_request(url, payload, stream=True)

        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
            if chunk.get("done"):
                break

    def summarize(self, text: str) -> str:
        """Summarize text using Ollama."""
        system_prompt = (
            "You are a helpful assistant that creates concise summaries. "
            "Focus on key points and main ideas."
        )
        prompt = f"Please summarize the following text:\n\n{text}"
        return self.generate(prompt, system=system_prompt)

    def stream_generate(
        self, prompt: str, model: Optional[str] = None, system: Optional[str] = None
    ) -> Iterator[str]:
        """Stream text generation from Ollama API."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            payload["system"] = system

        response = self._make_request(url, payload, stream=True)

        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
                except json.JSONDecodeError:
                    continue
