"""Persisted, per-plugin settings.

Plugins declare the options they accept in their ``plugin.json`` under a
``settings_schema`` key. Values chosen by the user are stored in a single JSON
file so they survive restarts and can be edited from the web UI.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _default_settings_path() -> Path:
    """Where plugin settings are stored.

    Settings are written at runtime, so they must not land inside the installed
    package. SPEECHSCRIBE_CONFIG_DIR overrides the location; otherwise they go
    in the repository's config/ directory when running from a source checkout,
    falling back to the user's config directory when installed.
    """
    override = os.environ.get("SPEECHSCRIBE_CONFIG_DIR")
    if override:
        return Path(override).expanduser() / "plugin_settings.json"

    repo_config = Path(__file__).resolve().parents[3] / "config"
    if repo_config.is_dir():
        return repo_config / "plugin_settings.json"

    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "speechscribe" / "plugin_settings.json"


DEFAULT_SETTINGS_PATH = _default_settings_path()

# Field types a plugin may declare in its settings_schema.
VALID_FIELD_TYPES = {"string", "integer", "number", "boolean", "select"}


class PluginSettingsError(ValueError):
    """Raised when a submitted settings payload does not match the schema."""


class PluginSettingsStore:
    """Reads and writes plugin settings from a JSON file."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DEFAULT_SETTINGS_PATH
        self._lock = threading.RLock()
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self._cache is not None:
            return self._cache

        if not self.path.exists():
            self._cache = {}
            return self._cache

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("settings file must contain a JSON object")
            self._cache = {
                key: value for key, value in data.items() if isinstance(value, dict)
            }
        except Exception as exc:
            logger.warning("Unable to read plugin settings from %s: %s", self.path, exc)
            self._cache = {}

        return self._cache

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._cache or {}, indent=2, sort_keys=True)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(payload + "\n", encoding="utf-8")
        tmp_path.replace(self.path)

    def get(self, plugin_id: str) -> Dict[str, Any]:
        """Return the stored overrides for a plugin (may be empty)."""
        with self._lock:
            return dict(self._load().get(plugin_id, {}))

    def set(self, plugin_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
        """Replace the stored overrides for a plugin and persist them."""
        with self._lock:
            store = self._load()
            store[plugin_id] = dict(values)
            self._flush()
            return dict(store[plugin_id])

    def clear(self, plugin_id: str) -> None:
        """Drop the stored overrides for a plugin, reverting it to defaults."""
        with self._lock:
            store = self._load()
            if store.pop(plugin_id, None) is not None:
                self._flush()

    def resolved(self, plugin_id: str, schema: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge schema defaults with the stored overrides for a plugin."""
        resolved: Dict[str, Any] = {}
        for field in schema:
            key = field.get("key")
            if key:
                resolved[key] = field.get("default")

        for key, value in self.get(plugin_id).items():
            if key in resolved or not schema:
                resolved[key] = value

        return resolved


def normalize_schema(raw_schema: Any) -> List[Dict[str, Any]]:
    """Return a plugin's settings_schema as a clean list of field definitions.

    Unknown or malformed entries are dropped rather than raising, so one bad
    manifest cannot take the whole plugin listing down.
    """
    if not isinstance(raw_schema, list):
        return []

    fields: List[Dict[str, Any]] = []
    for entry in raw_schema:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if not key or not isinstance(key, str):
            continue

        field_type = entry.get("type", "string")
        if field_type not in VALID_FIELD_TYPES:
            field_type = "string"

        field: Dict[str, Any] = {
            "key": key,
            "type": field_type,
            "label": entry.get("label", key.replace("_", " ").title()),
            "description": entry.get("description", ""),
            "default": entry.get("default"),
        }
        if field_type == "select":
            options = entry.get("options")
            field["options"] = (
                [str(o) for o in options] if isinstance(options, list) else []
            )

        fields.append(field)

    return fields


def coerce_value(field: Dict[str, Any], value: Any) -> Any:
    """Coerce and validate one submitted value against its field definition."""
    key = field["key"]
    field_type = field["type"]

    if value is None:
        return field.get("default")

    try:
        if field_type == "integer":
            return int(value)
        if field_type == "number":
            return float(value)
        if field_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
    except (TypeError, ValueError):
        raise PluginSettingsError(f"'{key}' must be of type {field_type}")

    text = str(value)
    if field_type == "select":
        options = field.get("options") or []
        if options and text not in options:
            raise PluginSettingsError(f"'{key}' must be one of: {', '.join(options)}")
    return text


def validate_settings(
    schema: List[Dict[str, Any]], submitted: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate a submitted payload, returning only known, coerced fields."""
    if not isinstance(submitted, dict):
        raise PluginSettingsError("Settings payload must be an object")

    by_key = {field["key"]: field for field in schema}
    unknown = sorted(set(submitted) - set(by_key))
    if unknown:
        raise PluginSettingsError(f"Unknown setting(s): {', '.join(unknown)}")

    return {key: coerce_value(by_key[key], value) for key, value in submitted.items()}


_STORE: Optional[PluginSettingsStore] = None


def get_settings_store(path: Optional[Path] = None) -> PluginSettingsStore:
    global _STORE
    if _STORE is None:
        _STORE = PluginSettingsStore(path=path)
    return _STORE
