import importlib
import importlib.util
import json
import logging
import threading
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PluginDescriptor:
    name: str
    type: str
    entry: str
    description: str
    path: Path
    metadata: Dict[str, Any]
    last_modified: float
    module_name: str
    class_name: str

    @property
    def plugin_id(self) -> str:
        """Return the plugin ID (directory name)."""
        return self.path.name

    @property
    def version(self) -> str:
        """Return the plugin version from metadata."""
        return self.metadata.get("version", "1.0.0")

    @property
    def capabilities(self) -> List[str]:
        """Return the list of capabilities from metadata."""
        return self.metadata.get("capabilities", [])

    @property
    def supported_models(self) -> List[str]:
        """Return the list of supported models from metadata."""
        return self.metadata.get("supported_models", [])

    @property
    def entry_class(self):
        """Return the entry class."""
        # This will be resolved by the loader
        # For now, return the class name
        return self.__dict__.get("_entry_class")

    def set_entry_class(self, cls):
        """Set the entry class."""
        self.__dict__["_entry_class"] = cls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "entry": self.entry,
            "description": self.description,
            "metadata": self.metadata,
            "last_modified": self.last_modified,
            "module": self.module_name,
            "class": self.class_name,
            "path": str(self.path),
        }


class PluginLoader:
    SUPPORTED_TYPES = {"asr", "tts", "summarization", "translation", "llm"}
    WATCH_INTERVAL = 2.5

    def __init__(self, plugin_root: Optional[Path] = None):
        self.plugin_root = (
            plugin_root or Path(__file__).resolve().parents[2] / "plugins"
        )
        self.plugin_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._descriptors: Dict[str, PluginDescriptor] = {}
        self._mtimes: Dict[str, float] = {}
        self._stop_event = threading.Event()
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        self.reload_plugins()

    def _watch_loop(self) -> None:
        while not self._stop_event.wait(self.WATCH_INTERVAL):
            self.reload_plugins()

    def stop(self) -> None:
        self._stop_event.set()
        self._watch_thread.join(timeout=1.0)

    def reload_plugins(self) -> None:
        with self._lock:
            discovered = set()
            for plugin_dir in sorted(self.plugin_root.iterdir()):
                if not plugin_dir.is_dir():
                    continue
                plugin_json = plugin_dir / "plugin.json"
                plugin_py = plugin_dir / "plugin.py"
                if not plugin_json.exists() or not plugin_py.exists():
                    continue

                last_modified = max(
                    plugin_json.stat().st_mtime, plugin_py.stat().st_mtime
                )
                plugin_id = plugin_dir.name
                discovered.add(plugin_id)
                if self._mtimes.get(plugin_id) == last_modified:
                    continue

                descriptor = self._try_build_descriptor(plugin_dir, last_modified)
                if descriptor is None:
                    continue
                self._descriptors[plugin_id] = descriptor
                self._mtimes[plugin_id] = descriptor.last_modified
                logger.info("Loaded plugin %s (%s)", descriptor.name, descriptor.type)

            stale_names = set(self._descriptors) - discovered
            for stale in stale_names:
                logger.info("Removing stale plugin %s", stale)
                self._descriptors.pop(stale, None)
                self._mtimes.pop(stale, None)

    def _try_build_descriptor(
        self, plugin_dir: Path, last_modified: float
    ) -> Optional[PluginDescriptor]:
        plugin_json = plugin_dir / "plugin.json"
        plugin_py = plugin_dir / "plugin.py"
        if not plugin_json.exists() or not plugin_py.exists():
            return None

        try:
            metadata = json.loads(plugin_json.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Skipped plugin %s: invalid JSON (%s)", plugin_dir.name, exc)
            return None

        plugin_type = metadata.get("type")
        if plugin_type not in self.SUPPORTED_TYPES:
            logger.warning(
                "Plugin %s has unsupported type %s", plugin_dir.name, plugin_type
            )
            return None

        entry = metadata.get("entry")
        if not entry:
            logger.warning("Plugin %s is missing entry definition", plugin_dir.name)
            return None

        module_name, class_name = self._split_entry(entry)
        module = self._load_plugin_module(plugin_py, plugin_dir.name)
        if module is None:
            return None

        entry_module = self._resolve_entry_module(module, module_name)
        if entry_module is None or not hasattr(entry_module, class_name):
            logger.warning(
                "Plugin %s entry %s cannot be resolved", plugin_dir.name, entry
            )
            return None

        # Get the entry class
        entry_class = getattr(entry_module, class_name)

        descriptor = PluginDescriptor(
            name=metadata.get("name", plugin_dir.name),
            type=plugin_type,
            entry=entry,
            description=metadata.get("description", ""),
            path=plugin_dir,
            metadata=metadata,
            last_modified=last_modified,
            module_name=module_name,
            class_name=class_name,
        )
        descriptor.set_entry_class(entry_class)
        return descriptor

    def _split_entry(self, entry: str) -> Tuple[str, str]:
        if "." not in entry:
            return "plugin", entry
        module_name, class_name = entry.rsplit(".", 1)
        return module_name, class_name

    def _load_plugin_module(
        self, plugin_py: Path, plugin_name: str
    ) -> Optional[ModuleType]:
        spec_name = f"speechscribe.plugins.{plugin_name}.plugin"
        try:
            spec = importlib.util.spec_from_file_location(spec_name, plugin_py)
            if spec is None or spec.loader is None:
                raise ImportError("Invalid plugin module specification")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Unable to load plugin module %s: %s", plugin_name, exc)
            return None

    def _resolve_entry_module(
        self, root_module: ModuleType, module_name: str
    ) -> Optional[ModuleType]:
        if module_name.startswith("plugin"):
            module = root_module
            for part in module_name.split(".")[1:]:
                module = getattr(module, part, None)
                if module is None:
                    return None
            return module

        try:
            return importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - optional dependency
            logger.warning(
                "Unable to import module %s for plugin entry: %s", module_name, exc
            )
            return None

    def plugins(self) -> List[PluginDescriptor]:
        self.reload_plugins()
        with self._lock:
            return sorted(self._descriptors.values(), key=lambda desc: desc.name)

    def plugins_by_type(self, plugin_type: str) -> List[PluginDescriptor]:
        """Get all plugins of a specific type."""
        return [p for p in self.plugins() if p.type == plugin_type]


_LOADER: Optional[PluginLoader] = None


def get_plugin_loader(plugin_root: Optional[Path] = None) -> PluginLoader:
    global _LOADER
    if _LOADER is None:
        _LOADER = PluginLoader(plugin_root=plugin_root)
    return _LOADER
