"""
Configuration management for VMTranscriber.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for VMTranscriber."""
    
    DEFAULT_CONFIG = {
        'model': 'small',
        'device': 'cpu',
        'output_formats': ['txt', 'srt', 'vtt'],
        'audio_quality': {
            'sample_rate': 16000,
            'channels': 1,
            'convert_to_wav': True
        },
        'transcription': {
            'vad_filter': True,
            'min_silence_duration_ms': 300,
            'translate': False,
            'language': None
        },
        'output': {
            'directory': None,
            'filename_template': '{input_name}_transcript'
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        if config_path is None:
            config_path = self._get_default_config_path()
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        logger.info(f"Configuration loaded from: {self.config_path}")
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path."""
        if os.name == 'nt':  # Windows
            config_dir = Path.home() / 'AppData' / 'Local' / 'VMTranscriber'
        else:  # Unix-like
            config_dir = Path.home() / '.config' / 'vmtranscriber'
        
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / 'config.yaml'
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                    logger.debug("User configuration loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load user config: {e}. Using defaults.")
                user_config = {}
        else:
            user_config = {}
            logger.info("No configuration file found, creating default")
        
        # Merge user config with defaults
        config = self._merge_configs(self.DEFAULT_CONFIG, user_config)
        
        # Save default config if it doesn't exist
        if not self.config_path.exists():
            self._save_config(config)
        
        return config
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user configuration with defaults."""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.debug("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Save the updated configuration
        self._save_config(self.config)
        logger.info(f"Configuration updated: {key} = {value}")
    
    def save(self):
        """Save current configuration to file."""
        self._save_config(self.config)
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.config = self.DEFAULT_CONFIG.copy()
        self._save_config(self.config)
        logger.info("Configuration reset to defaults")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get transcription model configuration."""
        return {
            'model': self.get('model'),
            'device': self.get('device'),
            'vad_filter': self.get('transcription.vad_filter'),
            'min_silence_duration_ms': self.get('transcription.min_silence_duration_ms'),
            'translate': self.get('transcription.translate'),
            'language': self.get('transcription.language')
        }
    
    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio processing configuration."""
        return self.get('audio_quality')
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.get('output')
