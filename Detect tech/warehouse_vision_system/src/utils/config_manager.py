"""
Configuration Manager for Warehouse Vision System
Handles loading, validation, and management of system configuration
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import jsonschema

import structlog

logger = structlog.get_logger()

class ConfigManager:
    """Configuration management with validation and hot-reload capabilities"""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self.schema = None
        self.watchers = []
        
        # Load configuration
        self._load_config()
        self._load_schema()
        self._validate_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    self.config = yaml.safe_load(f)
                elif self.config_path.suffix.lower() == '.json':
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_schema(self):
        """Load configuration schema for validation"""
        schema_path = self.config_path.parent / "config_schema.yaml"
        
        if schema_path.exists():
            try:
                with open(schema_path, 'r') as f:
                    self.schema = yaml.safe_load(f)
                logger.info("Configuration schema loaded")
            except Exception as e:
                logger.warning(f"Failed to load schema: {e}")
                self.schema = None
        else:
            logger.info("No configuration schema found, skipping validation")
            self.schema = None
    
    def _validate_config(self):
        """Validate configuration against schema"""
        if self.schema is None:
            return
        
        try:
            jsonschema.validate(self.config, self.schema)
            logger.info("Configuration validation passed")
        except jsonschema.ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
        
        # Validate if schema exists
        if self.schema:
            try:
                jsonschema.validate(self.config, self.schema)
            except jsonschema.ValidationError as e:
                logger.warning(f"Configuration update validation failed: {e}")
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        for key, value in updates.items():
            self.set(key, value)
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif save_path.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {save_path.suffix}")
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
        self._validate_config()
        logger.info("Configuration reloaded")
    
    def get_camera_config(self, camera_id: int) -> Dict[str, Any]:
        """Get configuration for specific camera"""
        cameras = self.get('cameras', {})
        
        # Generate camera-specific config
        camera_config = {
            'id': camera_id,
            'resolution': cameras.get('resolution', '1080p'),
            'fps': cameras.get('fps', 30),
            'codec': cameras.get('codec', 'h264'),
            'protocol': cameras.get('protocol', 'rtsp')
        }
        
        return camera_config
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific AI model"""
        models = self.get('models', {})
        return models.get(model_name, {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration"""
        return {
            'max_latency_ms': self.get('performance.max_latency_ms', 100),
            'target_fps': self.get('performance.target_fps', 30),
            'max_memory_gb': self.get('performance.max_memory_gb', 6),
            'max_power_watts': self.get('performance.max_power_watts', 15)
        }
    
    def get_deepstream_config(self) -> Dict[str, Any]:
        """Get DeepStream-specific configuration"""
        return self.get('deepstream', {})
    
    def get_reconstruction_config(self) -> Dict[str, Any]:
        """Get 3D reconstruction configuration"""
        return self.get('reconstruction', {})
    
    def get_analytics_config(self) -> Dict[str, Any]:
        """Get analytics configuration"""
        return self.get('analytics', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get('monitoring', {})
    
    def get_failure_handling_config(self) -> Dict[str, Any]:
        """Get failure handling configuration"""
        return self.get('failure_handling', {})
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get('system.debug', False)
    
    def get_log_level(self) -> str:
        """Get configured log level"""
        return self.get('monitoring.log_level', 'INFO')
    
    def add_watcher(self, callback):
        """Add configuration change watcher"""
        self.watchers.append(callback)
    
    def notify_watchers(self):
        """Notify all configuration watchers"""
        for watcher in self.watchers:
            try:
                watcher(self.config)
            except Exception as e:
                logger.error(f"Configuration watcher error: {e}")
    
    def __getitem__(self, key):
        """Dictionary-style access"""
        return self.get(key)
    
    def __setitem__(self, key, value):
        """Dictionary-style assignment"""
        self.set(key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self.config.copy()
    
    def merge(self, other_config: Dict[str, Any]):
        """Merge another configuration dictionary"""
        def deep_merge(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(self.config, other_config)
        self._validate_config()
