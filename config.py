#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for the distributed translation system.
Handles loading and validation of YAML configuration and command line overrides.
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """
    Handles loading, validating and providing access to configuration.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'input_file': None,
        'output_file': None,
        'columns_to_translate': [],
        'source_language_column': None,
        'target_language': 'english',
        'batch_size': 10,
        'openai': {
            'model': 'gpt-3.5-turbo',
            'temperature': 0.1,
            'max_tokens': 1500,
            'api_key_env': 'OPENAI_API_KEY'
        },
        'checkpoint': {
            'enabled': True,
            'interval': 1,  # Every partition
            'directory': './checkpoints',
            'max_checkpoints': 5
        },
        'cache': {
            'type': 'sqlite',
            'location': './cache/translations.db',
            'ttl': 2592000,  # 30 days in seconds
        },
        'spark': {
            'executor_memory': '4g',
            'driver_memory': '4g',
            'executor_cores': 2,
            'default_parallelism': 4
        },
        'logging': {
            'level': 'INFO',
            'log_file': 'translation_process.log'
        },
        'resume_from_checkpoint': True,
        'write_intermediate_results': False,
        'intermediate_directory': './intermediate',
        'retry': {
            'max_attempts': 3,
            'backoff_factor': 2
        }
    }
    
    def __init__(self, config_path: str, cli_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
            cli_overrides: Dictionary of command-line overrides
        """
        self.config_path = config_path
        self.cli_overrides = cli_overrides or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file and apply overrides.
        
        Returns:
            Dict containing merged configuration
        """
        # Start with default configuration
        config = self.DEFAULT_CONFIG.copy()
        
        # Load from YAML file if provided and exists
        if self.config_path:
            try:
                if os.path.exists(self.config_path):
                    with open(self.config_path, 'r', encoding='utf-8') as config_file:
                        file_config = yaml.safe_load(config_file)
                        if file_config:
                            # Update with file configuration
                            self._deep_update(config, file_config)
                        else:
                            self.logger.warning(f"Empty or invalid configuration file: {self.config_path}")
                else:
                    self.logger.warning(f"Configuration file not found: {self.config_path}")
            except Exception as e:
                self.logger.error(f"Error loading configuration file: {str(e)}")
                # Continue with default configuration
        
        # Apply command-line overrides
        self._apply_cli_overrides(config)
        
        # Validate the configuration
        self._validate_config(config)
        
        return config
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with updates
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                # If both target and source have dict at this key, recursively update
                self._deep_update(target[key], value)
            else:
                # Otherwise, just update the value
                target[key] = value
    
    def _apply_cli_overrides(self, config: Dict[str, Any]) -> None:
        """
        Apply command-line overrides to the configuration.
        
        Args:
            config: Configuration dictionary to update
        """
        for key, value in self.cli_overrides.items():
            # Handle nested keys using dot notation (e.g., 'openai.model')
            if '.' in key:
                parts = key.split('.')
                current = config
                
                # Navigate to the deepest level
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    elif not isinstance(current[part], dict):
                        # If a non-dict exists at this level, replace it with a dict
                        current[part] = {}
                    current = current[part]
                
                # Set the value at the deepest level
                current[parts[-1]] = value
            else:
                # Simple case: top-level override
                config[key] = value
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration and set defaults for missing values.
        
        Args:
            config: Configuration dictionary to validate
        
        Raises:
            ValueError: If critical configuration is missing
        """
        # Validate required fields
        if not config.get('input_file'):
            raise ValueError("Input file path is required in configuration")
        
        if not config.get('output_file'):
            raise ValueError("Output file path is required in configuration")
        
        # Validate columns to translate
        if not config.get('columns_to_translate'):
            raise ValueError("At least one column to translate must be specified")
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(config['output_file'])), exist_ok=True)
        
        checkpoint_dir = config.get('checkpoint', {}).get('directory')
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        cache_location = config.get('cache', {}).get('location')
        if cache_location and config.get('cache', {}).get('type') == 'sqlite':
            os.makedirs(os.path.dirname(os.path.abspath(cache_location)), exist_ok=True)
        
        intermediate_dir = config.get('intermediate_directory')
        if intermediate_dir and config.get('write_intermediate_results'):
            os.makedirs(intermediate_dir, exist_ok=True)
        
        # Validate OpenAI settings
        api_key_env = config.get('openai', {}).get('api_key_env')
        if api_key_env and not os.environ.get(api_key_env):
            self.logger.warning(f"OpenAI API key environment variable '{api_key_env}' not set")
        
        # Validate batch size
        if config.get('batch_size', 0) < 1:
            self.logger.warning("Invalid batch size, setting to default (10)")
            config['batch_size'] = 10
        
        # Validate retry settings
        retry_config = config.get('retry', {})
        if retry_config.get('max_attempts', 0) < 1:
            retry_config['max_attempts'] = 3
        if retry_config.get('backoff_factor', 0) < 0:
            retry_config['backoff_factor'] = 2
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration.
        
        Returns:
            Dict containing the merged and validated configuration
        """
        return self.config
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        if '.' in key:
            # Handle nested keys
            parts = key.split('.')
            current = self.config
            
            # Traverse the nested dictionaries
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
            
            return current
        else:
            # Simple case: top-level key
            return self.config.get(key, default)