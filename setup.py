#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for the Distributed Translation System.
Creates the necessary directory structure and template configuration files.
"""

import os
import sys
import shutil
import yaml

# Define the base directory structure
DIRECTORIES = [
    "input",            # Place input files here
    "output",           # Translated files will be saved here
    "config",           # Configuration files
    "cache",            # Cache for translations
    "checkpoints",      # Checkpoints for resuming interrupted processes
    "logs"              # Log files
]

# Template for OpenAI API key file
OPENAI_KEY_TEMPLATE = """# OpenAI API Key
# Replace the value below with your API key from https://platform.openai.com/api-keys
OPENAI_API_KEY=your-api-key-here
"""

# Template YAML configuration with detailed comments
CONFIG_YAML_TEMPLATE = """# Distributed Translation System Configuration

#--------------------
# FILE PATHS
#--------------------
# Input file to translate (supports .dta, .csv, .parquet, .json)
input_file: "./input/your_file.dta"

# Output file where translated data will be saved (same format as input)
output_file: "./output/your_file_translated.dta"

#--------------------
# TRANSLATION SETTINGS
#--------------------
# List of columns to translate - replace with your actual column names
columns_to_translate: 
  - "column_1"
  - "column_2"
  - "column_3"

# Column that contains the source language code (optional)
# If provided, this column will be used to determine the source language for each row
# If not provided, language will be auto-detected
source_language_column: "language_column"

# Target language to translate into
target_language: "english"

# Number of texts to process in each batch
# Larger values are more efficient but increase memory usage
# Recommended: 10-20 for long texts, 50-100 for short texts
batch_size: 50

#--------------------
# OPENAI API SETTINGS
#--------------------
openai:
  # Model to use for translation
  # Options: "gpt-3.5-turbo" (faster, less accurate), "gpt-4" (slower, more accurate)
  model: "gpt-3.5-turbo"
  
  # Temperature controls randomness (0.0-1.0)
  # Lower values are more deterministic and consistent
  temperature: 0.1
  
  # Maximum tokens in completion (adjust based on text length)
  max_tokens: 1500
  
  # Environment variable name that contains your OpenAI API key
  api_key_env: "OPENAI_API_KEY"

#--------------------
# CACHING SETTINGS
#--------------------
# Caching is used to avoid redundant API calls for identical texts
cache:
  # Cache type: "sqlite" (recommended), "postgres" (for distributed setups), "memory" (for small jobs)
  type: "sqlite"
  
  # Path to the cache database
  location: "./cache/translations.db"
  
  # Cache expiration time in seconds (default: 30 days)
  ttl: 2592000

#--------------------
# CHECKPOINT SETTINGS
#--------------------
# Checkpoints allow resuming interrupted processes
checkpoint:
  # Enable/disable checkpointing
  enabled: true
  
  # Save checkpoint every N partitions
  interval: 1
  
  # Directory to store checkpoints
  directory: "./checkpoints"
  
  # Maximum number of checkpoints to keep
  max_checkpoints: 5

#--------------------
# SPARK SETTINGS
#--------------------
# Performance settings for PySpark processing
# Adjust based on your machine's resources
spark:
  # Memory allocated to each executor
  # Recommended: 50-80% of available RAM divided by number of executor cores
  executor_memory: "4g"
  
  # Memory allocated to the driver
  # Recommended: ~50% of executor_memory
  driver_memory: "2g"
  
  # Number of cores per executor
  # Recommended: Number of CPU cores on your machine minus 1
  executor_cores: 2
  
  # Number of parallel tasks
  # Recommended: 2-4 times the number of executor cores
  default_parallelism: 4

#--------------------
# LOGGING SETTINGS
#--------------------
logging:
  # Logging level: DEBUG, INFO, WARNING, ERROR
  level: "INFO"
  
  # Log file path
  log_file: "./logs/translation_process.log"

#--------------------
# ADVANCED SETTINGS
#--------------------
# Write intermediate results to disk
write_intermediate_results: false

# Directory for intermediate results
intermediate_directory: "./intermediate"

# Retry settings for API calls
retry:
  max_attempts: 3
  backoff_factor: 2
"""

def create_directory_structure():
    """Create the directory structure for the translation system."""
    print("Creating directory structure...")
    
    for directory in DIRECTORIES:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created directory: {directory}")

def create_openai_key_template():
    """Create a template file for the OpenAI API key."""
    key_file = "OPENAI_KEY.env"
    if not os.path.exists(key_file):
        with open(key_file, "w") as f:
            f.write(OPENAI_KEY_TEMPLATE)
        print(f"Created template file: {key_file}")
    else:
        print(f"File already exists: {key_file}")

def create_config_template():
    """Create a template YAML configuration file."""
    config_file = os.path.join("config", "config.yaml")
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            f.write(CONFIG_YAML_TEMPLATE)
        print(f"Created template configuration: {config_file}")
    else:
        print(f"File already exists: {config_file}")

def main():
    """Main entry point for the setup script."""
    print("Setting up Distributed Translation System...")
    
    # Create the directory structure
    create_directory_structure()
    
    # Create template files
    create_openai_key_template()
    create_config_template()
    
    print("\nSetup completed successfully!")
    print("""
Quick Start:
1. Add your OpenAI API key to OPENAI_KEY.env
2. Place your input file in the input/ directory
3. Edit config/config.yaml to specify your input/output files and columns to translate
4. Run: python main.py --config config/config.yaml
""")

if __name__ == "__main__":
    main()