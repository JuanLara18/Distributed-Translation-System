# Distributed Translation System

A powerful and scalable solution for translating text columns in large datasets using OpenAI's language models with PySpark distributed processing.

## What It Does

This system allows you to translate text columns in data files (Stata, CSV, Parquet, JSON) with these key features:

- **Distributed processing** with PySpark for handling large datasets efficiently
- **Smart caching** to avoid redundant API calls and reduce costs
- **Fault tolerance** with checkpointing to resume interrupted processes
- **Multiple file formats** with preserved metadata (especially for Stata files)
- **Language detection** for automatic source language identification
- **Batch processing** for optimized throughput

## Prerequisites

- Python 3.7+
- Java 8+ (for PySpark)
- OpenAI API key

## Installation

```bash
# Clone the repository
git clone https://github.com/JuanLara18/Distributed-Translation-System.git
cd Distributed-Translation-System

# Run the setup script to create directories and template files
python setup.py

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### Step 1: Prepare Your Data
Place your input file (Stata, CSV, Parquet, or JSON) in the `input/` directory.

### Step 2: Configure the Translation
Edit the config file at `config/config.yaml`:

```yaml
# Specify input and output files
input_file: "./input/your_file.dta"
output_file: "./output/your_file_translated.dta"

# Specify columns to translate
columns_to_translate: 
  - "column_name_1"
  - "column_name_2"

# Specify source language column (optional)
source_language_column: "language_column_name"

# Specify target language
target_language: "english"
```

### Step 3: Run the Translation

```bash
# Load your API key into the environment
source OPENAI_KEY.env  # On Windows: set /p OPENAI_API_KEY=<OPENAI_KEY.env

# Run the translation process
python main.py --config config/config.yaml
```

### Step 4: Get Your Translated Data
The translated output will be saved to the location specified in your config file, typically in the `output/` directory.

## Repository Structure

```
distributed-translation/
├── main.py                 # Main orchestrator script
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── README.md               # This document
└── modules/                # Core functionality modules
    ├── __init__.py
    ├── translator.py       # Translation logic and API integration
    ├── cache.py            # Translation caching implementation
    ├── checkpoint.py       # Process state management for fault tolerance
    ├── file_manager.py     # Data I/O operations for various formats
    └── utilities.py        # Common utility functions
```

## Architecture

The system uses a modular architecture with well-defined interfaces:

- **TranslationOrchestrator** (in `main.py`): Central controller coordinating all components
- **ConfigManager** (in `config.py`): Manages configuration loading and validation
- **TranslationManager** (in `modules/translator.py`): Manages the translation process
- **CacheManager** (in `modules/cache.py`): Coordinates caching operations
- **CheckpointManager** (in `modules/checkpoint.py`): Handles state persistence
- **DataReader/DataWriter** (in `modules/file_manager.py`): Handle I/O operations

The system follows a flow where:
1. Data is read from input files
2. Translation is applied to specified columns
3. Results are cached to avoid redundant API calls
4. Results are checkpointed for fault tolerance
5. Translated data is written to output files


## Command-line Options

The script supports several command-line options to override configuration:

```bash
# Basic usage
python main.py --config config.yaml

# Override configuration settings
python main.py --config config.yaml --input_file new_input.dta --target_language spanish

# Resume from checkpoint after interruption
python main.py --config config.yaml --resume

# Force restart (ignore existing checkpoints)
python main.py --config config.yaml --force_restart

# Enable verbose logging
python main.py --config config.yaml --verbose
```

## Configuration Options

### Essential Settings

| Setting | Description | Example |
|---------|-------------|---------|
| `input_file` | Path to your input data file | `"data/input.dta"` |
| `output_file` | Where to save the translated data | `"data/output.dta"` |
| `columns_to_translate` | List of columns to translate | `["text_col", "description_col"]` |
| `target_language` | Language to translate into | `"english"` |
| `source_language_column` | Column containing source languages (optional) | `"language_col"` |

### OpenAI Settings

```yaml
openai:
  model: "gpt-3.5-turbo"  # or "gpt-4" for higher quality
  temperature: 0.1  # Lower for more consistent translations
  max_tokens: 1500
  api_key_env: "OPENAI_API_KEY"
```

### Performance Tuning

```yaml
# For smaller datasets (<100MB)
spark:
  executor_memory: "2g"
  driver_memory: "2g"
  executor_cores: 1
  default_parallelism: 2

# For larger datasets (>1GB)
spark:
  executor_memory: "8g"
  driver_memory: "6g"
  executor_cores: 4
  default_parallelism: 8
```

### Caching Options

Caching significantly improves performance and reduces API costs by storing translations:

```yaml
cache:
  type: "sqlite"  # Options: "sqlite", "postgres", "memory"
  location: "./cache/translations.db"  # for SQLite
  ttl: 2592000  # Cache lifetime in seconds (30 days)
```

For PostgreSQL cache:
```yaml
cache:
  type: "postgres"
  connection_string: "postgresql://user:password@localhost/translations"
```

### Checkpoint Settings

Checkpointing enables fault tolerance and the ability to resume interrupted processes:

```yaml
checkpoint:
  enabled: true
  interval: 1  # Save every N partitions
  directory: "./checkpoints"
  max_checkpoints: 5
```

## How the Translation Process Works

1. The system reads your input file and splits it into partitions for distributed processing
2. For each text column to translate:
   - First checks if the translation is already in the cache
   - If not cached, sends text to OpenAI's API
   - Saves results to cache to avoid future redundant API calls
3. Translations are added as new columns with `_[target_language]` suffix
4. The processed data is written to your output file

## Advanced Topics

### Working with Large Datasets

For large datasets, consider:

1. **Increasing partitions**: Set higher `default_parallelism` in Spark config
2. **Using checkpointing**: Enable checkpoints to resume after interruptions
3. **PostgreSQL cache**: For multi-node setups, use a central PostgreSQL cache instead of SQLite

Example configuration for large datasets:

```yaml
spark:
  executor_memory: "10g"
  driver_memory: "8g"
  executor_cores: 6
  default_parallelism: 12

cache:
  type: "postgres"
  connection_string: "postgresql://user:password@centraldb/translations"

checkpoint:
  enabled: true
  interval: 2
  directory: "/shared/checkpoints"
```

### Supporting Stata Files

The system includes special support for Stata (.dta) files:

- Preserves variable labels and value labels
- Handles metadata correctly
- Supports different Stata versions (13-18)

### Custom Language Detection

If the `source_language_column` is not specified, the system automatically detects source languages, but you can customize detection behavior:

```yaml
# Automatic detection (default)
source_language_column: null

# Use a specific column for source language
source_language_column: "language_code"
```

## Troubleshooting

### Common Issues

- **API Key issues**: Make sure your OpenAI API key is set in your environment
- **Memory errors**: Reduce `batch_size` or increase `spark.executor_memory`
- **Missing translations**: Verify source language detection or specify a source language column
- **Corrupted checkpoints**: Use `--force_restart` to start fresh

### Enabling Debug Mode

```yaml
logging:
  level: "DEBUG"
  log_file: "debug.log"
```

## Performance Optimization

1. **Caching Strategy**: 
   - Use SQLite for single-machine processing
   - Use PostgreSQL for distributed setups
   - Consider memory cache only for small datasets

2. **Batch Size Tuning**:
   - Start with `batch_size: 10`
   - Decrease for large texts, increase for short texts

3. **Spark Configuration**:
   - Increase parallelism for more concurrent processing
   - Allocate sufficient memory to avoid OOM errors

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.