# Distributed Translation System

A robust, scalable system for batch translating text columns in data files using OpenAI language models with PySpark-based distributed processing.

## Overview

The Distributed Translation System provides an efficient, fault-tolerant solution for translating large datasets by leveraging distributed computing and advanced caching mechanisms. It's specifically designed for researchers and data scientists working with multilingual datasets (particularly Stata files) who need reliable, high-quality translations across multiple text columns.

## Key Features

- **Distributed Processing**: Leverages Apache Spark for parallel translation of large datasets, significantly reducing processing time
- **Intelligent Caching**: Prevents redundant API calls with SQLite/PostgreSQL backends, reducing costs and improving efficiency
- **Checkpointing & Fault Tolerance**: Automatically saves progress and allows resuming interrupted processes
- **Language Detection**: Optional automatic source language detection when not explicitly specified
- **Batch Optimization**: Efficiently groups translation requests to minimize API calls
- **Stata File Handling**: Preserves all metadata, value labels, and variable attributes when processing Stata files
- **Configurable & Flexible**: Extensive YAML-based configuration with command-line overrides

## Architecture

The system follows a modular architecture with these key components:

- **TranslationOrchestrator**: Coordinates the entire workflow
- **ConfigManager**: Handles configuration loading and validation
- **DataReader/DataWriter**: Manages data I/O operations with special handling for Stata files
- **TranslationManager**: Handles the core translation logic using OpenAI models
- **CacheManager**: Provides efficient translation caching with multiple backend options
- **CheckpointManager**: Ensures fault tolerance through granular checkpointing

## Module Structure

- `main.py`: Entry point and orchestration
- `config.py`: Configuration management with defaults and validation
- `translator.py`: Translation service interface and OpenAI implementation
- `file_manager.py`: Handles reading and writing various file formats
- `cache.py`: Translation cache implementation with SQLite, PostgreSQL, and in-memory options
- `checkpoint.py`: Manages process checkpointing for fault tolerance
- `utilities.py`: Utility functions and helpers for language detection, etc.

## Prerequisites

- Python 3.7 or higher
- Java 8 or higher (required for PySpark)
- Sufficient disk space for caching and checkpoints
- OpenAI API key

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/distributed-translation.git
cd distributed-translation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file for your API key
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Configuration

### Basic Configuration

Create a configuration file `config.yaml`:

```yaml
input_file: "path/to/input.dta"
output_file: "path/to/output.dta"
columns_to_translate: ["column1", "column2"]
source_language_column: "language_col"  # Optional: column specifying source language
target_language: "english"
batch_size: 10

openai:
  model: "gpt-3.5-turbo"
  temperature: 0.1
  max_tokens: 1500
  api_key_env: "OPENAI_API_KEY"  # This will use the key from .env file

spark:
  executor_memory: "4g"
  driver_memory: "4g"
  executor_cores: 2
  default_parallelism: 4
```

### Advanced Configuration Options

```yaml
# Cache configuration
cache:
  type: "sqlite"  # or "postgres" or "memory"
  location: "./cache/translations.db"  # for sqlite
  # connection_string: "postgresql://user:password@localhost/translations"  # for postgres
  ttl: 2592000  # 30 days in seconds

# Checkpoint configuration
checkpoint:
  enabled: true
  interval: 1  # Every partition
  directory: "./checkpoints"
  max_checkpoints: 5

# Logging configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: "translation_process.log"

# Retry configuration for API calls
retry:
  max_attempts: 3
  backoff_factor: 2
  
# Intermediate results
write_intermediate_results: false
intermediate_directory: "./intermediate"
```

## API Key Management

This project uses environment variables stored in a `.env` file for API key management:

1. Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```

2. The application will automatically load this file and use the API key for authentication.

3. Make sure to add `.env` to your `.gitignore` file to prevent accidentally committing your API key.

## Usage

### Basic Usage

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the translation process
python main.py --config config.yaml
```

### Command-line Options

```bash
# Get help
python main.py --help

# Override configuration settings
python main.py --config config.yaml --input_file new_input.dta --target_language spanish

# Resume from checkpoint
python main.py --config config.yaml --resume

# Specify custom checkpoint directory
python main.py --config config.yaml --checkpoint_dir ./custom_checkpoints
```

### Processing Different File Types

The system supports multiple file formats:

```bash
# Stata files (default)
python main.py --config config.yaml --input_file data.dta --output_file translated.dta

# CSV files
python main.py --config config.yaml --input_file data.csv --output_file translated.csv

# Parquet files
python main.py --config config.yaml --input_file data.parquet --output_file translated.parquet

# JSON files
python main.py --config config.yaml --input_file data.json --output_file translated.json
```

## Cache Management

The translation cache can be managed with utility scripts:

```bash
# Clear cache
python utils/cache_manager.py --clear

# Show cache statistics
python utils/cache_manager.py --stats

# Export cache to file
python utils/cache_manager.py --export translations.json

# Import cache from file
python utils/cache_manager.py --import translations.json
```

## Performance Tuning

### For Smaller Datasets (<100MB)
```yaml
spark:
  executor_memory: "2g"
  driver_memory: "2g"
  executor_cores: 1
  default_parallelism: 2
```

### For Larger Datasets (>1GB)
```yaml
spark:
  executor_memory: "8g"
  driver_memory: "6g"
  executor_cores: 4
  default_parallelism: 8
```

## Troubleshooting

### Common Issues

1. **OpenAI API Rate Limits**: If you encounter rate limits, try increasing the `retry.max_attempts` and `retry.backoff_factor` values.

2. **Memory Issues**: If you experience out-of-memory errors, reduce `batch_size` or increase `spark.executor_memory`.

3. **Missing Translations**: Ensure your source language detection is working correctly or explicitly specify the source language.

4. **Checkpoint Errors**: If checkpoint files are corrupted, clear the checkpoint directory and restart the process.

### Debugging

Enable detailed logging for troubleshooting:
```yaml
logging:
  level: "DEBUG"
  log_file: "debug.log"
```

## Implementation Details

### Caching System

The system implements three different cache backends:

1. **SQLite Cache**: Default option, stores translations in a local SQLite database with WAL journaling for improved performance.

2. **PostgreSQL Cache**: Enterprise option for shared caching across multiple processes or servers.

3. **In-Memory Cache**: Lightweight option for testing or small workloads, does not persist between runs.

### Language Detection

The system includes a basic language detection implementation that can identify common languages based on character frequencies and word patterns. For production use with difficult language pairs, consider using the source_language_column parameter to explicitly specify source languages.

### Translation Process

1. Data is read from the input file and partitioned for parallel processing
2. For each text column to translate:
   - The system checks if a cached translation exists
   - If not cached, text is sent to OpenAI's API for translation
   - Results are cached to avoid redundant API calls
3. Translations are added as new columns with "_[target_language]" suffix
4. Processed data is written to the output file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.