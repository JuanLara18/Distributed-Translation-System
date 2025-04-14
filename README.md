# Distributed Translation System

A scalable system for translating text columns in large datasets using OpenAI's language models with PySpark.

## What it does

This system helps you translate text columns in data files (Stata, CSV, Parquet, JSON) with these key features:

- **Distributed processing** with PySpark for handling large datasets
- **Smart caching** to avoid redundant API calls and reduce costs
- **Checkpointing** to resume interrupted processes
- **Stata file support** with preserved metadata
- **Multiple language support** with automatic detection

## Getting started

### Prerequisites

- Python 3.7+
- Java 8+ (for PySpark)
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/distributed-translation.git
cd distributed-translation

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Basic usage

1. Create a simple configuration file `config.yaml`:

```yaml
input_file: "data.dta"  # Your input file (supports .dta, .csv, .parquet, .json)
output_file: "translated_data.dta"
columns_to_translate: ["text_column1", "text_column2"]
target_language: "english"

openai:
  model: "gpt-3.5-turbo"
  api_key_env: "OPENAI_API_KEY"  # Uses key from .env file
```

2. Run the translation process:

```bash
python main.py --config config.yaml
```

## Configuration options

### Essential settings

| Setting | Description | Example |
|---------|-------------|---------|
| `input_file` | Path to your input data file | `"data/input.dta"` |
| `output_file` | Where to save the translated data | `"data/output.dta"` |
| `columns_to_translate` | List of columns to translate | `["text_col", "description_col"]` |
| `target_language` | Language to translate into | `"english"` |
| `source_language_column` | Column containing source languages (optional) | `"language_col"` |

### OpenAI settings

```yaml
openai:
  model: "gpt-3.5-turbo"  # or "gpt-4" for higher quality
  temperature: 0.1  # Lower for more consistent translations
  max_tokens: 1500
  api_key_env: "OPENAI_API_KEY"
```

### Performance tuning

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

### Caching options

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

### Checkpoint settings

```yaml
checkpoint:
  enabled: true
  interval: 1  # Save every N partitions
  directory: "./checkpoints"
  max_checkpoints: 5
```

## Command-line options

```bash
# Basic usage
python main.py --config config.yaml

# Override configuration
python main.py --config config.yaml --input_file new_input.dta --target_language spanish

# Resume from checkpoint
python main.py --config config.yaml --resume

# Force restart (ignore checkpoints)
python main.py --config config.yaml --force_restart

# Enable verbose logging
python main.py --config config.yaml --verbose
```

## How translations are processed

1. The system reads your input file and splits it into partitions
2. For each text column to translate:
   - First checks if the translation is already in the cache
   - If not cached, sends text to OpenAI's API
   - Saves results to cache to avoid future redundant API calls
3. Translations are added as new columns with `_[target_language]` suffix
4. The processed data is written to your output file

## Troubleshooting

### Common errors

- **API Key issues**: Make sure your OpenAI API key is set in your `.env` file
- **Memory errors**: Reduce `batch_size` or increase `spark.executor_memory`
- **Missing translations**: Verify source language detection or specify a source language column
- **Corrupted checkpoints**: Use `--force_restart` to start fresh

### Enabling debug mode

```yaml
logging:
  level: "DEBUG"
  log_file: "debug.log"
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.