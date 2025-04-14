#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility module for the distributed translation system.
Provides common utility functions used across the system.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import re
import hashlib
import functools
import traceback
import uuid

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, BooleanType
import pandas as pd


def set_up_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file (if None, logs to console only)
        
    Returns:
        Root logger configured with the specified settings
    """
    # Map string levels to logging constants
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configure file handler if log file is specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set logging level
    root_logger.setLevel(log_level)
    
    # Return the configured logger
    return root_logger


def create_spark_session(app_name: str = "distributed_translation",
                         config: Optional[Dict[str, Any]] = None) -> SparkSession:
    """
    Create and configure a Spark session.
    
    Args:
        app_name: Name of the Spark application
        config: Dictionary of configuration options for Spark
        
    Returns:
        Configured SparkSession
    """
    # Default configuration
    default_config = {
        "executor_memory": "4g",
        "driver_memory": "4g",
        "executor_cores": 2,
        "default_parallelism": 4
    }
    
    # Merge with provided configuration
    spark_config = {**default_config, **(config or {})}
    
    # Build SparkSession
    builder = SparkSession.builder.appName(app_name)
    
    # Set configuration options
    builder = builder.config("spark.executor.memory", spark_config.get("executor_memory"))
    builder = builder.config("spark.driver.memory", spark_config.get("driver_memory"))
    builder = builder.config("spark.executor.cores", spark_config.get("executor_cores"))
    builder = builder.config("spark.default.parallelism", spark_config.get("default_parallelism"))
    
    # Set common Spark configuration for better performance
    builder = builder.config("spark.sql.adaptive.enabled", "true")
    builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    builder = builder.config("spark.sql.shuffle.partitions", str(spark_config.get("default_parallelism") * 2))
    builder = builder.config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
    builder = builder.config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
    
    # Create and return the session
    spark = builder.getOrCreate()
    
    # Set log level to ERROR to reduce Spark verbosity
    spark.sparkContext.setLogLevel("ERROR")
    
    # Log Spark configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Created Spark session with app_name: {app_name}")
    logger.info(f"Spark configuration: {spark_config}")
    
    return spark


def timer(func: Callable) -> Callable:
    """
    Decorator to measure execution time of functions.
    
    Args:
        func: Function to be timed
        
    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.2f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(f"Function '{func.__name__}' failed after {elapsed_time:.2f} seconds: {str(e)}")
            raise
            
    return wrapper


def retry(max_attempts: int = 3, backoff_factor: float = 2.0, 
          exceptions: tuple = (Exception,), logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Factor to multiply delay time by on each retry
        exceptions: Tuple of exceptions to catch and retry
        logger: Logger to use for logging retries
        
    Returns:
        Wrapped function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            local_logger = logger or logging.getLogger(func.__module__)
            attempt = 0
            delay = 1
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        local_logger.error(
                            f"Function '{func.__name__}' failed after {max_attempts} attempts: {str(e)}"
                        )
                        raise
                    
                    # Calculate backoff delay
                    wait_time = delay * (backoff_factor ** (attempt - 1))
                    local_logger.warning(
                        f"Retry {attempt}/{max_attempts} for function '{func.__name__}' "
                        f"after {wait_time:.2f}s due to: {str(e)}"
                    )
                    
                    # Sleep with the calculated delay
                    time.sleep(wait_time)
            
            # This should never be reached
            return None
            
        return wrapper
    return decorator


def detect_language(text: str) -> str:
    """
    Detect the language of a given text.
    This is a simple implementation that relies on character frequency patterns.
    For production use, consider using a more robust language detection library.
    
    Args:
        text: Text to analyze
        
    Returns:
        ISO 639-1 language code (2-letter) or 'unknown'
    """
    if not text or len(text.strip()) < 10:
        return 'unknown'
    
    # Simplified language detection based on character frequency
    # Real implementation would use a proper language detection library
    
    # Normalize text
    text = text.lower()
    
    # Define character sets for different language groups
    lang_chars = {
        'en': set('abcdefghijklmnopqrstuvwxyz'),
        'es': set('abcdefghijklmnñopqrstuvwxyzáéíóúü'),
        'fr': set('abcdefghijklmnopqrstuvwxyzàâæçéèêëîïôœùûüÿ'),
        'de': set('abcdefghijklmnopqrstuvwxyzäöüß'),
        'it': set('abcdefghijklmnopqrstuvwxyzàèéìíîòóùú'),
        'pt': set('abcdefghijklmnopqrstuvwxyzáàâãçéêíóôõú'),
        'ru': set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя'),
        'zh': set(),  # Chinese would need a different approach
        'ja': set(),  # Japanese would need a different approach
        'ar': set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
    }
    
    # Common word patterns in different languages
    lang_patterns = {
        'en': [r'\bthe\b', r'\band\b', r'\bof\b', r'\bto\b', r'\ba\b', r'\bin\b', r'\bis\b'],
        'es': [r'\bel\b', r'\bla\b', r'\bde\b', r'\ben\b', r'\by\b', r'\bque\b', r'\bun\b'],
        'fr': [r'\ble\b', r'\bla\b', r'\bde\b', r'\bet\b', r'\ben\b', r'\bun\b', r'\best\b'],
        'de': [r'\bder\b', r'\bdie\b', r'\bdas\b', r'\bin\b', r'\bund\b', r'\bist\b', r'\bzu\b'],
        'it': [r'\bil\b', r'\bla\b', r'\bdi\b', r'\be\b', r'\bin\b', r'\bche\b', r'\bun\b'],
        'pt': [r'\bo\b', r'\ba\b', r'\bde\b', r'\be\b', r'\bem\b', r'\bque\b', r'\bum\b'],
        'ru': [r'\bи\b', r'\bв\b', r'\bна\b', r'\bс\b', r'\bне\b', r'\bчто\b', r'\bэто\b'],
        'zh': [],  # Would need a different approach
        'ja': [],  # Would need a different approach
        'ar': [r'\bفي\b', r'\bمن\b', r'\bإلى\b', r'\bعلى\b', r'\bأن\b', r'\bهذا\b', r'\bمع\b']
    }
    
    # Count character matches for each language
    char_scores = {}
    for lang, charset in lang_chars.items():
        if not charset:  # Skip languages with empty character sets
            continue
        
        # Count matching characters
        matches = sum(1 for char in text if char in charset)
        total_chars = sum(1 for char in text if char.isalpha())
        
        # Calculate score as a percentage of matching characters
        if total_chars > 0:
            char_scores[lang] = matches / total_chars
    
    # Check word patterns
    pattern_scores = {}
    for lang, patterns in lang_patterns.items():
        if not patterns:  # Skip languages with no patterns
            continue
        
        # Count matching patterns
        matches = sum(1 for pattern in patterns if re.search(pattern, text))
        pattern_scores[lang] = matches / len(patterns) if patterns else 0
    
    # Combine scores
    combined_scores = {}
    for lang in set(char_scores.keys()) | set(pattern_scores.keys()):
        char_weight = 0.7
        pattern_weight = 0.3
        
        char_score = char_scores.get(lang, 0)
        pattern_score = pattern_scores.get(lang, 0)
        
        combined_scores[lang] = (char_score * char_weight) + (pattern_score * pattern_weight)
    
    # Return the language with the highest score, or 'unknown' if no good match
    if combined_scores:
        best_lang = max(combined_scores.items(), key=lambda x: x[1])
        return best_lang[0] if best_lang[1] > 0.5 else 'unknown'
    
    return 'unknown'


def create_udf_detect_language() -> F.udf:
    """
    Create a Spark UDF for language detection.
    
    Returns:
        Spark UDF for language detection
    """
    return F.udf(detect_language, StringType())


def batch_dataframe(df: DataFrame, batch_size: int) -> List[DataFrame]:
    """
    Split a DataFrame into batches for easier processing.
    
    Args:
        df: DataFrame to split
        batch_size: Number of rows per batch
        
    Returns:
        List of DataFrame batches
    """
    # Add row number column
    df_with_index = df.withColumn("_row_num", F.monotonically_increasing_id())
    
    # Get total count
    total_rows = df.count()
    
    # Calculate number of batches
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    # Create and return batches
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size - 1
        
        batch = df_with_index.filter(
            (F.col("_row_num") >= start_idx) & 
            (F.col("_row_num") <= end_idx)
        ).drop("_row_num")
        
        batches.append(batch)
    
    return batches


def text_to_batches(texts: List[str], batch_size: int) -> List[List[str]]:
    """
    Split a list of texts into batches for efficient processing.
    
    Args:
        texts: List of text strings to split
        batch_size: Maximum number of texts per batch
        
    Returns:
        List of text batches
    """
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]


def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and normalizing line endings.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text


def truncate_text(text: str, max_length: int = 500, add_ellipsis: bool = True) -> str:
    """
    Truncate text to a maximum length, optionally adding an ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length in characters
        add_ellipsis: Whether to add an ellipsis (...) for truncated text
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    # Truncate to max_length
    truncated = text[:max_length]
    
    # Add ellipsis if requested
    if add_ellipsis:
        truncated += "..."
    
    return truncated


def format_time_elapsed(seconds: float) -> str:
    """
    Format elapsed time in a human-readable format.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string (e.g., "2h 30m 45s")
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in a human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "2.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024 or unit == 'TB':
            if unit == 'B':
                return f"{size_bytes} {unit}"
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024


def is_valid_language_code(code: str) -> bool:
    """
    Check if a string is a valid ISO 639-1 language code.
    
    Args:
        code: Language code to check
        
    Returns:
        Boolean indicating if code is valid
    """
    # Common ISO 639-1 language codes
    valid_codes = {
        'aa', 'ab', 'ae', 'af', 'ak', 'am', 'an', 'ar', 'as', 'av', 'ay', 'az',
        'ba', 'be', 'bg', 'bh', 'bi', 'bm', 'bn', 'bo', 'br', 'bs',
        'ca', 'ce', 'ch', 'co', 'cr', 'cs', 'cu', 'cv', 'cy',
        'da', 'de', 'dv', 'dz',
        'ee', 'el', 'en', 'eo', 'es', 'et', 'eu',
        'fa', 'ff', 'fi', 'fj', 'fo', 'fr', 'fy',
        'ga', 'gd', 'gl', 'gn', 'gu', 'gv',
        'ha', 'he', 'hi', 'ho', 'hr', 'ht', 'hu', 'hy', 'hz',
        'ia', 'id', 'ie', 'ig', 'ii', 'ik', 'io', 'is', 'it', 'iu',
        'ja', 'jv',
        'ka', 'kg', 'ki', 'kj', 'kk', 'kl', 'km', 'kn', 'ko', 'kr', 'ks', 'ku', 'kv', 'kw', 'ky',
        'la', 'lb', 'lg', 'li', 'ln', 'lo', 'lt', 'lu', 'lv',
        'mg', 'mh', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my',
        'na', 'nb', 'nd', 'ne', 'ng', 'nl', 'nn', 'no', 'nr', 'nv', 'ny',
        'oc', 'oj', 'om', 'or', 'os',
        'pa', 'pi', 'pl', 'ps', 'pt',
        'qu',
        'rm', 'rn', 'ro', 'ru', 'rw',
        'sa', 'sc', 'sd', 'se', 'sg', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'ss', 'st', 'su', 'sv', 'sw',
        'ta', 'te', 'tg', 'th', 'ti', 'tk', 'tl', 'tn', 'to', 'tr', 'ts', 'tt', 'tw', 'ty',
        'ug', 'uk', 'ur', 'uz',
        've', 'vi', 'vo',
        'wa', 'wo',
        'xh',
        'yi', 'yo',
        'za', 'zh', 'zu'
    }
    
    return code.lower() in valid_codes


def get_normalized_language_code(code: str) -> str:
    """
    Normalize a language code or name to a standard ISO 639-1 code.
    
    Args:
        code: Language code or name to normalize
        
    Returns:
        Normalized ISO 639-1 language code, or the original if not recognized
    """
    # Map of common language names and variations to ISO 639-1 codes
    language_map = {
        # English variations
        'english': 'en',
        'eng': 'en',
        'en-us': 'en',
        'en-gb': 'en',
        'en-uk': 'en',
        
        # Spanish variations
        'spanish': 'es',
        'español': 'es',
        'espanol': 'es',
        'spa': 'es',
        'es-es': 'es',
        'es-mx': 'es',
        'es-ar': 'es',
        
        # French variations
        'french': 'fr',
        'français': 'fr',
        'francais': 'fr',
        'fra': 'fr',
        'fre': 'fr',
        'fr-fr': 'fr',
        'fr-ca': 'fr',
        
        # German variations
        'german': 'de',
        'deutsch': 'de',
        'ger': 'de',
        'deu': 'de',
        'de-de': 'de',
        'de-at': 'de',
        'de-ch': 'de',
        
        # Chinese variations
        'chinese': 'zh',
        'mandarin': 'zh',
        'cantonese': 'zh',
        'chi': 'zh',
        'zho': 'zh',
        'zh-cn': 'zh',
        'zh-tw': 'zh',
        'zh-hk': 'zh',
        
        # Japanese variations
        'japanese': 'ja',
        'jpn': 'ja',
        'ja-jp': 'ja',
        
        # Russian variations
        'russian': 'ru',
        'русский': 'ru',
        'rus': 'ru',
        'ru-ru': 'ru',
        
        # Arabic variations
        'arabic': 'ar',
        'العربية': 'ar',
        'ara': 'ar',
        'ar-sa': 'ar',
        
        # Portuguese variations
        'portuguese': 'pt',
        'português': 'pt',
        'portugues': 'pt',
        'por': 'pt',
        'pt-pt': 'pt',
        'pt-br': 'pt',
        
        # Italian variations
        'italian': 'it',
        'italiano': 'it',
        'ita': 'it',
        'it-it': 'it',
        
        # Hindi variations
        'hindi': 'hi',
        'हिन्दी': 'hi',
        'hin': 'hi',
        'hi-in': 'hi',
        
        # Korean variations
        'korean': 'ko',
        '한국어': 'ko',
        'kor': 'ko',
        'ko-kr': 'ko',
        
        # Dutch variations
        'dutch': 'nl',
        'nederlands': 'nl',
        'nld': 'nl',
        'dut': 'nl',
        'nl-nl': 'nl',
        'nl-be': 'nl',
        
        # Polish variations
        'polish': 'pl',
        'polski': 'pl',
        'pol': 'pl',
        'pl-pl': 'pl',
        
        # Turkish variations
        'turkish': 'tr',
        'türkçe': 'tr',
        'turkce': 'tr',
        'tur': 'tr',
        'tr-tr': 'tr',
        
        # Swedish variations
        'swedish': 'sv',
        'svenska': 'sv',
        'swe': 'sv',
        'sv-se': 'sv',
        
        # Greek variations
        'greek': 'el',
        'ελληνικά': 'el',
        'ell': 'el',
        'gre': 'el',
        'el-gr': 'el',
        
        # Hebrew variations
        'hebrew': 'he',
        'עברית': 'he',
        'heb': 'he',
        'he-il': 'he',
        
        # Vietnamese variations
        'vietnamese': 'vi',
        'tiếng việt': 'vi',
        'tieng viet': 'vi',
        'vie': 'vi',
        'vi-vn': 'vi',
        
        # Indonesian variations
        'indonesian': 'id',
        'bahasa indonesia': 'id',
        'ind': 'id',
        'id-id': 'id',
        
        # Unknown language fallback
        'unknown': 'unknown',
        'unk': 'unknown',
        'undefined': 'unknown',
        'other': 'unknown'
    }
    
    # If already a valid 2-letter code, return as is
    if len(code) == 2 and is_valid_language_code(code):
        return code.lower()
    
    # Normalize the input
    normalized = code.lower().strip()
    
    # Look up in the map
    return language_map.get(normalized, code)


def get_default_source_language() -> str:
    """
    Get the default source language to use when none is specified.
    
    Returns:
        Default source language code ('auto' for automatic detection)
    """
    return 'auto'


def generate_unique_id() -> str:
    """
    Generate a unique identifier for tracking processing.
    
    Returns:
        Unique identifier string
    """
    return str(uuid.uuid4())


def format_stats_report(stats: Dict[str, Any]) -> str:
    """
    Format a statistics report in a human-readable format.
    
    Args:
        stats: Dictionary containing statistics
        
    Returns:
        Formatted statistics report
    """
    report = [
        "===== Translation Process Statistics =====",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    
    # Process timing stats
    if 'start_time' in stats and 'end_time' in stats:
        start_time = stats.get('start_time')
        end_time = stats.get('end_time')
        if start_time and end_time:
            elapsed_seconds = end_time - start_time
            report.append(f"\nExecution Time: {format_time_elapsed(elapsed_seconds)}")
    
    # Process row counts
    report.append("\n--- Row Counts ---")
    if 'total_rows' in stats:
        report.append(f"Total Rows: {stats['total_rows']:,}")
    if 'translated_rows' in stats:
        report.append(f"Translated Rows: {stats['translated_rows']:,}")
    
    # Process API and cache stats
    report.append("\n--- API & Cache Statistics ---")
    if 'api_calls' in stats:
        report.append(f"Total API Calls: {stats['api_calls']:,}")
    if 'cached_hits' in stats:
        report.append(f"Cache Hits: {stats['cached_hits']:,}")
    
    # Calculate cache hit rate
    if 'cached_hits' in stats and 'api_calls' in stats:
        total_requests = stats['cached_hits'] + stats['api_calls']
        if total_requests > 0:
            hit_rate = stats['cached_hits'] / total_requests * 100
            report.append(f"Cache Hit Rate: {hit_rate:.2f}%")
    
    # Process error stats
    if 'errors' in stats:
        report.append(f"\nErrors: {stats['errors']:,}")
    
    # Process cache details if available
    if 'cache' in stats:
        cache_stats = stats['cache']
        report.append("\n--- Cache Details ---")
        if 'total_entries' in cache_stats:
            report.append(f"Total Cache Entries: {cache_stats['total_entries']:,}")
        if 'size_bytes' in cache_stats:
            report.append(f"Cache Size: {format_file_size(cache_stats['size_bytes'])}")
    
    # Process language stats if available
    if 'languages' in stats:
        lang_stats = stats['languages']
        report.append("\n--- Language Statistics ---")
        if 'source' in lang_stats:
            source_langs = lang_stats['source']
            if source_langs:
                report.append(f"Source Languages: {', '.join(source_langs)}")
        if 'target' in lang_stats:
            report.append(f"Target Language: {lang_stats.get('target', 'unknown')}")
    
    # Process performance metrics if available
    if 'performance' in stats:
        perf_stats = stats['performance']
        report.append("\n--- Performance Metrics ---")
        if 'avg_translation_time' in perf_stats:
            report.append(f"Average Translation Time: {perf_stats['avg_translation_time']:.2f} seconds")
        if 'texts_per_second' in perf_stats:
            report.append(f"Processing Rate: {perf_stats['texts_per_second']:.2f} texts/second")
    
    report.append("\n=========================================")
    
    return "\n".join(report)


def save_stats_to_file(stats: Dict[str, Any], filepath: str) -> None:
    """
    Save statistics to a JSON file.
    
    Args:
        stats: Dictionary containing statistics
        filepath: Path to the output file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Add timestamp if not present
    if 'timestamp' not in stats:
        stats['timestamp'] = datetime.now().isoformat()
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)


def load_stats_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load statistics from a JSON file.
    
    Args:
        filepath: Path to the input file
        
    Returns:
        Dictionary containing statistics
    """
    if not os.path.exists(filepath):
        return {}
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading stats from {filepath}: {str(e)}")
        return {}


def merge_stats(stats1: Dict[str, Any], stats2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two statistics dictionaries, summing numeric values and merging lists and dicts.
    
    Args:
        stats1: First statistics dictionary
        stats2: Second statistics dictionary
        
    Returns:
        Merged statistics dictionary
    """
    if not stats1:
        return stats2.copy() if stats2 else {}
    if not stats2:
        return stats1.copy()
        
    result = stats1.copy()
    
    for key, value in stats2.items():
        if key not in result:
            result[key] = value
        elif isinstance(value, (int, float)) and isinstance(result[key], (int, float)):
            # Sum numeric values
            result[key] += value
        elif isinstance(value, list) and isinstance(result[key], list):
            # Merge lists (unique values)
            result[key] = list(set(result[key] + value))
        elif isinstance(value, dict) and isinstance(result[key], dict):
            # Recursively merge dictionaries
            result[key] = merge_stats(result[key], value)
    
    return result


def calculate_partition_sizes(total_size: int, num_partitions: int) -> List[int]:
    """
    Calculate balanced partition sizes for processing.
    
    Args:
        total_size: Total number of items to process
        num_partitions: Number of partitions to create
        
    Returns:
        List of partition sizes
    """
    base_size = total_size // num_partitions
    remainder = total_size % num_partitions
    
    # Distribute the remainder among the first 'remainder' partitions
    sizes = [base_size + 1 if i < remainder else base_size 
             for i in range(num_partitions)]
    
    return sizes


def estimate_completion_time(processed: int, total: int, elapsed_seconds: float) -> float:
    """
    Estimate remaining time to complete processing.
    
    Args:
        processed: Number of items processed so far
        total: Total number of items to process
        elapsed_seconds: Time elapsed so far in seconds
        
    Returns:
        Estimated remaining time in seconds, or -1 if cannot be estimated
    """
    if processed <= 0 or elapsed_seconds <= 0:
        return -1
    
    items_per_second = processed / elapsed_seconds
    remaining_items = total - processed
    
    if items_per_second > 0:
        return remaining_items / items_per_second
    
    return -1


def log_progress_bar(current: int, total: int, prefix: str = '', suffix: str = '', 
                    length: int = 50, fill: str = '█', logger: Optional[logging.Logger] = None) -> None:
    """
    Print a progress bar to the logger.
    
    Args:
        current: Current progress value
        total: Total value
        prefix: String before the progress bar
        suffix: String after the progress bar
        length: Length of the progress bar in characters
        fill: Character to use for the progress bar
        logger: Logger to use, or None to use print()
    """
    percent = min(100.0, (current / total) * 100) if total > 0 else 0
    filled_length = int(length * current // total) if total > 0 else 0
    bar = fill * filled_length + '-' * (length - filled_length)
    progress_str = f'\r{prefix} |{bar}| {percent:.1f}% {suffix}'
    
    if logger:
        logger.info(progress_str)
    else:
        print(progress_str, end='', file=sys.stdout)
        if current >= total:
            print(file=sys.stdout)
        sys.stdout.flush()


def create_udf_truncate_text(max_length: int = 500) -> F.udf:
    """
    Create a Spark UDF for truncating text.
    
    Args:
        max_length: Maximum length for the truncated text
        
    Returns:
        Spark UDF for text truncation
    """
    return F.udf(lambda text: truncate_text(text, max_length), StringType())


def create_udf_clean_text() -> F.udf:
    """
    Create a Spark UDF for cleaning text.
    
    Returns:
        Spark UDF for text cleaning
    """
    return F.udf(clean_text, StringType())


def create_udf_is_valid_language() -> F.udf:
    """
    Create a Spark UDF for validating language codes.
    
    Returns:
        Spark UDF for language code validation
    """
    return F.udf(is_valid_language_code, BooleanType())


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_bytes': memory_info.rss,
        'rss_human': format_file_size(memory_info.rss),
        'vms_bytes': memory_info.vms,
        'vms_human': format_file_size(memory_info.vms),
        'percent': process.memory_percent(),
        'available_system_memory': format_file_size(psutil.virtual_memory().available)
    }


def setup_temp_directory(base_dir: str = './temp') -> str:
    """
    Set up a temporary directory for processing artifacts.
    
    Args:
        base_dir: Base directory for temporary files
        
    Returns:
        Path to the created temporary directory
    """
    import tempfile
    
    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a unique subdirectory
    temp_dir = tempfile.mkdtemp(prefix='translation_', dir=base_dir)
    
    return temp_dir


def cleanup_temp_directory(temp_dir: str) -> None:
    """
    Clean up a temporary directory.
    
    Args:
        temp_dir: Path to the temporary directory to clean up
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to clean up temporary directory {temp_dir}: {str(e)}")


def safe_file_write(data: str, filepath: str) -> bool:
    """
    Safely write data to a file using a temporary file to prevent corruption.
    
    Args:
        data: Data to write
        filepath: Target file path
        
    Returns:
        Boolean indicating success
    """
    dir_path = os.path.dirname(os.path.abspath(filepath))
    
    # Create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    
    # Use a temporary file for atomic write
    temp_file = f"{filepath}.tmp.{os.getpid()}"
    
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(data)
        
        # On Windows, we need to remove the destination file first
        if os.name == 'nt' and os.path.exists(filepath):
            os.remove(filepath)
            
        # Rename the temporary file to the target filename (atomic on most systems)
        os.rename(temp_file, filepath)
        return True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error writing to file {filepath}: {str(e)}")
        
        # Clean up the temporary file
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        
        return False


def join_path(*paths: str) -> str:
    """
    Join path components in a platform-independent way.
    
    Args:
        *paths: Path components to join
        
    Returns:
        Joined path
    """
    return os.path.join(*paths)


def ensure_directory(directory: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Boolean indicating success
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error creating directory {directory}: {str(e)}")
        return False


def is_empty_string(text: Optional[str]) -> bool:
    """
    Check if a string is None, empty, or contains only whitespace.
    
    Args:
        text: String to check
        
    Returns:
        Boolean indicating if string is effectively empty
    """
    return text is None or text.strip() == ''


def get_exception_details() -> str:
    """
    Get detailed information about the current exception.
    Useful for logging detailed error information.
    
    Returns:
        Formatted exception details
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if not exc_type:
        return "No exception information available"
    
    trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
    return "".join(trace)