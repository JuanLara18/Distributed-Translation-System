#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translator module for the distributed translation system.
Handles translation of text using various translation services.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import logging
import concurrent.futures
from functools import partial
import re
import json

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

import re
from modules.utilities import detect_language, get_normalized_language_code

from modules.cache import CacheManager
from modules.utilities import (
    retry, detect_language, clean_text, 
    truncate_text, get_normalized_language_code
)

def translate_text_global(text: str, source_language: str, target_language: str, config_json: str) -> str:
    """
    Función global para traducir un texto sin capturar objetos no serializables.
    Se reconstruye el traductor usando una configuración mínima en formato JSON.
    """
    # Convertir el JSON a diccionario
    config = json.loads(config_json)
    # Crear una instancia del traductor localmente en el worker
    translator = OpenAITranslator(config)
    # Realizar la traducción
    return translator.translate(text, source_language, target_language)


class AbstractTranslator(ABC):
    """
    Abstract base class for translators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the translator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def translate(self, text: str, source_language: str, target_language: str) -> str:
        """
        Translate a single text string.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Translated text
        """
        pass
    
    @abstractmethod
    def batch_translate(self, texts: List[str], source_language: str, target_language: str) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            List of translated texts
        """
        pass


class OpenAITranslator(AbstractTranslator):
    """
    Translator implementation using OpenAI API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI translator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.api_key = os.environ.get(config.get('openai', {}).get('api_key_env', 'OPENAI_API_KEY'))
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        self.model = config.get('openai', {}).get('model', 'gpt-3.5-turbo')
        self.temperature = config.get('openai', {}).get('temperature', 0.1)
        self.max_tokens = config.get('openai', {}).get('max_tokens', 1500)
        self.retry_config = config.get('retry', {})
        
        # Initialize OpenAI client
        try:
            import openai
            openai.api_key = self.api_key
            self.client = openai
            self.logger.info(f"Initialized OpenAI client with model {self.model}")
        except ImportError:
            self.logger.error("Failed to import OpenAI module. Make sure it's installed.")
            raise
    
    @retry(max_attempts=3, backoff_factor=2)
    def translate(self, text: str, source_language: str, target_language: str) -> str:
        """
        Translate a text using OpenAI API.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
        
        # Normalize language codes
        source_language = get_normalized_language_code(source_language)
        target_language = get_normalized_language_code(target_language)
        
        # Handle auto-detection for source language
        if source_language == 'auto' or source_language == 'unknown':
            detected_language = detect_language(text)
            if detected_language != 'unknown':
                source_language = detected_language
                self.logger.debug(f"Auto-detected language as {source_language}")
            else:
                source_language = 'en'  # Default to English if detection fails
                self.logger.warning("Could not detect language, defaulting to English")
        
        system_prompt = self._get_system_prompt(source_language, target_language)
        
        try:
            start_time = time.time()
            
            # Call OpenAI API using recommended v1 API structure
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            translation = response.choices[0].message.content
            translation = self._validate_and_clean_translation(
                original_text=text, 
                translation=translation, 
                source_language=source_language, 
                target_language=target_language
            )
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Translation completed in {elapsed_time:.2f}s: {source_language} -> {target_language}, {len(text)} chars")
            
            return translation
            
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            # For transient errors, let the retry decorator handle it
            # For persistent errors, return original text as fallback
            raise
    
    def batch_translate(self, texts: List[str], source_languages: Union[List[str], str], target_language: str) -> List[str]:
        """
        Translate a batch of texts using parallel processing.
        
        Args:
            texts: List of texts to translate
            source_languages: Either a list of source languages (one per text) or a single language for all texts
            target_language: Target language code
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        # Filter out empty texts
        filtered_texts = [text for text in texts if text and text.strip()]
        if not filtered_texts:
            return [""] * len(texts)
        
        # Create a mapping of original indices to filtered indices
        original_to_filtered = {}
        filtered_index = 0
        for i, text in enumerate(texts):
            if text and text.strip():
                original_to_filtered[i] = filtered_index
                filtered_index += 1
        
        # Handle source_languages parameter - make it a list if it's a string
        if isinstance(source_languages, str):
            source_langs = [source_languages] * len(filtered_texts)
        else:
            # If it's already a list, make sure it has entries for each filtered text
            filtered_langs = []
            for i, text in enumerate(texts):
                if text and text.strip():
                    filtered_langs.append(source_languages[i])
            source_langs = filtered_langs
        
        # Normalize language codes
        source_langs = [get_normalized_language_code(lang) for lang in source_langs]
        target_language = get_normalized_language_code(target_language)
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(len(filtered_texts), os.cpu_count() * 3 or 2)  # Increased multiplier for better concurrency
        results = [""] * len(texts)
        
        # Create a list of translation tasks
        translation_tasks = []
        for i, (text, lang) in enumerate(zip(filtered_texts, source_langs)):
            translation_tasks.append((i, text, lang))
        
        # Counter for actual API calls made during this batch
        actual_api_calls = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all translation tasks
            future_to_task = {}
            for idx, text, lang in translation_tasks:
                # Use a wrapper function to track actual API calls
                def translate_and_count(text, lang, target):
                    nonlocal actual_api_calls
                    # Only increment if we actually make an API call (not from cache)
                    # This assumes the translate method handles caching internally
                    result = self.translate(text, lang, target)
                    # Increment counter safely with lock if this was an API call
                    # Note: This requires modifying the translate method to return
                    # a tuple (translation, from_api) where from_api is a boolean
                    # indicating if an API call was made
                    actual_api_calls += 1
                    return result
                    
                future = executor.submit(translate_and_count, text, lang, target_language)
                future_to_task[future] = (idx, text)
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                filtered_index, original_text = future_to_task[future]
                
                try:
                    translation = future.result()
                    
                    # Find all original indices that map to this filtered index
                    for orig_idx, filt_idx in original_to_filtered.items():
                        if filt_idx == filtered_index:
                            results[orig_idx] = translation
                            
                except Exception as e:
                    self.logger.error(f"Error in batch translation: {str(e)}")
                    # Find original index for error reporting
                    for orig_idx, filt_idx in original_to_filtered.items():
                        if filt_idx == filtered_index:
                            self.logger.error(f"Failed to translate text at index {orig_idx}")
                            # Use original text as fallback
                            results[orig_idx] = texts[orig_idx]
        
        # Update stats with actual API calls made
        self.logger.debug(f"Batch completed: {actual_api_calls} API calls for {len(filtered_texts)} texts")
        
        return results

    def _get_system_prompt(self, source_language: str, target_language: str) -> str:
        """
        Generate an aggressive system prompt that translates everything possible.
        """
        return f"""You are a professional translator. Translate ALL non-English text to English.

    CRITICAL RULES:
    1. Translate ANY foreign language words/phrases to English
    2. For job titles, use standard professional English terminology  
    3. Translate German, Spanish, French, etc. words even if mixed with English
    4. Keep English words unchanged, translate everything else
    5. Preserve numbers, codes, and proper names (like company names)
    6. Return ONLY the translation, no explanations

    EXAMPLES:
    - "Zeitmanagement 2" → "Time Management 2"
    - "Wareneingang, Verpackung & Expedition" → "Goods Receipt, Packaging & Dispatch" 
    - "Software Engineer" → "Software Engineer" (already English)
    - "Auftragsmgmt & Verkaufssteuerung" → "Order Management & Sales Control"
    - "Production Manager" → "Production Manager" (already English)

    Translate aggressively - when in doubt, translate it."""

    def _validate_and_clean_translation(self, original_text: str, translation: str, 
                                        source_language: str, target_language: str) -> str:
        """
        Simplified validation - accept almost all translations.
        """
        if not translation or not translation.strip():
            self.logger.warning("Empty translation received, returning original text")
            return original_text
        
        # Basic cleaning only
        translation = self._basic_cleaning(translation)
        
        # Only reject if it's clearly an error message
        if any(phrase in translation.lower() for phrase in [
            "i cannot translate", "i can't translate", "unable to translate",
            "translation not possible", "error:", "failed to translate"
        ]):
            self.logger.warning(f"API error detected in translation: {translation[:50]}")
            return original_text
        
        # Accept the translation
        return translation

    def _is_translation_failure(self, translation: str, original_text: str) -> bool:
        """Only detect obvious API failures."""
        if not translation.strip():
            return True
        
        # Only flag clear error messages as failures
        error_phrases = [
            "i cannot translate", "i can't translate", "unable to translate",
            "translation not possible", "error:", "failed to translate"
        ]
        
        return any(phrase in translation.lower() for phrase in error_phrases)

    def _validate_target_language(self, translation: str, target_language: str) -> bool:
        """Always return True - disable language detection validation."""
        return True

    def _validate_length(self, original: str, translation: str, min_ratio: float, max_ratio: float) -> bool:
        """Always return True - disable length validation.""" 
        return True

    def _basic_cleaning(self, text: str) -> str:
        """Remove common artifacts from OpenAI responses."""
        if not text:
            return ""
        
        # Remove common prefixes
        prefixes_to_remove = [
            r'^(Translation:\s*)',
            r'^(Translated text:\s*)',
            r'^(Result:\s*)',
            r'^(Output:\s*)',
            r'^(Here is the translation:\s*)',
            r'^(The translation is:\s*)',
        ]
        
        for prefix_pattern in prefixes_to_remove:
            text = re.sub(prefix_pattern, '', text, flags=re.IGNORECASE)
        
        # Remove quotes if the entire text is quoted
        if ((text.startswith('"') and text.endswith('"')) or 
            (text.startswith("'") and text.endswith("'"))):
            text = text[1:-1]
        
        # Remove markdown formatting that sometimes appears
        text = re.sub(r'^\*\*(.*)\*\*$', r'\1', text)  # Bold
        text = re.sub(r'^\*(.*)\*$', r'\1', text)      # Italic
        
        # Clean up extra whitespace but preserve intentional formatting
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Only strip leading/trailing whitespace, preserve internal spacing
            cleaned_line = re.sub(r'^\s+|\s+$', '', line)
            if cleaned_line or (line and line.isspace()):  # Preserve blank lines
                cleaned_lines.append(cleaned_line)
        
        text = '\n'.join(cleaned_lines)
        
        return text

    def _salvage_translation(self, translation: str, original_text: str) -> str:
        """Try to salvage a partially failed translation."""
        
        # If translation contains some useful content along with error messages,
        # try to extract the useful part
        lines = translation.split('\n')
        useful_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            # Skip lines that contain error indicators
            if any(indicator in line_lower for indicator in 
                ["i cannot", "i can't", "unable to", "error:", "i apologize", "i'm sorry"]):
                continue
            
            # Keep lines that seem to contain actual content
            if line.strip() and len(line.strip()) > 3:
                useful_lines.append(line)
        
        if useful_lines:
            salvaged = '\n'.join(useful_lines).strip()
            if len(salvaged) > 0:
                return salvaged
        
        # If salvaging fails, return original
        return original_text

    def _preserve_numbers(self, original: str, translation: str) -> str:
        """Ensure important numbers are preserved in translation."""
        
        # Find all numbers in original text
        original_numbers = re.findall(r'\b\d+(?:[.,]\d+)*\b', original)
        translation_numbers = re.findall(r'\b\d+(?:[.,]\d+)*\b', translation)
        
        # If we lost important numbers, try to restore them
        if len(original_numbers) > len(translation_numbers):
            missing_numbers = [num for num in original_numbers if num not in translation_numbers]
            
            # Log the missing numbers
            self.logger.debug(f"Numbers potentially lost in translation: {missing_numbers}")
            
            # For now, just log - more sophisticated restoration could be implemented
            # This would require more complex logic to determine where numbers should go
        
        return translation

    def _preserve_urls(self, original: str, translation: str) -> str:
        """Ensure URLs are preserved in translation."""
        
        # Find URLs in original
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
        original_urls = re.findall(url_pattern, original)
        translation_urls = re.findall(url_pattern, translation)
        
        # If URLs are missing from translation, try to add them back
        missing_urls = [url for url in original_urls if url not in translation]
        
        if missing_urls:
            self.logger.debug(f"URLs missing from translation: {missing_urls}")
            # Append missing URLs at the end as a simple restoration
            if translation.strip():
                translation = translation.strip() + ' ' + ' '.join(missing_urls)
            else:
                translation = ' '.join(missing_urls)
        
        return translation

    def _preserve_emails(self, original: str, translation: str) -> str:
        """Ensure email addresses are preserved in translation."""
        
        # Find emails in original
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        original_emails = re.findall(email_pattern, original)
        translation_emails = re.findall(email_pattern, translation)
        
        # If emails are missing from translation, try to add them back
        missing_emails = [email for email in original_emails if email not in translation]
        
        if missing_emails:
            self.logger.debug(f"Email addresses missing from translation: {missing_emails}")
            # Append missing emails at the end
            if translation.strip():
                translation = translation.strip() + ' ' + ' '.join(missing_emails)
            else:
                translation = ' '.join(missing_emails)
        
        return translation

    def _preserve_structure(self, original: str, translation: str) -> str:
        """Preserve basic text structure like line breaks."""
        
        # Count line breaks in original
        original_lines = len(original.split('\n'))
        translation_lines = len(translation.split('\n'))
        
        # If original had multiple lines but translation is single line,
        # and original has clear paragraph breaks, try to restore some structure
        if original_lines > 2 and translation_lines == 1:
            # Look for paragraph indicators in original
            if '\n\n' in original:
                # This is a complex problem - for now just log it
                self.logger.debug("Multi-paragraph structure may have been lost in translation")
        
        return translation

    def _has_critical_errors(self, translation: str) -> bool:
        """Check for critical errors that would make translation unusable."""
        
        critical_patterns = [
            r'<\|.*?\|>',  # OpenAI special tokens
            r'\[ERROR\]',   # Error markers
            r'\[FAILED\]',  # Failure markers
            r'(?i)api\s*error',  # API error messages
            r'(?i)rate\s*limit',  # Rate limit messages
            r'(?i)quota\s*exceeded',  # Quota messages
        ]
        
        for pattern in critical_patterns:
            if re.search(pattern, translation):
                return True
        
        return False


class TranslationManager:
    """
    Manages the translation process for dataframes.
    """
    
    def __init__(self, config: Dict[str, Any], cache_manager: CacheManager):
        """
        Initialize the translation manager.
        
        Args:
            config: Configuration dictionary
            cache_manager: Cache manager instance
        """
        self.config = config
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create translator instance
        self.translator = OpenAITranslator(config)
        
        self.columns_to_translate = config.get('columns_to_translate', [])
        self.source_language_column = config.get('source_language_column')
        self.target_language = config.get('target_language', 'english')
        self.batch_size = config.get('batch_size', 10)
        
        self.stats = {
            'translated_rows': 0,
            'cached_hits': 0,
            'api_calls': 0,
            'errors': 0
        }
        
    def process_dataframe(self, df: DataFrame) -> DataFrame:
        """
        Process a dataframe by translating all required columns efficiently
        using batch processing and caching.
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame with translations added
        """
        if not self.columns_to_translate:
            self.logger.warning("No columns specified for translation")
            return df
            
        # Create an output dataframe starting with the original data
        processed_df = df
        
        # Process each column
        for column in self.columns_to_translate:
            if column not in df.columns:
                self.logger.warning(f"Column '{column}' not found in dataframe, skipping")
                continue
            
            # Generate the output column name
            output_column = f"{column}_{self.target_language}"
            
            # Collect data for processing
            self.logger.info(f"Collecting data for column '{column}'")
            
            if self.source_language_column and self.source_language_column in df.columns:
                self.logger.info(f"Translating column '{column}' using language from '{self.source_language_column}'")
                
                # Convert to pandas for easier processing
                pandas_df = df.select(column, self.source_language_column).toPandas()
                
                # Remove rows with empty values
                pandas_df = pandas_df[pandas_df[column].notna() & (pandas_df[column] != "")]
                
                # Create a dictionary mapping texts to languages
                text_lang_pairs = {}
                for _, row in pandas_df.iterrows():
                    text = row[column]
                    lang = row[self.source_language_column] if row[self.source_language_column] else "auto"
                    if text and isinstance(text, str):
                        text_lang_pairs[text] = lang
                
                self.logger.info(f"Found {len(text_lang_pairs)} unique non-empty texts to translate in column '{column}'")
                
                # If no texts to translate, skip this column
                if not text_lang_pairs:
                    self.logger.warning(f"No texts to translate in column '{column}', skipping")
                    continue
                
                # Process in batches
                batch_size = self.batch_size
                texts = list(text_lang_pairs.keys())
                langs = [text_lang_pairs[text] for text in texts]
                
                # Initialize translations dictionary
                translations = {}
                
                # Process in batches
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_langs = langs[i:i+batch_size]
                    
                    self.logger.info(f"Processing batch {i//batch_size + 1} of {(len(texts) + batch_size - 1)//batch_size} for column '{column}'")
                    
                    # First check cache for all texts in batch
                    batch_cache_results = {}
                    for text, lang in zip(batch_texts, batch_langs):
                        cached = self.cache_manager.get(text, lang, self.target_language)
                        if cached:
                            batch_cache_results[text] = cached
                            self.stats['cached_hits'] += 1
                    
                    # Texts not found in cache
                    texts_to_translate = []
                    langs_to_translate = []
                    for text, lang in zip(batch_texts, batch_langs):
                        if text not in batch_cache_results:
                            texts_to_translate.append(text)
                            langs_to_translate.append(lang)
                    
                    # Only call API if we have texts to translate
                    try:
                        # Use batch_translate to parallel process all texts
                        api_results = self.translator.batch_translate(texts_to_translate, langs_to_translate, self.target_language)
                        
                        # Update stats once for the whole batch
                        self.stats['api_calls'] += len(texts_to_translate)
                        self.stats['translated_rows'] =  self.stats['api_calls'] + self.stats['cached_hits']
                        
                        # Add to cache and results
                        for idx, text in enumerate(texts_to_translate):
                            translation = api_results[idx]
                            batch_cache_results[text] = translation
                            self.cache_manager.set(text, langs_to_translate[idx], self.target_language, translation)
                            
                    except Exception as e:
                        self.logger.error(f"Batch translation error: {str(e)}")
                        # Use original texts as fallback for any failures
                        for text in texts_to_translate:
                            if text not in batch_cache_results:
                                batch_cache_results[text] = text
                                self.stats['errors'] += 1
                    
                    # Add batch results to all translations
                    translations.update(batch_cache_results)
                
                # Create UDF to map texts to translations
                def map_translation(text):
                    if not text or not isinstance(text, str) or text.strip() == "":
                        return ""
                    return translations.get(text, text)
                
                map_udf = F.udf(map_translation, StringType())
                
                # Apply translations to DataFrame
                processed_df = processed_df.withColumn(output_column, map_udf(F.col(column)))
                
                self.logger.info(f"Completed translation of column '{column}'")
            else:
                # No source language column, use auto-detection for all
                self.logger.info(f"Translating column '{column}' with auto language detection")
                
                # Convert to pandas for easier processing
                pandas_df = df.select(column).toPandas()
                
                # Remove rows with empty values
                pandas_df = pandas_df[pandas_df[column].notna() & (pandas_df[column] != "")]
                
                # Get unique texts
                unique_texts = pandas_df[column].unique().tolist()
                unique_texts = [text for text in unique_texts if text and isinstance(text, str)]
                
                self.logger.info(f"Found {len(unique_texts)} unique non-empty texts to translate in column '{column}'")
                
                # If no texts to translate, skip this column
                if not unique_texts:
                    self.logger.warning(f"No texts to translate in column '{column}', skipping")
                    continue
                
                # Process in batches
                batch_size = self.batch_size
                
                # Initialize translations dictionary
                translations = {}
                
                # Process in batches
                for i in range(0, len(unique_texts), batch_size):
                    batch_texts = unique_texts[i:i+batch_size]
                    
                    self.logger.info(f"Processing batch {i//batch_size + 1} of {(len(unique_texts) + batch_size - 1)//batch_size} for column '{column}'")
                    
                    # First check cache for all texts in batch
                    batch_cache_results = {}
                    for text in batch_texts:
                        cached = self.cache_manager.get(text, "auto", self.target_language)
                        if cached:
                            batch_cache_results[text] = cached
                            self.stats['cached_hits'] += 1
                    
                    # Texts not found in cache
                    texts_to_translate = [text for text in batch_texts if text not in batch_cache_results]
                    
                    # Only call API if we have texts to translate
                    try:
                        # Create a list of "auto" source languages, one for each text
                        auto_langs = ["auto"] * len(texts_to_translate)
                        
                        # Use batch_translate to parallel process all texts
                        api_results = self.translator.batch_translate(texts_to_translate, auto_langs, self.target_language)
                        
                        # Update stats once for the whole batch
                        self.stats['api_calls'] += len(texts_to_translate)
                        self.stats['translated_rows'] += len(texts_to_translate)
                        
                        # Add to cache and results
                        for idx, text in enumerate(texts_to_translate):
                            translation = api_results[idx]
                            batch_cache_results[text] = translation
                            self.cache_manager.set(text, "auto", self.target_language, translation)
                            
                    except Exception as e:
                        self.logger.error(f"Batch translation error: {str(e)}")
                        # Use original texts as fallback for any failures
                        for text in texts_to_translate:
                            if text not in batch_cache_results:
                                batch_cache_results[text] = text
                                self.stats['errors'] += 1
                    
                    # Add batch results to all translations
                    translations.update(batch_cache_results)
                
                # Create UDF to map texts to translations
                def map_translation(text):
                    if not text or not isinstance(text, str) or text.strip() == "":
                        return ""
                    return translations.get(text, text)
                
                map_udf = F.udf(map_translation, StringType())
                
                # Apply translations to DataFrame
                processed_df = processed_df.withColumn(output_column, map_udf(F.col(column)))
                
                self.logger.info(f"Completed translation of column '{column}'")
        
        return processed_df
    
    def _translate_with_language(self, text: str, source_language: str) -> str:
        """
        Translate text while considering source language.
        This function will be called by the UDF.
        
        Args:
            text: Text to translate
            source_language: Source language
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
            
        try:
            # Check cache first
            cached_translation = self.cache_manager.get(text, source_language, self.target_language)
            if cached_translation:
                self.stats['cached_hits'] += 1
                return cached_translation
                
            # Not in cache, perform translation
            translation = self.translator.translate(text, source_language, self.target_language)
            self.stats['api_calls'] += 1
            
            # Save to cache
            self.cache_manager.set(text, source_language, self.target_language, translation)
            
            self.stats['translated_rows'] += 1
            return translation
            
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            self.stats['errors'] += 1
            # Return original text as fallback on error
            return text
    
    def apply_translations(self, df: DataFrame) -> DataFrame:
        """
        Apply all cached translations to a dataframe.
        This is used for the final processing of the complete dataset.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with translations applied
        """
        if not self.columns_to_translate:
            return df
            
        # Create an output dataframe starting with the original data
        output_df = df
        
        # Serialize necessary configuration for cache access
        cache_config = self.config.get('cache', {})
        target_language = self.target_language
        
        # Convert to JSON for serialization
        config_json = json.dumps({
            'cache': cache_config,
            'target_language': target_language
        })
        
        # Define a UDF that creates its own cache connection
        def get_cached_translation(text: str, source_language: str) -> str:
            if not text or not text.strip():
                return ""
            
            # Import necessary modules within the function 
            # so they're available on worker nodes
            import json
            import sqlite3
            import os
            import time
            import hashlib
            
            # Parse configuration
            config = json.loads(config_json)
            target_lang = config.get('target_language', 'english')
            cache_config = config.get('cache', {})
            cache_location = cache_config.get('location', './cache/translations.db')
            ttl = cache_config.get('ttl', 2592000)  # Default 30 days
            
            # Create connection inside the function
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(cache_location)), exist_ok=True)
                
                # Connect to the database
                conn = sqlite3.connect(cache_location, check_same_thread=False, timeout=30.0)
                cursor = conn.cursor()
                
                # Generate a unique ID for the entry
                key = f"{source_language}:{target_lang}:{text}"
                entry_id = hashlib.md5(key.encode('utf-8')).hexdigest()
                
                # Check if translation exists in cache
                current_time = int(time.time())
                expiry_time = current_time - ttl
                
                cursor.execute(
                    '''
                    SELECT translation FROM translations 
                    WHERE id = ? AND timestamp > ?
                    ''',
                    (entry_id, expiry_time)
                )
                
                result = cursor.fetchone()
                
                # Close connection
                cursor.close()
                conn.close()
                
                if result:
                    return result[0]
                
                # Return original if not in cache
                return text
                
            except Exception:
                # Return original text on any error
                return text
        
        # Create the UDF
        get_cached_udf = F.udf(get_cached_translation, StringType())
        
        # Apply translations from cache
        for column in self.columns_to_translate:
            if column not in df.columns:
                continue
                
            output_column = f"{column}_{self.target_language}"
            
            if self.source_language_column and self.source_language_column in df.columns:
                output_df = output_df.withColumn(
                    output_column,
                    get_cached_udf(F.col(column), F.col(self.source_language_column))
                )
            else:
                output_df = output_df.withColumn(
                    output_column,
                    get_cached_udf(F.col(column), F.lit("auto"))
                )
        
        return output_df

    def batch_translate_without_stats(self, texts: List[str], source_languages: List[str], target_language: str) -> List[str]:
        """
        Translate a batch of texts without updating internal statistics.
        Used for pre-caching to avoid double-counting stats.
        
        Args:
            texts: List of texts to translate
            source_languages: List of source languages
            target_language: Target language code
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
            
        # Use the translator's batch_translate method directly
        return self.translator.batch_translate(texts, source_languages, target_language)

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics from the translation process.
        
        Returns:
            Dictionary of translation statistics
        """
        # Get cache statistics
        cache_stats = self.cache_manager.get_stats()
        
        # Merge with translation stats
        merged_stats = {
            **self.stats,
            'languages': {
                'source': cache_stats.get('languages', {}).get('source', []),
                'target': self.target_language
            },
            'cache': {
                'entries': cache_stats.get('valid_entries', 0),
                'size_bytes': cache_stats.get('size_bytes', 0),
                'hit_rate': cache_stats.get('hit_rate', 0) * 100  # Convert to percentage
            }
        }
        
        return merged_stats