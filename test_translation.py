#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify translation quality on a small sample.
Run this before the full translation process.
"""

import os
import sys
import argparse
import logging
import pandas as pd
from typing import Dict, Any
import random

from config import ConfigManager
from modules.file_manager import DataReader
from modules.translator import TranslationManager
from modules.cache import CacheManager
from modules.utilities import set_up_logging, create_spark_session


def create_test_sample(df, sample_percentage: float = 1.0, max_rows: int = 100):
    """Create a random sample for testing."""
    total_rows = df.count()
    sample_size = min(max_rows, int(total_rows * sample_percentage / 100))
    
    print(f"Total rows: {total_rows:,}")
    print(f"Sample size: {sample_size} ({sample_percentage}%)")
    
    # Get random sample
    sample_df = df.sample(fraction=sample_size/total_rows, seed=42)
    return sample_df


def test_translation(config_path: str, sample_percentage: float = 1.0, max_rows: int = 100):
    """Test translation on a small sample."""
    
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.get_config()
    
    # Set up logging
    set_up_logging('INFO')
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("TRANSLATION TEST")
    print("=" * 60)
    
    # Check API key
    api_key_env = config.get('openai', {}).get('api_key_env', 'OPENAI_API_KEY')
    if not os.environ.get(api_key_env):
        print(f"❌ ERROR: OpenAI API key not found in environment variable '{api_key_env}'")
        return False
    
    print(f"✅ API key found")
    
    try:
        # Create Spark session
        spark = create_spark_session("translation_test", config.get('spark', {}))
        
        # Read data
        print(f"\n📖 Reading input file: {config.get('input_file')}")
        data_reader = DataReader(spark, config)
        df = data_reader.read_data()
        
        # Create sample
        print(f"\n🎲 Creating test sample...")
        sample_df = create_test_sample(df, sample_percentage, max_rows)
        actual_sample_size = sample_df.count()
        
        if actual_sample_size == 0:
            print("❌ ERROR: No data in sample")
            return False
        
        # Initialize translation components
        cache_manager = CacheManager(config)
        translation_manager = TranslationManager(config, cache_manager)
        
        columns_to_translate = config.get('columns_to_translate', [])
        print(f"\n🔤 Columns to translate: {columns_to_translate}")
        
        # Convert to pandas for easier display
        print(f"\n⚙️  Running translation test...")
        sample_pandas = sample_df.toPandas()
        
        # Show original sample
        print(f"\n📋 ORIGINAL SAMPLE ({actual_sample_size} rows):")
        print("-" * 80)
        for col in columns_to_translate:
            if col in sample_pandas.columns:
                print(f"\n{col.upper()}:")
                unique_texts = sample_pandas[col].dropna().unique()[:10]  # Show max 10 unique
                for i, text in enumerate(unique_texts, 1):
                    if text and str(text).strip():
                        print(f"  {i:2d}. {str(text)[:70]}{'...' if len(str(text)) > 70 else ''}")
        
        # Run translation
        translated_df = translation_manager.process_dataframe(sample_df)
        translated_pandas = translated_df.toPandas()
        
        # Show translated results
        target_language = config.get('target_language', 'english')
        print(f"\n🌐 TRANSLATED RESULTS (to {target_language}):")
        print("-" * 80)
        
        success_count = 0
        total_count = 0
        
        for col in columns_to_translate:
            if col in sample_pandas.columns:
                translated_col = f"{col}_{target_language}"
                if translated_col in translated_pandas.columns:
                    print(f"\n{col.upper()} → {translated_col.upper()}:")
                    
                    # Compare original vs translated
                    for idx, row in translated_pandas.head(10).iterrows():
                        original = str(row[col]) if pd.notna(row[col]) else ""
                        translated = str(row[translated_col]) if pd.notna(row[translated_col]) else ""
                        
                        if original.strip():
                            total_count += 1
                            # Check if translation is different (indicating actual translation occurred)
                            if translated != original and translated.strip():
                                success_count += 1
                                status = "✅"
                            elif translated == original:
                                status = "🔄"  # Same as original (might be already in target language)
                            else:
                                status = "❌"  # Failed translation
                            
                            print(f"  {status} {original[:35]:<35} → {translated[:35]}")
        
        # Show statistics
        stats = translation_manager.get_stats()
        print(f"\n📊 TRANSLATION STATISTICS:")
        print("-" * 40)
        print(f"Total API calls: {stats.get('api_calls', 0)}")
        print(f"Cache hits: {stats.get('cached_hits', 0)}")
        print(f"Errors: {stats.get('errors', 0)}")
        print(f"Translations attempted: {total_count}")
        print(f"Successful translations: {success_count}")
        
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
            print(f"Success rate: {success_rate:.1f}%")
            
            if success_rate > 80:
                print(f"\n🎉 SUCCESS: Translation quality looks good!")
                result = True
            elif success_rate > 50:
                print(f"\n⚠️  WARNING: Translation quality is moderate. Check results above.")
                result = True
            else:
                print(f"\n❌ POOR: Translation quality is low. Check configuration.")
                result = False
        else:
            print(f"\n❌ ERROR: No texts were processed for translation")
            result = False
        
        # Clean up
        spark.stop()
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test translation on a small sample')
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    parser.add_argument('--sample-percent', type=float, default=1.0, 
                       help='Percentage of data to sample (default: 1.0)')
    parser.add_argument('--max-rows', type=int, default=100,
                       help='Maximum number of rows to test (default: 100)')
    
    args = parser.parse_args()
    
    success = test_translation(args.config, args.sample_percent, args.max_rows)
    
    if success:
        print(f"\n✅ Test completed successfully. You can now run the full translation.")
        sys.exit(0)
    else:
        print(f"\n❌ Test failed. Please check the configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()