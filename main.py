#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main orchestrator for the distributed translation system.
Coordinates workflow between data reading, translation, caching, and checkpointing.
"""

import os
import sys
import time
import logging
import argparse
import json
from typing import Dict, List, Any, Optional, Tuple
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from config import ConfigManager
from modules.file_manager import DataReader, DataWriter
from modules.translator import TranslationManager
from modules.cache import CacheManager
from modules.checkpoint import CheckpointManager
from modules.utilities import set_up_logging, create_spark_session, format_stats_report


class TranslationOrchestrator:
    """
    Main orchestrator that coordinates the entire translation process.
    """
    
    def __init__(self, config_path: str, cli_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the orchestrator with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
            cli_args: Command line arguments that override config file settings
        """
        self.config_manager = ConfigManager(config_path, cli_args)
        self.config = self.config_manager.get_config()
        
        self.logger = set_up_logging(
            self.config.get('logging', {}).get('level', 'INFO'),
            self.config.get('logging', {}).get('log_file', 'translation_process.log')
        )
        
        self.spark = create_spark_session(
            app_name="distributed_translation",
            config=self.config.get('spark', {})
        )
        
        self.data_reader = DataReader(self.spark, self.config)
        self.data_writer = DataWriter(self.spark, self.config)
        self.cache_manager = CacheManager(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)
        self.translation_manager = TranslationManager(self.config, self.cache_manager)
        
        self.stats = {
            'total_rows': 0,
            'translated_rows': 0,
            'cached_hits': 0,
            'api_calls': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
        }
    
    def initialize(self) -> bool:
        """
        Performs initialization tasks before starting the process.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            self.logger.info("Initializing translation process")
            
            # Validate required configuration
            if not self.config.get('input_file'):
                self.logger.error("Input file not specified in configuration")
                return False
            
            if not self.config.get('output_file'):
                self.logger.error("Output file not specified in configuration")
                return False
            
            if not self.config.get('columns_to_translate'):
                self.logger.error("No columns specified for translation")
                return False
            
            # Check if OpenAI API key is set
            api_key_env = self.config.get('openai', {}).get('api_key_env', 'OPENAI_API_KEY')
            if not os.environ.get(api_key_env):
                self.logger.error(f"OpenAI API key environment variable '{api_key_env}' not set")
                return False
            
            # Create output directory if it doesn't exist
            output_file = self.config.get('output_file')
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Create intermediate directory if needed
            if self.config.get('write_intermediate_results', False):
                intermediate_dir = self.config.get('intermediate_directory', './intermediate')
                os.makedirs(intermediate_dir, exist_ok=True)
                
            # Initialize start time
            self.stats['start_time'] = time.time()
            
            # Clear checkpoints if force restart is specified
            if self.config.get('force_restart', False):
                self.checkpoint_manager.clear_checkpoints()
                
            self.logger.info("Initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def process(self) -> bool:
        """
        Main processing method that orchestrates the entire workflow.
        
        Returns:
            bool: True if processing completed successfully, False otherwise
        """
        try:
            # Check if we should resume from checkpoint
            resume = self.checkpoint_manager.resume_or_restart(
                force_restart=self.config.get('force_restart', False)
            )
            
            if resume:
                self.logger.info("Resuming from checkpoint")
            else:
                self.logger.info("Starting new translation process")
            
            # Read the input data
            df = self.data_reader.read_data()
            
            # Get total row count
            total_rows = df.count()
            self.stats['total_rows'] = total_rows
            self.logger.info(f"Processing {total_rows} rows from input file")
            
            # Determine number of partitions
            default_parallelism = self.config.get('spark', {}).get('default_parallelism', 4)
            num_partitions = self.config.get('num_partitions', default_parallelism)
            
            # Repartition if needed
            if df.rdd.getNumPartitions() != num_partitions:
                df = df.repartition(num_partitions)
                
            self.logger.info(f"Data split into {num_partitions} partitions")
            
            # Process each partition
            partitions_processed = 0
            translation_results = []
            
            for partition_id in range(num_partitions):
                # Check if this partition has already been processed (if resuming)
                if resume and self.checkpoint_manager.has_checkpoint(partition_id):
                    self.logger.info(f"Partition {partition_id} already processed, skipping")
                    partitions_processed += 1
                    continue
                    
                self.logger.info(f"Processing partition {partition_id} of {num_partitions}")
                
                # Get the partition data
                partition_df = df.filter(F.spark_partition_id() == partition_id)
                
                # Process the partition
                start_time = time.time()
                processed_df = self.translation_manager.process_dataframe(partition_df)
                
                # Update statistics
                partition_stats = self.translation_manager.get_stats()
                partition_stats['processing_time'] = time.time() - start_time
                self.update_stats(partition_stats)
                
                # Save checkpoint
                if self.checkpoint_manager.should_checkpoint(partition_id):
                    self.checkpoint_manager.mark_partition_processed(partition_id, processed_df)
                    
                # Save intermediate results if configured
                if self.config.get('write_intermediate_results', False):
                    self.data_writer.write_intermediate_data(processed_df, partition_id)
                    
                # Add to results
                translation_results.append(processed_df)
                partitions_processed += 1
                
                # Log progress
                self.log_progress(partitions_processed, num_partitions)
            
            # Combine all processed partitions
            if translation_results:
                combined_df = translation_results[0]
                for df in translation_results[1:]:
                    combined_df = combined_df.unionAll(df)
            else:
                # If all partitions were loaded from checkpoints, apply cache to the original DataFrame
                combined_df = self.translation_manager.apply_translations(df)
            
            # Write the final output
            self.logger.info("Writing translated data to output file")
            self.data_writer.write_data(combined_df)
            
            # Save final checkpoint state
            self.checkpoint_manager.save_global_state(self.stats)
            
            self.logger.info("Processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
            
        finally:
            # Update end time
            self.stats['end_time'] = time.time()
    
    def update_stats(self, partition_stats: Dict[str, Any]) -> None:
        """
        Updates global statistics with partition statistics.
        
        Args:
            partition_stats: Statistics from processing a partition
        """
        # Update counts
        self.stats['translated_rows'] += partition_stats.get('translated_rows', 0)
        self.stats['cached_hits'] += partition_stats.get('cached_hits', 0)
        self.stats['api_calls'] += partition_stats.get('api_calls', 0)
        self.stats['errors'] += partition_stats.get('errors', 0)
        
        # Track languages if available
        if 'languages' in partition_stats:
            if 'languages' not in self.stats:
                self.stats['languages'] = {
                    'source': set(),
                    'target': partition_stats['languages']['target']
                }
            
            # Update source languages
            for lang in partition_stats['languages'].get('source', []):
                self.stats['languages']['source'].add(lang)
                
        # Track performance metrics
        if 'processing_time' in partition_stats:
            if 'performance' not in self.stats:
                self.stats['performance'] = {
                    'total_processing_time': 0,
                    'partition_times': []
                }
            
            self.stats['performance']['total_processing_time'] += partition_stats['processing_time']
            self.stats['performance']['partition_times'].append(partition_stats['processing_time'])
            
            # Calculate average translation time
            if 'translated_rows' in partition_stats and partition_stats['translated_rows'] > 0:
                avg_time = partition_stats['processing_time'] / partition_stats['translated_rows']
                if 'avg_translation_time' not in self.stats['performance']:
                    self.stats['performance']['avg_translation_time'] = avg_time
                else:
                    # Running average
                    current = self.stats['performance']['avg_translation_time']
                    count = len(self.stats['performance']['partition_times'])
                    self.stats['performance']['avg_translation_time'] = (current * (count - 1) + avg_time) / count
            
            # Calculate texts per second
            num_rows = partition_stats.get('translated_rows', 0) + partition_stats.get('cached_hits', 0)
            if num_rows > 0 and partition_stats['processing_time'] > 0:
                texts_per_second = num_rows / partition_stats['processing_time']
                if 'texts_per_second' not in self.stats['performance']:
                    self.stats['performance']['texts_per_second'] = texts_per_second
                else:
                    # Running average
                    current = self.stats['performance']['texts_per_second']
                    count = len(self.stats['performance']['partition_times'])
                    self.stats['performance']['texts_per_second'] = (current * (count - 1) + texts_per_second) / count
    
    def log_progress(self, current_partition: int, total_partitions: int) -> None:
        """
        Logs the current progress of the process.
        
        Args:
            current_partition: Current partition being processed
            total_partitions: Total number of partitions
        """
        # Calculate progress percentage
        progress_pct = (current_partition / total_partitions) * 100
        
        # Calculate elapsed time
        elapsed = time.time() - self.stats['start_time']
        
        # Estimate remaining time
        if current_partition > 0:
            estimated_total = elapsed * (total_partitions / current_partition)
            remaining = estimated_total - elapsed
            time_str = f"Elapsed: {self._format_time(elapsed)}, Remaining: {self._format_time(remaining)}"
        else:
            time_str = f"Elapsed: {self._format_time(elapsed)}"
        
        # Log progress
        self.logger.info(
            f"Progress: {current_partition}/{total_partitions} partitions "
            f"({progress_pct:.1f}%) - {time_str}"
        )
        
        # Log statistics
        self.logger.info(
            f"Statistics: {self.stats['translated_rows']} translations, "
            f"{self.stats['cached_hits']} cache hits, "
            f"{self.stats['api_calls']} API calls, "
            f"{self.stats['errors']} errors"
        )

    def _format_time(self, seconds: float) -> str:
        """
        Format time in a human-readable way.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        else:
            return f"{s}s"
    
    def finalize(self) -> None:
        """
        Performs finalization tasks after completing the process.
        """
        # Set end time if not already set
        if not self.stats.get('end_time'):
            self.stats['end_time'] = time.time()
        
        # Calculate total elapsed time
        elapsed = self.stats['end_time'] - self.stats['start_time']
        
        # Format and log final statistics report
        report = format_stats_report(self.stats)
        self.logger.info(f"Translation process summary:\n{report}")
        
        # Flush cache to ensure all writes are saved
        self.cache_manager.cache.flush()
        
        # Clean up cache if configured
        if self.config.get('cleanup_cache_on_completion', False):
            self.logger.info("Cleaning up expired cache entries")
            removed = self.cache_manager.cache.cleanup()
            self.logger.info(f"Removed {removed} expired cache entries")
        
        # Write statistics to file if configured
        if self.config.get('write_stats_to_file', False):
            try:
                stats_file = self.config.get('stats_file', 'translation_stats.json')
                with open(stats_file, 'w') as f:
                    # Convert any sets to lists for JSON serialization
                    stats_copy = self.stats.copy()
                    if 'languages' in stats_copy and 'source' in stats_copy['languages'] and isinstance(stats_copy['languages']['source'], set):
                        stats_copy['languages']['source'] = list(stats_copy['languages']['source'])
                    
                    json.dump(stats_copy, f, indent=2)
                self.logger.info(f"Statistics written to {stats_file}")
            except Exception as e:
                self.logger.error(f"Failed to write statistics file: {str(e)}")


def main():
    """
    Entry point for the translation process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Distributed Translation System')
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    parser.add_argument('--input_file', help='Override input file path')
    parser.add_argument('--output_file', help='Override output file path')
    parser.add_argument('--target_language', help='Override target language')
    parser.add_argument('--columns', help='Comma-separated list of columns to translate')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
    parser.add_argument('--force_restart', action='store_true', help='Force restart (ignore checkpoints)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Convert args to dictionary for config overrides
    cli_args = {}
    
    # Process explicit overrides
    if args.input_file:
        cli_args['input_file'] = args.input_file
    if args.output_file:
        cli_args['output_file'] = args.output_file
    if args.target_language:
        cli_args['target_language'] = args.target_language
    if args.columns:
        cli_args['columns_to_translate'] = args.columns.split(',')
        
    # Set log level if verbose
    if args.verbose:
        cli_args['logging'] = {'level': 'DEBUG'}
    
    # Process resume/force_restart flags
    if args.resume:
        cli_args['resume_from_checkpoint'] = True
    if args.force_restart:
        cli_args['force_restart'] = True
        cli_args['resume_from_checkpoint'] = False
    
    # Create and run the orchestrator
    orchestrator = TranslationOrchestrator(args.config, cli_args)
    
    try:
        # Initialize
        if not orchestrator.initialize():
            print("Initialization failed. See log for details.")
            sys.exit(1)
        
        # Process
        success = orchestrator.process()
        
        # Finalize
        orchestrator.finalize()
        
        if success:
            print("Translation process completed successfully.")
            sys.exit(0)
        else:
            print("Translation process failed. See log for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        orchestrator.finalize()
        sys.exit(130)
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        orchestrator.logger.error(f"Unhandled exception: {str(e)}")
        import traceback
        orchestrator.logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()