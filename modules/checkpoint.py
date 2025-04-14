#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpoint management module for the distributed translation system.
Handles saving and restoring processing state to enable fault tolerance.
"""

import os
import json
import time
import glob
from typing import Dict, List, Any, Optional, Set
import logging
import shutil

from pyspark.sql import DataFrame


class CheckpointManager:
    """
    Manages checkpoints for fault-tolerant processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the checkpoint manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.enabled = config.get('checkpoint', {}).get('enabled', True)
        self.interval = config.get('checkpoint', {}).get('interval', 1)
        self.directory = config.get('checkpoint', {}).get('directory', './checkpoints')
        self.max_checkpoints = config.get('checkpoint', {}).get('max_checkpoints', 5)
        
        # Track processed partitions
        self.processed_partitions = set()
        
        # Create checkpoint directory if it doesn't exist
        if self.enabled:
            os.makedirs(self.directory, exist_ok=True)
            self._load_processed_partitions()
    
    def _load_processed_partitions(self) -> None:
        """
        Load the set of already processed partitions from existing checkpoints.
        """
        if not self.enabled:
            return
            
        try:
            # Look for metadata file
            metadata_path = os.path.join(self.directory, 'checkpoint_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.processed_partitions = set(metadata.get('processed_partitions', []))
                    
                    self.logger.info(f"Loaded {len(self.processed_partitions)} processed partitions from checkpoint metadata")
            else:
                # Legacy approach: infer from checkpoint files
                checkpoint_files = glob.glob(os.path.join(self.directory, 'checkpoint_*.parquet'))
                for f in checkpoint_files:
                    try:
                        # Extract partition ID from filename
                        partition_id = int(os.path.basename(f).split('_')[1].split('.')[0])
                        self.processed_partitions.add(partition_id)
                    except (ValueError, IndexError):
                        pass
                
                self.logger.info(f"Inferred {len(self.processed_partitions)} processed partitions from checkpoint files")
                
                # Save to metadata file for future use
                self._save_metadata()
        except Exception as e:
            self.logger.error(f"Error loading processed partitions: {str(e)}")
    
    def _save_metadata(self) -> None:
        """
        Save checkpoint metadata to a JSON file.
        """
        if not self.enabled:
            return
            
        try:
            metadata = {
                'processed_partitions': list(self.processed_partitions),
                'timestamp': int(time.time()),
                'count': len(self.processed_partitions)
            }
            
            metadata_path = os.path.join(self.directory, 'checkpoint_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            self.logger.debug(f"Saved checkpoint metadata with {len(self.processed_partitions)} processed partitions")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint metadata: {str(e)}")
    
    def should_checkpoint(self, partition_id: int) -> bool:
        """
        Determine if a checkpoint should be created for this partition.
        
        Args:
            partition_id: Current partition ID
            
        Returns:
            Boolean indicating if checkpoint should be created
        """
        if not self.enabled:
            return False
            
        # Always checkpoint if this is the first partition or based on the interval
        return partition_id == 0 or partition_id % self.interval == 0
    
    def has_checkpoint(self, partition_id: int) -> bool:
        """
        Check if a checkpoint exists for a specific partition.
        
        Args:
            partition_id: Partition ID to check
            
        Returns:
            Boolean indicating if checkpoint exists
        """
        if not self.enabled:
            return False
            
        return partition_id in self.processed_partitions
    
    def get_next_unprocessed_partition(self, total_partitions: int) -> Optional[int]:
        """
        Get the next partition that hasn't been processed yet.
        
        Args:
            total_partitions: Total number of partitions
            
        Returns:
            Next unprocessed partition ID or None if all processed
        """
        if not self.enabled:
            return 0
            
        for i in range(total_partitions):
            if i not in self.processed_partitions:
                return i
                
        return None
    
    def mark_partition_processed(self, partition_id: int, df: Optional[DataFrame] = None) -> None:
        """
        Mark a partition as processed and optionally save checkpoint data.
        
        Args:
            partition_id: Partition ID that was processed
            df: DataFrame containing the processed data (optional)
        """
        if not self.enabled:
            return
            
        # Add to processed set
        self.processed_partitions.add(partition_id)
        
        # Only save metadata if checkpointing is due
        if self.should_checkpoint(partition_id):
            self._save_metadata()
            
            # Save DataFrame checkpoint if provided
            if df is not None:
                checkpoint_path = os.path.join(self.directory, f'checkpoint_{partition_id}.parquet')
                try:
                    df.write.mode('overwrite').parquet(checkpoint_path)
                    self.logger.info(f"Saved checkpoint for partition {partition_id}")
                except Exception as e:
                    self.logger.error(f"Error saving checkpoint data for partition {partition_id}: {str(e)}")
            
            # Clean up old checkpoints if needed
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self) -> None:
        """
        Remove old checkpoints to stay within the maximum limit.
        """
        if not self.enabled or self.max_checkpoints <= 0:
            return
            
        try:
            # Get all checkpoint files
            checkpoint_files = glob.glob(os.path.join(self.directory, 'checkpoint_*.parquet'))
            
            # Sort by modification time (oldest first)
            checkpoint_files.sort(key=os.path.getmtime)
            
            # Remove oldest files if we have too many
            files_to_remove = len(checkpoint_files) - self.max_checkpoints
            if files_to_remove > 0:
                for i in range(files_to_remove):
                    os.remove(checkpoint_files[i])
                    self.logger.info(f"Removed old checkpoint: {os.path.basename(checkpoint_files[i])}")
        except Exception as e:
            self.logger.error(f"Error cleaning up old checkpoints: {str(e)}")
    
    def get_checkpoint_path(self, partition_id: int) -> str:
        """
        Get the path to a specific checkpoint file.
        
        Args:
            partition_id: Partition ID
            
        Returns:
            Path to the checkpoint file
        """
        return os.path.join(self.directory, f'checkpoint_{partition_id}.parquet')
    
    def get_processed_count(self) -> int:
        """
        Get the number of partitions that have been processed.
        
        Returns:
            Count of processed partitions
        """
        return len(self.processed_partitions)
    
    def clear_checkpoints(self) -> bool:
        """
        Clear all checkpoint data.
        
        Returns:
            Boolean indicating success
        """
        if not self.enabled:
            return True
            
        try:
            # Remove all checkpoint files
            shutil.rmtree(self.directory)
            os.makedirs(self.directory, exist_ok=True)
            
            # Reset processed partitions
            self.processed_partitions = set()
            
            self.logger.info("All checkpoints cleared")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing checkpoints: {str(e)}")
            return False
    
    def resume_or_restart(self, force_restart: bool = False) -> bool:
        """
        Determine if processing should resume from checkpoint or restart.
        
        Args:
            force_restart: Force restarting from scratch
            
        Returns:
            True to resume, False to restart
        """
        if not self.enabled or force_restart:
            return False
            
        return len(self.processed_partitions) > 0
    
    def get_checkpoint_state(self) -> Dict[str, Any]:
        """
        Get the current checkpoint state information.
        
        Returns:
            Dictionary containing checkpoint state details
        """
        if not self.enabled:
            return {
                'enabled': False,
                'processed_count': 0,
                'state': 'disabled'
            }
            
        return {
            'enabled': True,
            'processed_count': len(self.processed_partitions),
            'processed_partitions': sorted(list(self.processed_partitions)),
            'checkpoint_directory': self.directory,
            'interval': self.interval,
            'max_checkpoints': self.max_checkpoints,
            'last_updated': int(time.time())
        }
    
    def load_checkpoint_data(self, partition_id: int) -> Optional[DataFrame]:
        """
        Load DataFrame data from a specific checkpoint.
        
        Args:
            partition_id: Partition ID to load
            
        Returns:
            DataFrame from checkpoint or None if not found/valid
        """
        if not self.enabled or partition_id not in self.processed_partitions:
            return None
            
        checkpoint_path = self.get_checkpoint_path(partition_id)
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint file not found for partition {partition_id} at {checkpoint_path}")
            return None
            
        try:
            # This requires a SparkSession to be available
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            
            df = spark.read.parquet(checkpoint_path)
            self.logger.info(f"Loaded checkpoint data for partition {partition_id}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading checkpoint data for partition {partition_id}: {str(e)}")
            return None
    
    def save_global_state(self, state: Dict[str, Any]) -> bool:
        """
        Save global processing state to a checkpoint file.
        
        Args:
            state: Dictionary containing global state information
            
        Returns:
            Boolean indicating success
        """
        if not self.enabled:
            return False
            
        try:
            # Add timestamp to state
            state_with_timestamp = {
                **state,
                'timestamp': int(time.time()),
                'processed_partitions': sorted(list(self.processed_partitions))
            }
            
            # Save to state file
            state_path = os.path.join(self.directory, 'global_state.json')
            with open(state_path, 'w') as f:
                json.dump(state_with_timestamp, f, indent=2)
                
            self.logger.info("Saved global processing state to checkpoint")
            return True
        except Exception as e:
            self.logger.error(f"Error saving global state: {str(e)}")
            return False
    
    def load_global_state(self) -> Optional[Dict[str, Any]]:
        """
        Load global processing state from checkpoint file.
        
        Returns:
            Dictionary containing global state or None if not found/valid
        """
        if not self.enabled:
            return None
            
        state_path = os.path.join(self.directory, 'global_state.json')
        if not os.path.exists(state_path):
            self.logger.info("No global state checkpoint found")
            return None
            
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            self.logger.info("Loaded global processing state from checkpoint")
            return state
        except Exception as e:
            self.logger.error(f"Error loading global state: {str(e)}")
            return None
    
    def get_most_recent_checkpoint_time(self) -> Optional[int]:
        """
        Get the timestamp of the most recent checkpoint.
        
        Returns:
            Unix timestamp of the most recent checkpoint or None if no checkpoints exist
        """
        if not self.enabled:
            return None
            
        try:
            # Check metadata file first
            metadata_path = os.path.join(self.directory, 'checkpoint_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('timestamp')
            
            # If no metadata, check modification time of checkpoint files
            checkpoint_files = glob.glob(os.path.join(self.directory, 'checkpoint_*.parquet'))
            if not checkpoint_files:
                return None
                
            # Get most recent modification time
            most_recent = max(os.path.getmtime(f) for f in checkpoint_files)
            return int(most_recent)
        except Exception as e:
            self.logger.error(f"Error getting most recent checkpoint time: {str(e)}")
            return None
    
    def checkpoint_exists(self) -> bool:
        """
        Check if any checkpoints exist.
        
        Returns:
            Boolean indicating if checkpoints exist
        """
        if not self.enabled:
            return False
            
        metadata_path = os.path.join(self.directory, 'checkpoint_metadata.json')
        if os.path.exists(metadata_path):
            return True
            
        checkpoint_files = glob.glob(os.path.join(self.directory, 'checkpoint_*.parquet'))
        return len(checkpoint_files) > 0
    
    def is_processing_complete(self, total_partitions: int) -> bool:
        """
        Check if processing is complete based on checkpoints.
        
        Args:
            total_partitions: Total number of partitions
            
        Returns:
            Boolean indicating if all partitions have been processed
        """
        if not self.enabled:
            return False
            
        # Check if all partition IDs are in processed_partitions
        return all(partition_id in self.processed_partitions for partition_id in range(total_partitions))