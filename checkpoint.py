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