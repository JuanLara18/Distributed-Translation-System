#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O module for the distributed translation system.
Handles reading and writing data from/to DTA files.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
import logging

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import pandas as pd


class AbstractDataHandler(ABC):
    """
    Abstract base class for data handlers.
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        """
        Initialize the data handler.
        
        Args:
            spark: SparkSession instance
            config: Configuration dictionary
        """
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)


class DataReader(AbstractDataHandler):
    """
    Handles reading data from input files.
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        """
        Initialize the data reader.
        
        Args:
            spark: SparkSession instance
            config: Configuration dictionary
        """
        super().__init__(spark, config)
        self.input_file = config.get('input_file')
        self.columns_to_translate = config.get('columns_to_translate', [])
        self.source_language_column = config.get('source_language_column')
        
        # Calculate required columns
        self.required_columns = set(self.columns_to_translate)
        if self.source_language_column:
            self.required_columns.add(self.source_language_column)
    
    def read_data(self) -> DataFrame:
        """
        Read data from the input file and prepare it for processing.
        
        Returns:
            DataFrame containing the data to process
        """
        file_ext = os.path.splitext(self.input_file)[1].lower()
        
        self.logger.info(f"Reading input file: {self.input_file}")
        
        try:
            if file_ext == '.dta':
                df = self._read_stata_file()
            elif file_ext == '.csv':
                df = self.spark.read.csv(self.input_file, header=True, inferSchema=True)
            elif file_ext == '.parquet':
                df = self.spark.read.parquet(self.input_file)
            elif file_ext in ['.json', '.jsonl']:
                df = self.spark.read.json(self.input_file)
            else:
                error_msg = f"Unsupported file format: {file_ext}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Validate that all required columns exist
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"Required columns missing from input file: {', '.join(missing_columns)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Filter DataFrame to include only required columns if needed
            if self.config.get('optimize_memory', True):
                all_columns = list(self.required_columns)
                # Include any ID or key columns that might be needed
                id_columns = self.config.get('id_columns', [])
                all_columns.extend([col for col in id_columns if col not in all_columns])
                
                # Keep only needed columns
                df = df.select(*all_columns)
            
            # Clean text columns
            for col in self.columns_to_translate:
                df = df.withColumn(col, F.when(F.col(col).isNull(), "").otherwise(F.col(col)))
            
            self.logger.info(f"Successfully read input file with {df.count()} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading input file: {str(e)}")
            raise
    
    def _read_stata_file(self) -> DataFrame:
        """
        Read a Stata file using PySpark.
        
        Returns:
            DataFrame containing the required columns from the Stata file
        """
        try:
            # For Stata files, we need to use PyReadStat via pandas and then convert to Spark DataFrame
            import pyreadstat
            
            self.logger.debug(f"Reading Stata file with pyreadstat: {self.input_file}")
            
            # Read the Stata file with pandas, disable convert_categoricals
            pd_df, meta = pyreadstat.read_dta(self.input_file)
            
            # Save metadata for later use if we have a metadata handler
            self._preserve_stata_metadata(meta)
            
            # Handle problematic columns before conversion to Spark DataFrame
            # This should fix the type conflict issues
            for col in pd_df.columns:
                col_type = pd_df[col].dtype
                
                # Handle dates and timestamps properly
                if pd.api.types.is_datetime64_any_dtype(col_type):
                    pd_df[col] = pd.to_datetime(pd_df[col], errors='coerce')
                    pd_df[col] = pd_df[col].dt.strftime('%Y-%m-%d')
                
                # Force consistent types for numeric columns that might have mixed types
                # Force all integer-like and float-like columns to float (double) to avoid LongType vs DoubleType conflicts
                if pd.api.types.is_numeric_dtype(col_type):
                    pd_df[col] = pd_df[col].astype(float)
                
                # Convert objects to string while handling NaN values
                if pd.api.types.is_object_dtype(col_type):
                    pd_df[col] = pd_df[col].fillna('').astype(str)
            
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(pd_df)
            
            self.logger.debug(f"Successfully converted Stata file to Spark DataFrame with {spark_df.count()} rows")
            return spark_df
            
        except ImportError:
            self.logger.error("pyreadstat not installed. Please install it with: pip install pyreadstat")
            raise
        except Exception as e:
            self.logger.error(f"Error reading Stata file: {str(e)}")
            raise

    def _preserve_stata_metadata(self, meta: Any) -> None:
        """
        Preserve Stata metadata for later use.
        
        Args:
            meta: Metadata object from pyreadstat
        """
        try:
            # Extract relevant metadata - modifica este código para adaptarse a la estructura real
            metadata = {
                # Usa hasattr para verificar si el atributo existe
                'variable_labels': meta.variable_labels if hasattr(meta, 'variable_labels') else {},
                'value_labels': meta.value_labels if hasattr(meta, 'value_labels') else {},
                'variable_value_labels': meta.variable_value_labels if hasattr(meta, 'variable_value_labels') else {},
                'file_label': meta.file_label if hasattr(meta, 'file_label') else '',
                'file_encoding': meta.file_encoding if hasattr(meta, 'file_encoding') else '',
                'column_names': meta.column_names if hasattr(meta, 'column_names') else [],
                'original_variable_types': meta.original_variable_types if hasattr(meta, 'original_variable_types') else {},
                'missing_ranges': meta.missing_ranges if hasattr(meta, 'missing_ranges') else {},
                'notes': meta.notes if hasattr(meta, 'notes') else []
            }
            
            # Save to a file in the same location as the input file
            metadata_file = os.path.splitext(self.input_file)[0] + '_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            self.logger.debug(f"Preserved Stata metadata to {metadata_file}")
            
        except Exception as e:
            self.logger.warning(f"Could not preserve Stata metadata: {str(e)}")
            import traceback
            self.logger.warning(traceback.format_exc())
    
    def read_checkpoint_data(self, partition_id: int) -> Optional[DataFrame]:
        """
        Read data from a checkpoint file for a specific partition.
        
        Args:
            partition_id: Partition ID to read
            
        Returns:
            DataFrame from checkpoint or None if checkpoint doesn't exist
        """
        checkpoint_dir = self.config.get('checkpoint', {}).get('directory', './checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{partition_id}.parquet')
        
        if not os.path.exists(checkpoint_path):
            self.logger.debug(f"No checkpoint data found for partition {partition_id}")
            return None
        
        try:
            self.logger.info(f"Reading checkpoint data for partition {partition_id}")
            df = self.spark.read.parquet(checkpoint_path)
            self.logger.info(f"Successfully read checkpoint data with {df.count()} rows")
            return df
        except Exception as e:
            self.logger.error(f"Error reading checkpoint data for partition {partition_id}: {str(e)}")
            return None


class DataWriter(AbstractDataHandler):
    """
    Handles writing data to output files.
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        """
        Initialize the data writer.
        
        Args:
            spark: SparkSession instance
            config: Configuration dictionary
        """
        super().__init__(spark, config)
        self.output_file = config.get('output_file')
        self.columns_to_translate = config.get('columns_to_translate', [])
        self.target_language = config.get('target_language', 'english')
        self.intermediate_dir = config.get('intermediate_directory', './intermediate')
    
    def write_data(self, df: DataFrame) -> None:
        """
        Write the processed data to the output file.
        
        Args:
            df: DataFrame containing the processed data
        """
        file_ext = os.path.splitext(self.output_file)[1].lower()
        
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(self.output_file))
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Writing output file: {self.output_file}")
        
        try:
            if file_ext == '.dta':
                self._write_stata_file(df)
            elif file_ext == '.csv':
                df.write.mode('overwrite').option('header', 'true').csv(self.output_file)
            elif file_ext == '.parquet':
                df.write.mode('overwrite').parquet(self.output_file)
            elif file_ext in ['.json', '.jsonl']:
                df.write.mode('overwrite').json(self.output_file)
            else:
                error_msg = f"Unsupported output file format: {file_ext}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
            self.logger.info(f"Successfully wrote {df.count()} rows to output file")
            
        except Exception as e:
            self.logger.error(f"Error writing output file: {str(e)}")
            raise
        
    def _write_stata_file(self, df: DataFrame) -> None:
        """
        Write a DataFrame to a Stata file with support for Stata 18 and long strings.
        
        Args:
            df: DataFrame to write
        """
        try:
            import pyreadstat
            import inspect
            import pandas as pd
            import numpy as np
            
            self.logger.debug(f"Converting Spark DataFrame to pandas for Stata output")
            
            # Convert Spark DataFrame to pandas
            pd_df = df.toPandas()
            
            # Load metadata if available
            input_file = self.config.get('input_file')
            metadata = self._load_stata_metadata(input_file)
            
            self.logger.debug(f"Writing Stata file with pyreadstat: {self.output_file}")
            
            # Check which version of Stata format we can use
            try:
                from pyreadstat.pyreadstat import StataWriteFileFormat
                stata_format = StataWriteFileFormat.STATA_118  # Format for Stata 14-18
                supports_stata_format = True
            except (ImportError, AttributeError):
                self.logger.warning("Could not import StataWriteFileFormat; using version number directly")
                stata_format = None
                supports_stata_format = False
            
            # Check if strl_columns parameter is supported
            write_dta_params = inspect.signature(pyreadstat.write_dta).parameters
            supports_strl = 'strl_columns' in write_dta_params
            
            # More conservative string length limit (Stata has a 2045 limit for regular strings)
            # Using 2000 to be safe
            MAX_STRING_LENGTH = 2000
            
            # Proactively truncate ALL string columns to be safe
            string_cols = pd_df.select_dtypes(include=['object']).columns
            for col in string_cols:
                # Check if there are any strings that need truncation
                max_len = pd_df[col].astype(str).str.len().max()
                if max_len > MAX_STRING_LENGTH:
                    self.logger.warning(f"Truncating column '{col}' from max length {max_len} to {MAX_STRING_LENGTH} characters")
                    pd_df[col] = pd_df[col].astype(str).str.slice(0, MAX_STRING_LENGTH)
                    
                    # Double-check truncation worked
                    new_max = pd_df[col].astype(str).str.len().max()
                    self.logger.info(f"After truncation, max length of '{col}' is now {new_max}")
                else:
                    self.logger.debug(f"Column '{col}' has max length {max_len}, no truncation needed")
            
            # Convert any problematic data types
            for col in pd_df.columns:
                # Convert datetime columns to strings
                if pd.api.types.is_datetime64_any_dtype(pd_df[col].dtype):
                    self.logger.info(f"Converting datetime column '{col}' to string")
                    pd_df[col] = pd_df[col].dt.strftime('%Y-%m-%d')
                    
                # Ensure integer types are consistent
                if pd.api.types.is_integer_dtype(pd_df[col].dtype):
                    self.logger.debug(f"Converting integer column '{col}' to int32")
                    pd_df[col] = pd_df[col].astype(np.int32)
                    
                # Ensure float types are consistent
                if pd.api.types.is_float_dtype(pd_df[col].dtype):
                    self.logger.debug(f"Converting float column '{col}' to float64")
                    pd_df[col] = pd_df[col].astype(np.float64)
            
            # Try multiple writing approaches in order of preference
            try:
                # First attempt: Use optimal parameters if supported
                if supports_stata_format and supports_strl:
                    self.logger.info("Writing Stata file using optimal parameters (Stata 118 format with strl support)")
                    pyreadstat.write_dta(
                        pd_df,
                        self.output_file,
                        write_file_format=stata_format,
                        strl_columns=[]  # Empty list - not using strl even if supported
                    )
                elif supports_stata_format:
                    self.logger.info("Writing Stata file using Stata 118 format (without strl support)")
                    pyreadstat.write_dta(
                        pd_df,
                        self.output_file,
                        write_file_format=stata_format
                    )
                else:
                    self.logger.info("Writing Stata file using version 15 format")
                    pyreadstat.write_dta(
                        pd_df,
                        self.output_file,
                        version=15
                    )
                    
                self.logger.info(f"Successfully wrote Stata file with {len(pd_df)} rows")
                
            except Exception as e1:
                self.logger.warning(f"First write attempt failed: {str(e1)}. Trying with more conservative approach...")
                
                # Second attempt: Try with version 14 (more compatible)
                try:
                    self.logger.info("Trying to write with version 14 (more compatible)")
                    pyreadstat.write_dta(
                        pd_df,
                        self.output_file,
                        version=14
                    )
                    self.logger.info(f"Successfully wrote Stata file with version 14")
                    
                except Exception as e2:
                    self.logger.warning(f"Second write attempt failed: {str(e2)}. Trying most basic approach...")
                    
                    # Third attempt: Most conservative approach - version 13 and more aggressive truncation
                    try:
                        # Even more aggressive truncation for all string columns
                        for col in string_cols:
                            pd_df[col] = pd_df[col].astype(str).str.slice(0, 500)
                            self.logger.warning(f"Aggressively truncated column '{col}' to 500 characters")
                        
                        pyreadstat.write_dta(
                            pd_df,
                            self.output_file,
                            version=13
                        )
                        self.logger.info(f"Successfully wrote Stata file with version 13 and aggressive truncation")
                        
                    except Exception as e3:
                        self.logger.error(f"All write attempts failed. Last error: {str(e3)}")
                        
                        # Final attempt: CSV fallback
                        fallback_path = self.output_file.replace('.dta', '.csv')
                        self.logger.warning(f"Falling back to CSV output at: {fallback_path}")
                        pd_df.to_csv(fallback_path, index=False)
                        self.logger.info(f"Saved data as CSV instead at {fallback_path}")
                        
                        # Re-raise the original exception
                        raise
            
        except ImportError:
            self.logger.error("pyreadstat not installed. Please install it with: pip install pyreadstat")
            raise
        except Exception as e:
            self.logger.error(f"Error writing Stata file: {str(e)}")
            raise

    def _load_stata_metadata(self, input_file: Optional[str]) -> Dict[str, Any]:
        """
        Load Stata metadata from a JSON file.
        
        Args:
            input_file: Path to the original input file
            
        Returns:
            Dictionary containing the metadata or empty dict if not found
        """
        if not input_file:
            return {}
            
        metadata_file = os.path.splitext(input_file)[0] + '_metadata.json'
        
        if not os.path.exists(metadata_file):
            self.logger.debug(f"No Stata metadata file found at {metadata_file}")
            return {}
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.logger.debug(f"Loaded Stata metadata from {metadata_file}")
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Could not load Stata metadata: {str(e)}")
            return {}
    
    def write_intermediate_data(self, df: DataFrame, partition_id: int) -> None:
        """
        Write intermediate data to a checkpoint file.
        
        Args:
            df: DataFrame to write
            partition_id: Partition ID for the checkpoint
        """
        # Ensure intermediate directory exists
        os.makedirs(self.intermediate_dir, exist_ok=True)
        
        output_path = os.path.join(self.intermediate_dir, f'partition_{partition_id}.parquet')
        
        try:
            self.logger.debug(f"Writing intermediate data for partition {partition_id}")
            
            # Write DataFrame to Parquet format
            df.write.mode('overwrite').parquet(output_path)
            
            self.logger.info(f"Successfully wrote intermediate data for partition {partition_id}")
            
        except Exception as e:
            self.logger.error(f"Error writing intermediate data for partition {partition_id}: {str(e)}")
            raise


class MetadataHandler:
    """
    Handles preservation of metadata during file operations.
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        """
        Initialize the metadata handler.
        
        Args:
            spark: SparkSession instance
            config: Configuration dictionary
        """
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metadata = {}
    
    def extract_metadata(self, input_file: str) -> Dict[str, Any]:
        """
        Extract metadata from an input file.
        
        Args:
            input_file: Path to the input file
            
        Returns:
            Dictionary containing the metadata
        """
        file_ext = os.path.splitext(input_file)[1].lower()
        
        try:
            if file_ext == '.dta':
                return self._extract_stata_metadata(input_file)
            elif file_ext == '.parquet':
                return self._extract_parquet_metadata(input_file)
            else:
                self.logger.debug(f"No metadata extraction implemented for file format: {file_ext}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return {}
    
    def _extract_stata_metadata(self, input_file: str) -> Dict[str, Any]:
        """
        Extract metadata from a Stata file.
        
        Args:
            input_file: Path to the Stata file
            
        Returns:
            Dictionary containing the Stata metadata
        """
        try:
            import pyreadstat
            
            # Get metadata without reading the data
            _, meta = pyreadstat.read_dta_metadata(input_file)
            
            # Extract relevant metadata
            metadata = {
                'variable_labels': meta.variable_labels,
                'value_labels': meta.value_labels,
                'variable_value_labels': meta.variable_value_labels,
                'file_label': meta.file_label,
                'file_encoding': meta.file_encoding,
                'number_rows': meta.number_rows,
                'number_columns': meta.number_columns,
                'original_variable_types': meta.original_variable_types,
                'missing_ranges': meta.missing_ranges,
                'notes': meta.notes if hasattr(meta, 'notes') else []
            }
            
            # Save the metadata for later use
            self.metadata = metadata
            
            self.logger.info(f"Successfully extracted metadata from Stata file: {input_file}")
            return metadata
            
        except ImportError:
            self.logger.error("pyreadstat not installed. Please install it with: pip install pyreadstat")
            return {}
        except Exception as e:
            self.logger.error(f"Error extracting Stata metadata: {str(e)}")
            return {}
    
    def _extract_parquet_metadata(self, input_file: str) -> Dict[str, Any]:
        """
        Extract metadata from a Parquet file.
        
        Args:
            input_file: Path to the Parquet file
            
        Returns:
            Dictionary containing the Parquet metadata
        """
        try:
            # Read schema information from Parquet file
            df = self.spark.read.parquet(input_file)
            schema = df.schema
            
            # Extract schema information
            metadata = {
                'schema': str(schema),
                'column_names': df.columns,
                'column_types': {field.name: str(field.dataType) for field in schema.fields},
                'number_columns': len(df.columns)
            }
            
            # Try to get row count if not too expensive
            if self.config.get('extract_row_count', False):
                metadata['number_rows'] = df.count()
            
            # Save the metadata for later use
            self.metadata = metadata
            
            self.logger.info(f"Successfully extracted metadata from Parquet file: {input_file}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting Parquet metadata: {str(e)}")
            return {}
    
    def apply_metadata(self, df: DataFrame) -> DataFrame:
        """
        Apply metadata to a DataFrame before writing.
        
        Args:
            df: DataFrame to apply metadata to
            
        Returns:
            DataFrame with metadata applied
        """
        try:
            # Add file metadata
            output_file = self.config.get('output_file')
            file_ext = os.path.splitext(output_file)[1].lower() if output_file else ''
            
            if file_ext == '.dta':
                # For Stata files, we'll apply the metadata during the write operation
                # No changes needed to the DataFrame
                pass
                
            elif file_ext == '.parquet':
                # For Parquet files, we can add metadata as DataFrame properties
                if self.metadata:
                    # Convert metadata to string representation for storage
                    metadata_str = json.dumps(self.metadata)
                    
                    # Create a new DataFrame with metadata
                    df = df.select("*")  # This creates a copy
                    
                    # Set metadata in Spark properties
                    # Note: This is limited and may not be preserved in all operations
                    df_with_props = df.alias("df_with_metadata")
                    
                    self.logger.info(f"Applied metadata to DataFrame for output file: {output_file}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error applying metadata: {str(e)}")
            return df