"""Avazu dataset processor."""

import os
import time
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from collections import defaultdict

from .base import BaseDataProcessor
from .config import AvazuConfig
from tqdm import tqdm

class AvazuProcessor(BaseDataProcessor):

    def __init__(self, config: Optional[AvazuConfig] = None):
        """
        Initialize Avazu processor.
        
        Args:
            config: Avazu-specific configuration (if None, uses default)
        """
        if config is None:
            config = AvazuConfig()
        super().__init__(config)
        
        self.label_index = config.label_index
        self.total_columns = config.total_columns
        self.field_size = config.field_size
        self.min_counts = config.min_category_counts
        self.categorical_columns = [i for i in range(self.total_columns) if i != self.label_index]
    
    def load_raw_data(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load raw Avazu data with highly optimized reading for large files.
        
        Uses chunk-based processing to handle very large datasets efficiently.
        
        Args:
            nrows: Maximum number of rows to load
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        source_path = self.config.source_file
        
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        if self.config.verbose > 0:
            print(f"Loading large Avazu data from {source_path}")
            if nrows:
                print(f"Loading first {nrows:,} rows")
            else:
                print("Loading entire dataset (this may take several minutes for large files)")
            print(f"Processed data will be saved to: {self.config.data_path}")

        chunk_size = 100000
        
        features_list = []
        labels_list = []
        total_rows_processed = 0
        
        if self.config.verbose > 0:
            print("Reading data in chunks for memory efficiency...")
        
        try:
            chunk_reader = pd.read_csv(
                source_path,
                dtype=str,
                na_values=['', 'nan', 'null', 'None'],
                keep_default_na=True,
                chunksize=chunk_size,
                nrows=nrows
            )
            
            chunk_iter = enumerate(chunk_reader)
            chunk_progress = tqdm(
                chunk_iter,
                desc="Reading chunks",
                unit="chunk",
                verbose=self.config.verbose
            )
            
            for chunk_idx, chunk in chunk_progress:
                if len(chunk) == 0:
                    continue
                    
                chunk_labels = pd.to_numeric(
                    chunk.iloc[:, self.label_index], 
                    errors='coerce'
                ).fillna(0).astype(int).values
                
                chunk_labels = (chunk_labels != 0).astype(int)
                labels_list.append(chunk_labels)
                
                feature_columns = [i for i in range(len(chunk.columns)) if i != self.label_index]
                chunk_features = chunk.iloc[:, feature_columns].copy()
                features_list.append(chunk_features)
                
                total_rows_processed += len(chunk)
                
                if nrows and total_rows_processed >= nrows:
                    if total_rows_processed > nrows:
                        excess = total_rows_processed - nrows
                        features_list[-1] = features_list[-1].iloc[:-excess]
                        labels_list[-1] = labels_list[-1][:-excess]
                    break
        
        except pd.errors.EmptyDataError:
            raise ValueError("No data could be loaded from source file")
        except Exception as e:
            raise RuntimeError(f"Error reading data file: {str(e)}")
        
        if not features_list:
            raise ValueError("No data loaded from source file")
        
        if self.config.verbose > 0:
            print(f"Combining {len(features_list)} chunks...")
        
        features = pd.concat(features_list, ignore_index=True)
        labels = np.concatenate(labels_list, axis=0)
        
        del features_list, labels_list
        
        if self.config.verbose > 0:
            print(f"Loaded {len(features):,} samples with {features.shape[1]} features")
            print(f"Label distribution: {np.bincount(labels)}")
            
        return features, labels
    
    def preprocess_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess Avazu features into index and value arrays using vectorized operations.
        
        Args:
            data: Raw feature dataframe
            
        Returns:
            Tuple of (feature_indices, feature_values)
        """
        num_samples = len(data)
        num_features = self.field_size
        
        if self.config.verbose > 0:
            print(f"Starting feature preprocessing for {num_samples:,} samples, {num_features} features")
        
        self._build_avazu_vocabulary_optimized(data)
        
        feature_indices = np.zeros((num_samples, num_features), dtype=np.int32)
        feature_values = np.ones((num_samples, num_features), dtype=np.float32)
        
        self._process_features_vectorized(data, feature_indices, feature_values)
        
        if self.config.verbose > 0:
            print("Feature preprocessing completed")
            
        return feature_indices, feature_values
    
    def _build_avazu_vocabulary_optimized(self, data: pd.DataFrame) -> None:
        if self.config.verbose > 0:
            print("Building Avazu vocabulary...")
            
        vocab = {}
        feature_index = 0
        
        for feature_idx in range(self.field_size):
            original_col = feature_idx if feature_idx < self.label_index else feature_idx + 1
            
            column_data = data.iloc[:, feature_idx]
            valid_mask = ~column_data.isna() & (column_data != '') & (column_data != 'nan')
            valid_values = column_data[valid_mask]
            
            if len(valid_values) > 0:
                value_counts = valid_values.astype(str).value_counts()
            else:
                value_counts = pd.Series(dtype='int64')
            
            min_count = self.min_counts[original_col] if original_col < len(self.min_counts) else 5
            
            vocab[original_col] = {}
            vocab[original_col]['<MISSING>'] = feature_index
            missing_index = feature_index
            feature_index += 1
            
            frequent_values = value_counts[value_counts >= min_count]
            for value in frequent_values.index:
                vocab[original_col][value] = feature_index
                feature_index += 1
            
            infrequent_values = value_counts[value_counts < min_count]
            for value in infrequent_values.index:
                vocab[original_col][value] = missing_index
            
            if self.config.verbose > 1:
                print(f"  Column {original_col}: {len(frequent_values)} frequent, {len(infrequent_values)} infrequent values")
        
        self.feature_vocab = vocab
        self.feature_size = feature_index
        
        if self.config.verbose > 0:
            print(f"Built Avazu vocabulary with {self.feature_size:,} total features")
    
    def _process_features_vectorized(self, 
                                   data: pd.DataFrame, 
                                   feature_indices: np.ndarray, 
                                   feature_values: np.ndarray) -> None:
        if self.config.verbose > 0:
            print("Processing features with vectorized operations...")
        
        for feature_idx in range(self.field_size):
            original_col = feature_idx if feature_idx < self.label_index else feature_idx + 1
            
            column_data = data.iloc[:, feature_idx]
            is_missing = column_data.isna() | (column_data == '') | (column_data == 'nan')
            
            def map_value_to_index(value):
                if pd.isna(value) or value == '' or value == 'nan':
                    return self._get_avazu_index(original_col, None)
                else:
                    return self._get_avazu_index(original_col, str(value))
            
            feature_indices[:, feature_idx] = column_data.apply(map_value_to_index).values
            feature_values[:, feature_idx] = (~is_missing).astype(float)
    
    def _get_avazu_index(self, col_idx: int, value: Optional[str]) -> int:
        """
        Get feature index for an Avazu categorical value.
        
        Args:
            col_idx: Original column index
            value: Categorical value (None for missing)
            
        Returns:
            Feature index
        """
        col_vocab = self.feature_vocab.get(col_idx, {})
        
        if value is None or value == '' or value == 'nan':
            return col_vocab.get('<MISSING>', 0)
        
        return col_vocab.get(str(value), col_vocab.get('<MISSING>', 0))
    
    
    @classmethod
    def create_from_source(cls,
                          source_path: str,
                          output_path: str,
                          nrows: Optional[int] = None) -> 'AvazuProcessor':
        """
        Create processor and process data from source file.
        
        Args:
            source_path: Path to source data file (CSV or text format)
            output_path: Output directory path
            nrows: Maximum number of rows to process
            
        Returns:
            Configured processor instance
        """
        config = AvazuConfig(data_path=output_path)
        config.source_file = source_path
        
        
        processor = cls(config)
        processor.process_dataset(nrows=nrows, save=True)
        
        return processor

def preprocess_avazu_dataset(source_path: str,
                            output_path: str, 
                            nrows: Optional[int] = None,
                            verbose: int = 1) -> int:
    """
    Convenience function to preprocess Avazu dataset with performance optimizations.
    
    Args:
        source_path: Path to raw Avazu data file (CSV or text format)
        output_path: Output directory for processed data
        nrows: Maximum number of rows to process (None for all)
        verbose: Verbosity level (0=quiet, 1=normal, 2=detailed)
        
    Returns:
        Total number of features after processing
    """
    # Create processor with optimizations
    processor = AvazuProcessor.create_from_source(
        source_path=source_path,
        output_path=output_path,
        nrows=nrows
    )
    
    if verbose > 0:
        print(f"Final feature size: {processor.get_feature_size():,}")
    
    return processor.get_feature_size()

