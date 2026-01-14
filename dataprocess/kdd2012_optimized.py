"""KDD2012 dataset processor."""

import os
import time
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Iterator

from .base import BaseDataProcessor
from .config import KDD2012Config
from tqdm import tqdm

class KDD2012Processor(BaseDataProcessor):
    
    def __init__(self, config: Optional[KDD2012Config] = None):
        if config is None:
            config = KDD2012Config()
        super().__init__(config)
        
        self.field_size = config.field_size
        self.chunk_size = getattr(config, 'chunk_size', 50000)
    
    def load_raw_data(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        source_path = self.config.source_file
        
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        if self.config.verbose > 0:
            print(f"Loading KDD2012 data from {source_path}")
            if nrows:
                print(f"Loading first {nrows:,} rows")
            else:
                print("Loading entire dataset (this may take several minutes for large files)")
            print(f"Processed data will be saved to: {self.config.data_path}")
            print(f"Processed data will be saved to: {self.config.data_path}")
        
        try:
            if self.config.verbose > 0:
                print("Reading data in chunks for memory efficiency...")
            
            features_list = []
            labels_list = []
            
            chunk_progress = tqdm(
                self._read_data_chunks(source_path, nrows),
                desc="Loading chunks",
                unit="chunk",
                verbose=self.config.verbose
            )
            
            total_rows_processed = 0
            
            for chunk_idx, chunk_df in enumerate(chunk_progress):
                try:
                    if len(chunk_df) == 0:
                        continue
                    
                    chunk_labels = chunk_df.iloc[:, 0]
                    
                    if chunk_labels.dtype == 'object':
                        chunk_labels = pd.to_numeric(chunk_labels, errors='coerce').fillna(0)
                    chunk_labels = (chunk_labels > 0).astype('int8').values
                    
                    chunk_features = chunk_df.iloc[:, 1:]
                    
                    for col_idx in range(min(chunk_features.shape[1], self.field_size)):
                        col_data = chunk_features.iloc[:, col_idx]
                        if col_data.dtype == 'object':
                            chunk_features.iloc[:, col_idx] = col_data.astype('category')
                    
                    features_list.append(chunk_features)
                    labels_list.append(chunk_labels)
                    
                    total_rows_processed += len(chunk_df)
                    
                except Exception as e:
                    if self.config.verbose > 0:
                        print(f"Warning: Error processing chunk {chunk_idx + 1}: {e}")
                    continue
            
            if not features_list:
                raise RuntimeError("No valid data chunks were processed")
            
            if self.config.verbose > 0:
                print(f"Combining {len(features_list)} chunks...")
            
            features = pd.concat(features_list, ignore_index=True)
            labels = np.concatenate(labels_list)
            
            del features_list, labels_list
            
            self._timing_info['data_loading'] = time.time() - start_time
            
            if self.config.verbose > 0: print(f"Loaded {len(features)} samples with {features.shape[1]} features")
            if self.config.verbose > 0: print(f"Label distribution: {np.bincount(labels)}")
            
            return features, labels
            
        except Exception as e:
            raise RuntimeError(f"Error reading KDD2012 data file: {str(e)}")
    
    def _count_lines(self, file_path: str) -> int:
        """Fast line counting for progress tracking."""
        try:
            with open(file_path, 'rb') as f:
                lines = sum(1 for _ in f)
            return max(0, lines - 1)  # Subtract header if exists
        except Exception:
            return 0  # Fallback if counting fails
    
    def _read_data_chunks(self, file_path: str, nrows: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """
        Generator for reading KDD2012 data in chunks.
        
        Args:
            file_path: Path to data file
            nrows: Maximum number of rows to read
            
        Yields:
            DataFrame chunks
        """
        try:
            # KDD2012 is typically tab-separated
            chunk_reader = pd.read_csv(
                file_path,
                sep='\t',
                header=None,
                dtype=str,
                na_values=['', 'NULL', 'null', 'NA'],
                chunksize=self.chunk_size,
                nrows=nrows,
                engine='c',  # Use C engine for speed
                low_memory=False
            )
            
            for chunk in chunk_reader:
                if len(chunk) > 0:
                    yield chunk
                    
        except Exception as e:
            # Try comma-separated as fallback
            try:
                chunk_reader = pd.read_csv(
                    file_path,
                    sep=',',
                    header=None,
                    dtype=str,
                    na_values=['', 'NULL', 'null', 'NA'],
                    chunksize=self.chunk_size,
                    nrows=nrows,
                    engine='c',
                    low_memory=False
                )
                
                for chunk in chunk_reader:
                    if len(chunk) > 0:
                        yield chunk
                        
            except Exception as e2:
                raise RuntimeError(f"Failed to read KDD2012 data chunks with both tab and comma separators: {str(e)} | {str(e2)}")
    
    def preprocess_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memory-optimized feature preprocessing with vectorized operations.
        
        Args:
            data: Raw feature dataframe
            
        Returns:
            Tuple of (feature_indices, feature_values)
        """
        start_time = time.time()
        num_samples = len(data)
        num_features = min(data.shape[1], self.field_size)
        
        # Simplified logging
        if self.config.verbose > 0: print(f"Starting KDD2012 feature preprocessing for {num_samples:,} samples, {num_features} features")
        
        # Build vocabulary for categorical features
        categorical_columns = list(range(num_features))
        self._build_kdd_vocabulary_optimized(data, categorical_columns)
        
        # Initialize output arrays
        feature_indices = np.zeros((num_samples, num_features), dtype=np.int32)
        feature_values = np.zeros((num_samples, num_features), dtype=np.float32)
        
        try:
            # Process categorical features (vectorized)
            # Processing KDD2012 categorical features
            self._process_categorical_features_vectorized(data, categorical_columns, feature_indices, feature_values)
            
            self._timing_info['feature_preprocessing'] = time.time() - start_time
            
            if self.config.verbose > 0: print(f"KDD2012 feature preprocessing completed in {self._timing_info['feature_preprocessing']:.2f}s")
            
            return feature_indices, feature_values
            
        except Exception as e:
            raise RuntimeError(f"KDD2012 feature preprocessing failed: {str(e)}")
    
    def _build_kdd_vocabulary_optimized(self, data: pd.DataFrame, categorical_columns: List[int]) -> None:
        """
        Build vocabulary for KDD2012 categorical features with optimized counting.
        
        Args:
            data: Input feature dataframe
            categorical_columns: List of categorical column indices
        """
        # Simplified logging
        if self.config.verbose > 0: print("Building KDD2012 vocabulary...")
        
        vocab = {}
        feature_index = 1  # Start from 1
        min_count = getattr(self.config, 'min_category_count', 1)
        
        # Process each column with progress tracking
        column_progress = tqdm(
            categorical_columns,
            desc="Building vocabulary",
            unit="column",
            verbose=self.config.verbose
        )
        
        for col_idx in column_progress:
            if col_idx >= data.shape[1]:
                continue
                
            col_data = data.iloc[:, col_idx].fillna('')
            
            # Count values efficiently using pandas value_counts
            value_counts = col_data.value_counts()
            
            # Build vocabulary for this column
            col_vocab = {}
            
            # Reserve index for missing values
            col_vocab['<MISSING>'] = feature_index
            feature_index += 1
            
            # Add frequent values
            valid_values = 0
            for value, count in value_counts.items():
                if value != '' and count >= min_count:
                    col_vocab[str(value)] = feature_index
                    feature_index += 1
                    valid_values += 1
                else:
                    # Map infrequent values to missing
                    col_vocab[str(value)] = col_vocab['<MISSING>']
            
            vocab[col_idx] = col_vocab
            
            # Optional: limit vocabulary size per column
            max_vocab_per_column = getattr(self.config, 'max_vocab_per_column', 10000)
            if valid_values > max_vocab_per_column:
                if self.config.verbose > 0: print(f"Warning: Column {col_idx} has {valid_values} unique values (limit: {max_vocab_per_column})")
        
        self.feature_vocab = vocab
        self.feature_size = feature_index
        
        if self.config.verbose > 0: print(f"Built KDD2012 vocabulary with {self.feature_size} total features")
    
    def _process_categorical_features_vectorized(self, 
                                               data: pd.DataFrame,
                                               categorical_columns: List[int],
                                               feature_indices: np.ndarray,
                                               feature_values: np.ndarray) -> None:
        """Vectorized processing of KDD2012 categorical features."""
        
        for i, col_idx in enumerate(categorical_columns):
            if col_idx >= data.shape[1] or i >= feature_indices.shape[1]:
                continue
                
            col_vocab = self.feature_vocab.get(col_idx, {})
            missing_idx = col_vocab.get('<MISSING>', 0)
            
            # Get column data
            col_data = data.iloc[:, col_idx].fillna('')
            
            # Vectorized mapping with error handling
            def map_value(val):
                return col_vocab.get(str(val), missing_idx)
            
            # Apply mapping vectorized
            try:
                mapped_indices = col_data.map(map_value).values
                feature_indices[:, i] = mapped_indices
                
                # Set values (1.0 for valid, 0.0 for missing)
                feature_values[:, i] = (col_data != '').astype(np.float32)
                
            except Exception as e:
                # Fallback to iterative processing if vectorized fails
                for row_idx in range(len(col_data)):
                    val = col_data.iloc[row_idx]
                    if val == '':
                        feature_indices[row_idx, i] = missing_idx
                        feature_values[row_idx, i] = 0.0
                    else:
                        feature_indices[row_idx, i] = col_vocab.get(str(val), missing_idx)
                        feature_values[row_idx, i] = 1.0
    
    def scale_features_in_parts(self) -> None:
        """
        Apply scaling to features in data parts if needed.
        
        KDD2012 typically doesn't require the same logarithmic scaling as Criteo,
        but this method provides flexibility for custom scaling if needed.
        """
        # Simplified logging
        if self.config.verbose > 0: print("KDD2012 features typically don't require scaling")
        
        # For KDD2012, we typically don't apply scaling, but this can be customized
        # if the specific dataset variant requires it
        
        # Optional: Apply custom scaling if needed
        custom_scaling = getattr(self.config, 'apply_scaling', False)
        if custom_scaling:
            if self.config.verbose > 0: print("Applying custom scaling for KDD2012 features...")
            # Custom scaling logic can be added here if needed
            pass
    
    def get_performance_info(self) -> dict:
        """Get performance timing information."""
        total_time = sum(self._timing_info.values())
        
        info = self._timing_info.copy()
        info['total_time'] = total_time
        
        if 'data_loading' in info and 'feature_preprocessing' in info:
            info['processing_efficiency'] = info['feature_preprocessing'] / info['data_loading']
        
        return info
    
    def print_performance_summary(self) -> None:
        """Print performance summary."""
        if not self._timing_info:
            return
            
        # Simplified logging
        total_time = sum(self._timing_info.values())
        
        if self.config.verbose > 0: print(f"=== KDD2012 Processing Performance ===")
        for stage, duration in self._timing_info.items():
            if self.config.verbose > 0: print(f"{stage.replace('_', ' ').title()}: {duration:.2f}s")
        if self.config.verbose > 0: print(f"Total processing time: {total_time:.2f}s")
        
        if hasattr(self, 'feature_size'):
            if self.config.verbose > 0: print(f"Vocabulary size: {self.feature_size:,} features")
    
    @classmethod
    def create_from_source(cls,
                          source_path: str,
                          output_path: str,
                          nrows: Optional[int] = None) -> 'KDD2012Processor':
        """
        Create optimized processor and process data from source file.
        
        Args:
            source_path: Path to source data file
            output_path: Output directory path
            nrows: Maximum number of rows to process
            
        Returns:
            Configured processor instance
        """
        config = KDD2012Config(data_path=output_path)
        config.source_file = os.path.basename(source_path)
        
        # Copy source file to output directory if different
        source_dir = os.path.dirname(source_path)
        if source_dir != output_path:
            import shutil
            os.makedirs(output_path, exist_ok=True)
            shutil.copy2(source_path, os.path.join(output_path, config.source_file))
        
        processor = cls(config)
        processor.process_dataset(nrows=nrows, save=True)
        processor.print_performance_summary()
        
        return processor


def preprocess_kdd2012_dataset(source_path: str, 
                              output_path: str,
                              nrows: Optional[int] = None,
                              verbose: int = 1) -> KDD2012Processor:
    """
    Convenient function to preprocess KDD2012 dataset with optimization.
    
    Args:
        source_path: Path to source data file
        output_path: Output directory path
        nrows: Maximum number of rows to process
        verbose: Verbosity level
        
    Returns:
        Configured processor instance
    """
    config = KDD2012Config(data_path=output_path, verbose=verbose)
    config.source_file = os.path.basename(source_path)
    
    # Copy source file if needed
    source_dir = os.path.dirname(source_path)
    if source_dir != output_path:
        import shutil
        os.makedirs(output_path, exist_ok=True)
        shutil.copy2(source_path, os.path.join(output_path, config.source_file))
    
    processor = KDD2012Processor(config)
    
    start_time = time.time()
    processor.process_dataset(nrows=nrows, save=True)
    processing_time = time.time() - start_time
    
    # Performance summary
    processor.print_performance_summary()
    
    # Simplified logging
    if nrows:
        time_per_sample = processing_time / nrows * 1000
        if verbose > 0: 
            print(f"Processed {nrows:,} rows in {processing_time:.2f}s ({time_per_sample:.3f}ms per sample)")
    
    return processor
