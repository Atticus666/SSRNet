"""Criteo dataset processor."""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from collections import defaultdict

from .base import BaseDataProcessor
from .config import CriteoConfig
from tqdm import tqdm


class CriteoProcessor(BaseDataProcessor):

    def __init__(self, config: Optional[CriteoConfig] = None):
        if config is None:
            config = CriteoConfig()
        super().__init__(config)
        
        # Criteo dataset structure: 
        # Column 0: label
        # Columns 1-13: numerical features (13 features)
        # Columns 14-39: categorical features (26 features)
        self.num_numerical = config.num_numerical_features
        self.num_categorical = config.num_categorical_features
        self.total_fields = self.num_numerical + self.num_categorical
        self.min_category_count = 10  # Official threshold for rare categories

        
        self.num_buckets_per_feature = config.num_buckets_per_feature  
        self.use_numerical_discretization = config.use_numerical_discretization  
        self.numerical_quantiles = {}  
        if self.config.verbose > 0:
            print(f"CriteoProcessor initialized with:")
            print(f"  - Numerical discretization: {self.use_numerical_discretization}")
            print(f"  - Buckets per feature: {self.num_buckets_per_feature}")

    def load_raw_data(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        source_path = self.config.source_file
        
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        if self.config.verbose > 0:
            print(f"Loading Criteo data from {source_path}")
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
                sep='\t',
                header=None,
                dtype=str,
                na_values=['', 'nan', 'null', 'None'],
                keep_default_na=True,
                chunksize=chunk_size,
                nrows=nrows
            )
            
            chunk_iter = enumerate(chunk_reader)
            chunk_progress = tqdm(
                chunk_reader,
                desc="Loading chunks",
                unit="chunk",
                disable=not self.config.verbose
            )
            
            for chunk_idx, chunk_df in enumerate(chunk_progress):
                if len(chunk_df) == 0:
                    continue
                
                # Extract labels (column 0)
                chunk_labels = pd.to_numeric(chunk_df.iloc[:, 0], errors='coerce').fillna(0).astype(int).values
                
                # Extract features (columns 1-39)
                chunk_features = chunk_df.iloc[:, 1:40].copy()
                
                features_list.append(chunk_features)
                labels_list.append(chunk_labels)
                
                total_rows_processed += len(chunk_df)
                
                if nrows and total_rows_processed >= nrows:
                    if total_rows_processed > nrows:
                        excess = total_rows_processed - nrows
                        features_list[-1] = features_list[-1].iloc[:-excess]
                        labels_list[-1] = labels_list[-1][:-excess]
                    break
                    
        except Exception as e:
            raise RuntimeError(f"Error reading Criteo data file: {str(e)}")
        
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
        num_samples = len(data)
        
        if self.config.verbose > 0:
            print(f"Starting Criteo feature preprocessing for {num_samples:,} samples")
        
        # Build categorical vocabulary first
        self._build_criteo_vocabulary(data)
        
        feature_indices = np.zeros((num_samples, self.total_fields), dtype=np.int32)
        feature_values = np.zeros((num_samples, self.total_fields), dtype=np.float32)
        
        
        if self.use_numerical_discretization:
            self._process_numerical_features_discretized(data, feature_indices, feature_values)
        else:
            self._process_numerical_features(data, feature_indices, feature_values)

        # Process categorical features (columns 14-39 -> indices 13-38)  
        self._process_categorical_features(data, feature_indices, feature_values)
        
        if self.config.verbose > 0:
            print("Criteo feature preprocessing completed")
            
        return feature_indices, feature_values

    def _build_criteo_vocabulary(self, data: pd.DataFrame) -> None:
        """Build vocabulary for categorical features following official Criteo preprocessing."""
        if self.config.verbose > 0:
            print("Building Criteo categorical vocabulary...")
        
        # Initialize vocabulary with base indices for numerical features
        self.feature_vocab = {}
        self.feature_size_cumsum = []
        
        current_offset = 0

        if self.use_numerical_discretization:
            
            for i in range(self.num_numerical):
                col_data = data.iloc[:, i]
                numeric_values = pd.to_numeric(col_data, errors='coerce').dropna()
                
                if len(numeric_values) > 0:
                    
                    quantiles = np.linspace(0, 1, self.num_buckets_per_feature + 1)
                    bucket_edges = numeric_values.quantile(quantiles).values
                    
                    
                    bucket_edges = np.unique(bucket_edges)
                    actual_buckets = len(bucket_edges) - 1
                    
                    self.numerical_quantiles[i] = bucket_edges
                    
                    
                    vocab = {}
                    vocab['<MISSING>'] = 0  
                    for bucket_idx in range(actual_buckets):
                        vocab[f'bucket_{bucket_idx}'] = bucket_idx + 1
                    
                    self.feature_vocab[f'num_feat_{i}'] = vocab
                    feature_dim_size = actual_buckets + 1  # +1 for missing
                    
                    if self.config.verbose > 1:
                        print(f"  Numerical feature {i+1}: {actual_buckets} buckets, dim size: {feature_dim_size}")
                else:
                    
                    self.numerical_quantiles[i] = np.array([0, 1])
                    self.feature_vocab[f'num_feat_{i}'] = {'<MISSING>': 0}
                    feature_dim_size = 1
                
                self.feature_size_cumsum.append(current_offset + feature_dim_size)
                current_offset += feature_dim_size
        else:
            
            self.feature_size_cumsum = [self.num_numerical]
            current_offset = self.num_numerical
        
        # Process categorical features (columns 13-38 in data, which are original columns 14-39)
        for feat_idx in range(self.num_categorical):
            data_col_idx = self.num_numerical + feat_idx  # columns 13-38 in data
            original_col_idx = data_col_idx + 1  # original columns 14-39
            
            column_data = data.iloc[:, data_col_idx].astype(str)
            
            # Count occurrences of each category value
            value_counts = column_data.value_counts()
            
            # Create vocabulary: frequent values (>10 occurrences) get unique indices
            vocab = {}
            
            # Index 1 is reserved for rare/missing values (following official implementation)
            rare_index = 1
            vocab['<RARE>'] = rare_index
            
            # Start assigning indices from 2
            next_index = 2
            
            for value, count in value_counts.items():
                if value in ['', 'nan', 'None'] or pd.isna(value):
                    # Missing values map to rare index
                    vocab[str(value)] = rare_index
                elif count > self.min_category_count:
                    # Frequent values get unique indices
                    vocab[str(value)] = next_index
                    next_index += 1
                else:
                    # Rare values map to rare index
                    vocab[str(value)] = rare_index
            
            self.feature_vocab[original_col_idx] = vocab
            
            # Calculate cumulative feature sizes
            feature_dim_size = next_index
            self.feature_size_cumsum.append(self.feature_size_cumsum[-1] + feature_dim_size)
            
            if self.config.verbose > 1:
                frequent_count = sum(1 for count in value_counts.values() if count > self.min_category_count)
                print(f"  Feature {original_col_idx}: {frequent_count} frequent categories, "
                      f"dim size: {feature_dim_size}")
        
        self.total_feature_size = self.feature_size_cumsum[-1]
        
        if self.config.verbose > 0:
            print(f"Built vocabulary with {self.total_feature_size:,} total feature dimensions")
            print(f"Feature dimension sizes: {[self.feature_size_cumsum[0]] + [self.feature_size_cumsum[i+1] - self.feature_size_cumsum[i] for i in range(len(self.feature_size_cumsum)-1)]}")

    def _process_numerical_features(self, data: pd.DataFrame, 
                                   feature_indices: np.ndarray, 
                                   feature_values: np.ndarray) -> None:
        """Process numerical features (columns 1-13)."""
        
        for i in range(self.num_numerical):
            # Feature indices for numerical features: simply column index + 1
            feature_indices[:, i] = i + 1
            
            # Feature values: use original numerical values (with missing handling)
            col_data = data.iloc[:, i]
            numeric_values = pd.to_numeric(col_data, errors='coerce')
            
            # Handle missing values with 0 (following common practice)
            numeric_values = numeric_values.fillna(0.0)
            feature_values[:, i] = numeric_values.astype(np.float32)

    def _process_numerical_features_discretized(self, data: pd.DataFrame, 
                                            feature_indices: np.ndarray, 
                                            feature_values: np.ndarray) -> None:
        """Process numerical features with equal-frequency binning (optimized version)."""
        
        if self.config.verbose > 0:
            print("Processing numerical features with discretization...")
        
        
        from tqdm import tqdm
        
        feature_iter = range(self.num_numerical)
        if self.config.verbose > 0:
            feature_iter = tqdm(feature_iter, desc="Discretizing numerical features", unit="feature")
        
        for i in feature_iter:
            col_data = data.iloc[:, i]
            numeric_values = pd.to_numeric(col_data, errors='coerce')
            
            
            bucket_edges = self.numerical_quantiles[i]
            vocab = self.feature_vocab[f'num_feat_{i}']
            feature_offset = self.feature_size_cumsum[i-1] if i > 0 else 0
            
            
            actual_buckets = len(bucket_edges) - 1
            
            
            bucket_to_vocab = np.array([vocab[f'bucket_{j}'] for j in range(actual_buckets)])
            missing_vocab_value = vocab['<MISSING>']
            
            
            
            missing_mask = pd.isna(numeric_values)
            
            
            valid_values = numeric_values.fillna(0)  
            bucket_indices = np.searchsorted(bucket_edges[1:], valid_values, side='right')
            
            
            bucket_indices = np.clip(bucket_indices, 0, actual_buckets - 1)
            
            
            vocab_indices = bucket_to_vocab[bucket_indices]
            
            
            vocab_indices[missing_mask] = missing_vocab_value
            
            
            feature_indices[:, i] = vocab_indices + feature_offset
            feature_values[:, i] = 1.0  
        
        if self.config.verbose > 0:
            print("Numerical feature discretization completed")
            
            
    def _process_categorical_features(self, data: pd.DataFrame,
                                    feature_indices: np.ndarray,
                                    feature_values: np.ndarray) -> None:
        """Process categorical features (columns 14-39)."""
        
        for feat_idx in range(self.num_categorical):
            data_col_idx = self.num_numerical + feat_idx  # columns 13-38 in data
            original_col_idx = data_col_idx + 1  # original columns 14-39
            field_idx = self.num_numerical + feat_idx  # field indices 13-38
            
            column_data = data.iloc[:, data_col_idx].astype(str)
            vocab = self.feature_vocab[original_col_idx]
            
            # Map categorical values to vocabulary indices
            def map_to_vocab_index(value):
                if str(value) in vocab:
                    return vocab[str(value)]
                else:
                    return vocab['<RARE>']  # Default to rare index
            
            # Get vocabulary indices and add offset for this feature dimension
            vocab_indices = column_data.apply(map_to_vocab_index).values
            feature_offset = self.feature_size_cumsum[feat_idx]
            
            feature_indices[:, field_idx] = vocab_indices + feature_offset
            
            # Categorical features have value 1 (indicating presence)
            feature_values[:, field_idx] = 1.0

    def save_processed_data(self, 
                          feature_indices: np.ndarray,
                          feature_values: np.ndarray, 
                          labels: np.ndarray) -> None:
        """Save processed data in Criteo format (indices, values, labels)."""
        
        os.makedirs(self.config.data_path, exist_ok=True)
        
        indices_path = os.path.join(self.config.data_path, 'train_i.txt')
        values_path = os.path.join(self.config.data_path, 'train_x.txt')  
        labels_path = os.path.join(self.config.data_path, 'train_y.txt')
        
        if self.config.verbose > 0:
            print(f"Saving processed Criteo data to {self.config.data_path}")
        
        # Save feature indices
        with open(indices_path, 'w') as f:
            for i in range(len(feature_indices)):
                indices_line = ' '.join(map(str, feature_indices[i]))
                f.write(indices_line + '\n')
        
        # Save feature values  
        with open(values_path, 'w') as f:
            for i in range(len(feature_values)):
                values_line = ' '.join(map(str, feature_values[i]))
                f.write(values_line + '\n')
        
        # Save labels
        with open(labels_path, 'w') as f:
            for label in labels:
                f.write(f"{label}\n")
        
        # Save feature_size.npy
        feature_size_path = os.path.join(self.config.data_path, 'feature_size.npy')
        np.save(feature_size_path, np.array([self.total_feature_size]))
        
        if self.config.verbose > 0:
            print(f"Saved {len(labels):,} samples")
            print(f"  Feature indices: {indices_path}")
            print(f"  Feature values: {values_path}")
            print(f"  Labels: {labels_path}")
            print(f"  Feature size: {feature_size_path} (size={self.total_feature_size:,})")



    @classmethod
    def create_from_source(cls,
                          source_path: str,
                          output_path: str,
                          num_buckets_per_feature: int = 100,
                          use_numerical_discretization: bool = False,
                          nrows: Optional[int] = None) -> 'CriteoProcessor':
        """Create processor and process data from source file."""
        
        from .config import CriteoConfig
        config = CriteoConfig()
        config.source_file = source_path
        config.data_path = output_path
        config.num_buckets_per_feature = num_buckets_per_feature
        config.use_numerical_discretization = use_numerical_discretization
        
        processor = cls(config)
        
        # Load and process data
        features, labels = processor.load_raw_data(nrows=nrows)
        feature_indices, feature_values = processor.preprocess_features(features)
        
        # Save processed data
        processor.save_processed_data(feature_indices, feature_values, labels)
        
        return processor

    def process_dataset(self, nrows: Optional[int] = None, save: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Process the complete dataset pipeline."""
        features, labels = self.load_raw_data(nrows=nrows)
        feature_indices, feature_values = self.preprocess_features(features)
        
        if save:
            self.save_processed_data(feature_indices, feature_values, labels)
            
        return feature_indices, feature_values
    
    def get_feature_size(self) -> int:
        """Get total number of features after preprocessing."""
        if hasattr(self, 'total_feature_size'):
            return self.total_feature_size
        else:
            return 0


def preprocess_criteo_dataset(source_path: str,
                            output_path: str, 
                            num_buckets_per_feature: int = 100,
                            use_numerical_discretization: bool = False,
                            nrows: Optional[int] = None,
                            verbose: int = 1) -> int:
    """
    Convenience function to preprocess Criteo dataset with performance optimizations.
    
    Args:
        source_path: Path to raw Criteo data file (TSV format)
        output_path: Output directory for processed data
        nrows: Maximum number of rows to process (None for all)
        verbose: Verbosity level (0=quiet, 1=normal, 2=detailed)
        
    Returns:
        Total number of features after processing
    """
    # Create processor with optimizations
    processor = CriteoProcessor.create_from_source(
        source_path=source_path,
        output_path=output_path,
        num_buckets_per_feature=num_buckets_per_feature,
        use_numerical_discretization=use_numerical_discretization,
        nrows=nrows
    )
    
    if verbose > 0:
        print(f"Final feature size: {processor.get_feature_size():,}")
    
    return processor.get_feature_size()
