"""
Base data processor class with common functionality.

This module provides the base class and common utilities for data processing
across different CTR datasets.
"""

import os
import numpy as np
import pandas as pd
import math
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold

from .config import DataConfig


class BaseDataProcessor(ABC):
    """
    Abstract base class for dataset processors.
    
    Provides common functionality for data loading, preprocessing,
    and saving across different CTR datasets.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize base processor.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        self.feature_vocab = {}  # Feature vocabulary for categorical features
        self.feature_size = 0
        
    @abstractmethod
    def load_raw_data(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load raw data from source file.
        
        Args:
            nrows: Maximum number of rows to load
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        pass
    
    @abstractmethod
    def preprocess_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features into index and value arrays.
        
        Args:
            data: Raw feature dataframe
            
        Returns:
            Tuple of (feature_indices, feature_values)
        """
        pass
    
    def build_vocabulary(self, data: pd.DataFrame, categorical_columns: List[int]) -> Dict[int, Dict[str, int]]:
        """
        Build vocabulary for categorical features.
        
        Args:
            data: Input dataframe
            categorical_columns: List of categorical column indices
            
        Returns:
            Dictionary mapping column index to {value: index} dictionary
        """
        vocab = {}
        feature_index = 1  # Start from 1, reserve 0 for unknown/padding
        
        # Reserve space for numerical features if any
        num_numerical = getattr(self.config, 'num_numerical_features', 0)
        feature_index += num_numerical
        
        for col_idx in categorical_columns:
            vocab[col_idx] = {}
            value_counts = data.iloc[:, col_idx].value_counts()
            
            # Get minimum count threshold
            min_count = getattr(self.config, 'min_category_count', 1)
            if hasattr(self.config, 'min_category_counts'):
                min_count = self.config.min_category_counts[col_idx]
            
            for value, count in value_counts.items():
                if pd.isna(value) or value == '':
                    continue
                    
                if count >= min_count:
                    vocab[col_idx][str(value)] = feature_index
                    feature_index += 1
                else:
                    # Assign to "other" category (index 1 for this column)
                    vocab[col_idx][str(value)] = 1
                    
        self.feature_vocab = vocab
        self.feature_size = feature_index
        return vocab
    
    def encode_categorical_features(self, 
                                  data: pd.DataFrame, 
                                  categorical_columns: List[int]) -> np.ndarray:
        """
        Encode categorical features using vocabulary.
        
        Args:
            data: Input dataframe
            categorical_columns: List of categorical column indices
            
        Returns:
            Encoded feature indices array
        """
        encoded = np.zeros((len(data), len(categorical_columns)), dtype=np.int32)
        
        for i, col_idx in enumerate(categorical_columns):
            col_vocab = self.feature_vocab.get(col_idx, {})
            
            for j, value in enumerate(data.iloc[:, col_idx]):
                if pd.isna(value) or value == '':
                    encoded[j, i] = 0  # Unknown/missing value
                else:
                    encoded[j, i] = col_vocab.get(str(value), 1)  # Default to "other"
                    
        return encoded
    
    def scale_numerical_features(self, 
                                data: np.ndarray, 
                                method: str = 'log',
                                columns: Optional[List[int]] = None) -> np.ndarray:
        """
        Scale numerical features.
        
        Args:
            data: Input data array
            method: Scaling method ('log', 'minmax', 'standard')
            columns: Column indices to scale (if None, scale all)
            
        Returns:
            Scaled data array
        """
        scaled_data = data.copy()
        
        if columns is None:
            columns = list(range(data.shape[1]))
        
        if method == 'log':
            for col in columns:
                scaled_data[:, col] = np.array([
                    int(math.log(float(x))**2) if x > 2 else x 
                    for x in data[:, col]
                ])
        elif method == 'minmax':
            scaler = MinMaxScaler()
            scaled_data[:, columns] = scaler.fit_transform(data[:, columns])
        elif method == 'standard':
            scaler = StandardScaler()
            scaled_data[:, columns] = scaler.fit_transform(data[:, columns])
            # Add 1 to avoid negative values
            scaled_data[:, columns] += 1
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
            
        return scaled_data
    
    def save_processed_data(self, 
                           feature_indices: np.ndarray,
                           feature_values: np.ndarray, 
                           labels: np.ndarray) -> None:
        """
        Save processed data to files.
        
        Args:
            feature_indices: Feature index array
            feature_values: Feature value array  
            labels: Label array
        """
        # Save text files
        np.savetxt(self.config.train_i_path, feature_indices, fmt='%d', delimiter=' ')
        np.savetxt(self.config.train_x_path, feature_values, fmt='%s', delimiter=' ')
        np.savetxt(self.config.train_y_path, labels, fmt='%d')
        
        # Save feature size
        feature_size_path = os.path.join(self.config.data_path, self.config.feature_size_file)
        np.save(feature_size_path, np.array([self.feature_size]))
        
        if self.config.verbose > 0:
            print(f"Saved processed data to {self.config.data_path}")
            print(f"Feature size: {self.feature_size}")
            print(f"Data shape: {feature_indices.shape}")
    
    def load_processed_data(self, nrows: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load previously processed data.
        
        Args:
            nrows: Maximum number of rows to load
            
        Returns:
            Tuple of (feature_indices, feature_values, labels)
        """
        # Load from text files for compatibility
        feature_indices = np.loadtxt(self.config.train_i_path, dtype=np.int32, max_rows=nrows)
        feature_values = np.loadtxt(self.config.train_x_path, dtype=np.float32, max_rows=nrows)
        labels = np.loadtxt(self.config.train_y_path, dtype=np.int32, max_rows=nrows)
        
        return feature_indices, feature_values, labels
    
    def get_feature_size(self) -> int:
        """Get total feature size."""
        feature_size_path = os.path.join(self.config.data_path, self.config.feature_size_file)
        if os.path.exists(feature_size_path):
            return int(np.load(feature_size_path)[0])
        return self.feature_size
    
    def process_dataset(self, nrows: Optional[int] = None, save: bool = True) -> None:
        """
        Complete data processing pipeline.
        
        Args:
            nrows: Maximum number of rows to process
            save: Whether to save processed data
        """
        if self.config.verbose > 0:
            print("Loading raw data...")
        
        data, labels = self.load_raw_data(nrows)
        
        if self.config.verbose > 0:
            print(f"Loaded {len(data)} samples")
        from tqdm import tqdm
        # Simplified logging
        if self.config.verbose > 0: print("Preprocessing features...")
        
        feature_indices, feature_values = self.preprocess_features(data)
        
        if save:
            if self.config.verbose > 0:
                print("Saving processed data...")
            self.save_processed_data(feature_indices, feature_values, labels)
        
        if self.config.verbose > 0:
            if self.config.verbose > 0: print("Data processing completed!")


class DataScaler:
    """Utility class for scaling numerical features."""
    
    @staticmethod
    def log_scale(x: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Apply logarithmic scaling as used in the original implementation."""
        if isinstance(x, np.ndarray):
            return np.array([int(math.log(float(val))**2) if val > 2 else val for val in x])
        else:
            return int(math.log(float(x))**2) if x > 2 else x
    
    @staticmethod
    def scale_data_parts(config: DataConfig, 
                        numerical_columns: List[int],
                        scale_method: str = 'log') -> None:
        """
        Scale numerical features in data parts.
        
        Args:
            config: Data configuration
            numerical_columns: List of numerical column indices
            scale_method: Scaling method to use
        """
        for i in range(1, config.num_splits + 1):
            if config.verbose > 0:
                print(f'Scaling part {i}')
                
            # Load original data
            data_path = config.get_part_path(i, config.train_x_npy)
            if not os.path.exists(data_path):
                continue
                
            data = np.load(data_path)
            
            # Scale numerical columns
            if scale_method == 'log':
                for j, row in enumerate(data):
                    if j % 100000 == 0 and config.verbose > 0:
                        print(f"  Processing row {j}")
                    for col in numerical_columns:
                        data[j, col] = DataScaler.log_scale(row[col])
            
            # Save scaled data
            scaled_path = config.get_part_path(i, config.train_x2_npy)
            np.save(scaled_path, data)
        
        if config.verbose > 0:
            print("Scaling completed!")
