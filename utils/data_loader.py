"""
Data loading utilities for model.

This module provides efficient data loading functionality for CTR datasets
with support for batching, prefetching, and multi-fold data management.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List, Dict, Any
import logging

from dataprocess.config import DataConfig  


class DataLoader:
    """
    Data loader for model with optimized TF2.x data pipeline.
    
    Provides efficient loading of CTR datasets with proper batching,
    shuffling, and prefetching for optimal training performance.
    """
    
    def __init__(self,
                 config: DataConfig,
                 batch_size: int = 1024,
                 buffer_size: int = 10000,
                 prefetch_size: int = tf.data.AUTOTUNE,
                 shuffle: bool = True):
        """
        Initialize data loader.
        
        Args:
            config: Data configuration object
            batch_size: Batch size for training
            buffer_size: Buffer size for shuffling
            prefetch_size: Prefetch size for data pipeline
            shuffle: Whether to shuffle training data
        """
        self.config = config
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.prefetch_size = prefetch_size
        self.shuffle = shuffle
        
        # Cache for loaded datasets
        self._dataset_cache = {}
        self._feature_size = None
        
    def get_feature_size(self) -> int:
        """
        Get total feature size from saved feature_size.npy.
        
        Returns:
            Total number of features
        """
        if self._feature_size is None:
            feature_size_path = os.path.join(self.config.data_path, self.config.feature_size_file)
            
            if os.path.exists(feature_size_path):
                self._feature_size = int(np.load(feature_size_path)[0])
            else:
                raise FileNotFoundError(f"Feature size file not found: {feature_size_path}")
        print(f"feature_size: {self._feature_size}")         
        return self._feature_size
    
    def get_field_size(self) -> int:
        """
        Get field size (number of feature fields) from config.
        
        Returns:
            Number of feature fields
        """
        if not hasattr(self.config, 'field_size'):
            raise AttributeError(f"Config {type(self.config).__name__} does not have 'field_size' attribute")
        
        field_size = self.config.field_size
        if self.config.verbose > 0:
            print(f"field_size: {field_size}")
        return field_size
    
    def load_fold_dataset(self, 
                         fold_id: int,
                         file_names: Tuple[str, str, str],
                         cache_key: Optional[str] = None) -> tf.data.Dataset:
        """
        Load dataset for a specific fold.
        
        Args:
            fold_id: Fold identifier (1-based)
            file_names: Tuple of (indices_file, values_file, labels_file)
            cache_key: Optional cache key for dataset caching
            
        Returns:
            TensorFlow dataset ready for training/evaluation
        """
        if cache_key and cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]
        
        # Load data files
        indices_path = self.config.get_part_path(fold_id, file_names[0])
        values_path = self.config.get_part_path(fold_id, file_names[1])
        labels_path = self.config.get_part_path(fold_id, file_names[2])
        
        # Check if files exist
        for path in [indices_path, values_path, labels_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
        
        # Load numpy arrays
        feature_indices = np.load(indices_path).astype(np.int32)
        feature_values = np.load(values_path).astype(np.float32)
        labels = np.load(labels_path).astype(np.float32).reshape(-1, 1)
        
        if self.config.verbose > 1:
            print(f"Loaded fold {fold_id}: {len(feature_indices)} samples")
            print(f"  Indices shape: {feature_indices.shape}")
            print(f"  Values shape: {feature_values.shape}")
            print(f"  Labels shape: {labels.shape}")
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'feat_index': feature_indices,
            'feat_value': feature_values,
            'labels': labels
        })
        
        # Apply transformations
        if self.shuffle and fold_id >= 3:  # Only shuffle training folds
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
            
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(self.prefetch_size)
        
        # Cache dataset if key provided
        if cache_key:
            self._dataset_cache[cache_key] = dataset
            
        return dataset
    
    def create_dataset_from_arrays(self,
                                  feature_indices: np.ndarray,
                                  feature_values: np.ndarray,
                                  labels: np.ndarray,
                                  shuffle: Optional[bool] = None) -> tf.data.Dataset:
        """
        Create dataset from numpy arrays.
        
        Args:
            feature_indices: Feature index array
            feature_values: Feature value array
            labels: Label array
            shuffle: Whether to shuffle (uses default if None)
            
        Returns:
            TensorFlow dataset
        """
        if shuffle is None:
            shuffle = self.shuffle
            
        # Ensure proper shapes and types
        feature_indices = feature_indices.astype(np.int32)
        feature_values = feature_values.astype(np.float32)
        labels = labels.astype(np.float32).reshape(-1, 1)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'feat_index': feature_indices,
            'feat_value': feature_values,
            'labels': labels
        })
        
        # Apply transformations
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
            
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(self.prefetch_size)
        
        return dataset
    
    def load_training_datasets(self, 
                              file_names: Tuple[str, str, str],
                              start_fold: int = 3,
                              end_fold: int = 10) -> List[tf.data.Dataset]:
        """
        Load multiple training datasets.
        
        Args:
            file_names: Tuple of file names
            start_fold: Starting fold ID (inclusive)
            end_fold: Ending fold ID (inclusive)
            
        Returns:
            List of training datasets
        """
        datasets = []
        
        for fold_id in range(start_fold, end_fold + 1):
            try:
                cache_key = f"train_{fold_id}"
                dataset = self.load_fold_dataset(fold_id, file_names, cache_key)
                datasets.append(dataset)
            except FileNotFoundError:
                if self.config.verbose > 0:
                    print(f"Warning: Fold {fold_id} not found, skipping...")
                continue
        
        return datasets
    
    def get_validation_dataset(self, 
                              file_names: Tuple[str, str, str],
                              fold_id: int = 2) -> tf.data.Dataset:
        """
        Get validation dataset.
        
        Args:
            file_names: Tuple of file names
            fold_id: Validation fold ID (default: 2)
            
        Returns:
            Validation dataset
        """
        return self.load_fold_dataset(fold_id, file_names, f"val_{fold_id}")
    
    def get_test_dataset(self,
                        file_names: Tuple[str, str, str],
                        fold_id: int = 1) -> tf.data.Dataset:
        """
        Get test dataset.
        
        Args:
            file_names: Tuple of file names  
            fold_id: Test fold ID (default: 1)
            
        Returns:
            Test dataset
        """
        return self.load_fold_dataset(fold_id, file_names, f"test_{fold_id}")
    
    def clear_cache(self) -> None:
        """Clear dataset cache to free memory."""
        self._dataset_cache.clear()
        
    def get_dataset_info(self, dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset: TensorFlow dataset
            
        Returns:
            Dictionary with dataset information
        """
        # Get a sample batch to inspect structure
        sample_batch = next(iter(dataset.take(1)))
        
        info = {
            'batch_size': self.batch_size,
            'feature_indices_shape': sample_batch['feat_index'].shape,
            'feature_values_shape': sample_batch['feat_value'].shape,
            'labels_shape': sample_batch['labels'].shape,
            'feature_indices_dtype': sample_batch['feat_index'].dtype,
            'feature_values_dtype': sample_batch['feat_value'].dtype,
            'labels_dtype': sample_batch['labels'].dtype
        }
        
        return info


# class DatasetIterator:
#     """
#     Custom iterator for handling multiple data folds during training.
    
#     This iterator can cycle through multiple training data folds,
#     which is useful for large datasets split into multiple parts.
#     """
    
#     def __init__(self, data_loader: DataLoader, file_names: Tuple[str, str, str]):
#         """
#         Initialize dataset iterator.
        
#         Args:
#             data_loader: DataLoader instance
#             file_names: Tuple of file names
#         """
#         self.data_loader = data_loader
#         self.file_names = file_names
#         self.current_fold = 2  # Start from training fold
#         self.max_fold = 10
        
#     def __iter__(self):
#         """Return iterator."""
#         return self
        
#     def __next__(self) -> tf.data.Dataset:
#         """Get next dataset."""
#         if self.current_fold > self.max_fold:
#             raise StopIteration
            
#         try:
#             dataset = self.data_loader.load_fold_dataset(self.current_fold, self.file_names)
#             self.current_fold += 1
#             return dataset
#         except FileNotFoundError:
#             self.current_fold += 1
#             if self.current_fold > self.max_fold:
#                 raise StopIteration
#             return self.__next__()
    
#     def reset(self):
#         """Reset iterator to beginning."""
#         self.current_fold = 2


# def create_combined_dataset(datasets: List[tf.data.Dataset], 
#                           weights: Optional[List[float]] = None) -> tf.data.Dataset:
#     """
#     Combine multiple datasets into one.
    
#     Args:
#         datasets: List of datasets to combine
#         weights: Optional weights for sampling from each dataset
        
#     Returns:
#         Combined dataset
#     """
#     if not datasets:
#         raise ValueError("No datasets provided")
        
#     if len(datasets) == 1:
#         return datasets[0]
    
#     if weights is None:
#         # Interleave datasets equally
#         combined = tf.data.Dataset.from_tensor_slices(datasets)
#         combined = combined.interleave(
#             lambda x: x,
#             cycle_length=len(datasets),
#             block_length=1,
#             num_parallel_calls=tf.data.AUTOTUNE
#         )
#     else:
#         # Sample datasets according to weights
#         combined = tf.data.experimental.sample_from_datasets(datasets, weights)
    
#     return combined
