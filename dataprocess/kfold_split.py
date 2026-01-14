"""
K-fold data splitting utilities.

This module provides optimized stratified k-fold splitting functionality
for CTR datasets, maintaining the original implementation's logic.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple, Optional, Union

from .config import DataConfig
from tqdm import tqdm


class StratifiedDataSplitter:
    """
    Stratified k-fold data splitter for CTR datasets.
    
    This class handles splitting processed data into k folds while maintaining
    label distribution balance across splits.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize data splitter.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        self.fold_indices = None
        
    def load_processed_data(self, nrows: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load processed data from files.
        
        Args:
            nrows: Maximum number of rows to load
            
        Returns:
            Tuple of (features, labels)
        """
        # Simplified logging
        if self.config.verbose > 0: print("Loading processed data for splitting...")
            
        # Load features and labels
        features = pd.read_csv(
            self.config.train_x_path,
            header=None,
            sep=' ',
            nrows=nrows,
            dtype=np.float32
        ).values
        
        labels = pd.read_csv(
            self.config.train_y_path,
            header=None,
            sep=' ',
            nrows=nrows,
            dtype=np.int32,
            usecols=[0]  # Only use the first column (click label)
        ).values.reshape(-1)
        
        # Simplified logging
        if self.config.verbose > 0: print(f"Loaded {len(features)} samples with {features.shape[1]} features")
        if self.config.verbose > 0: print(f"Label distribution: {np.bincount(labels)}")
            
        return features, labels
    
    def create_stratified_splits(self, 
                                features: np.ndarray, 
                                labels: np.ndarray,
                                save_splits: bool = True) -> List[np.ndarray]:
        """
        Create stratified k-fold splits.
        
        Args:
            features: Feature array
            labels: Label array
            save_splits: Whether to save split data to files
            
        Returns:
            List of fold indices
        """
        # Simplified logging
        if self.config.verbose > 0: print(f"Creating {self.config.num_splits} stratified folds...")
            
        # Create stratified k-fold splitter
        skf = StratifiedKFold(
            n_splits=self.config.num_splits,
            shuffle=True,
            random_state=self.config.random_seed
        )
        
        # Generate splits
        fold_indices = []
        for train_idx, valid_idx in skf.split(features, labels):
            fold_indices.append(valid_idx)
        
        # 保存fold索引为list
        self.fold_indices = fold_indices
        
        # Save fold indices - 使用pickle处理不规则数组列表
        fold_index_path = os.path.join(self.config.data_path, self.config.fold_index_file.replace('.npy', '.pkl'))
        with open(fold_index_path, 'wb') as f:
            pickle.dump(self.fold_indices, f)
        
        if save_splits:
            self._save_fold_data(features, labels, fold_indices)
        
        # Simplified logging
        if self.config.verbose > 0: print(f"Created {len(fold_indices)} folds")
        if self.config.verbose > 0:
            for i, fold in enumerate(fold_indices):
                if self.config.verbose > 0: print(f"Fold {i+1}: {len(fold)} samples")
                
        return fold_indices
    
    def _save_fold_data(self, 
                       features: np.ndarray,
                       labels: np.ndarray, 
                       fold_indices: List[np.ndarray]) -> None:
        """
        Save fold data to separate directories.
        
        Args:
            features: Feature array
            labels: Label array
            fold_indices: List of fold indices
        """
        if self.config.verbose > 0:
            print("Saving fold data...")
            
        for i, fold_idx in enumerate(fold_indices):
            fold_num = i + 1
            if self.config.verbose > 0:
                print(f"Saving fold {fold_num}")
                
            # Extract fold data
            fold_features = features[fold_idx]
            fold_labels = labels[fold_idx]
            
            # Save features
            features_path = self.config.get_part_path(fold_num, self.config.train_x_npy)
            np.save(features_path, fold_features)
            
            # Save labels  
            labels_path = self.config.get_part_path(fold_num, self.config.train_y_npy)
            np.save(labels_path, fold_labels)
        
        if self.config.verbose > 0:
            print("Fold data saved successfully")
    
    def save_indices_data(self, fold_indices: Optional[List[np.ndarray]] = None) -> None:
        """
        Save feature indices for each fold.
        
        Args:
            fold_indices: List of fold indices (uses stored if None)
        """
        if fold_indices is None:
            fold_indices = self.fold_indices
            
        if fold_indices is None:
            raise ValueError("No fold indices available. Run create_stratified_splits first.")
            
        if self.config.verbose > 0:
            print("Saving feature indices for folds...")
            
        # Load feature indices
        feature_indices = pd.read_csv(
            self.config.train_i_path,
            header=None,
            sep=' ',
            dtype=np.int32
        ).values
        
        for i, fold_idx in enumerate(fold_indices):
            fold_num = i + 1
            if self.config.verbose > 0:
                print(f"Saving indices for fold {fold_num}")
                
            # Extract fold indices
            fold_feature_indices = feature_indices[fold_idx]
            
            # Save indices
            indices_path = self.config.get_part_path(fold_num, self.config.train_i_npy)
            np.save(indices_path, fold_feature_indices)
    
    def load_fold_indices(self) -> List[np.ndarray]:
        """
        Load previously saved fold indices.
        
        Returns:
            List of fold indices
        """
        fold_index_path = os.path.join(self.config.data_path, self.config.fold_index_file.replace('.npy', '.pkl'))
        
        if not os.path.exists(fold_index_path):
            raise FileNotFoundError(f"Fold indices not found: {fold_index_path}")
            
        with open(fold_index_path, 'rb') as f:
            self.fold_indices = pickle.load(f)
        return self.fold_indices
    
    def get_fold_data(self, fold_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for a specific fold.
        
        Args:
            fold_num: Fold number (1-indexed)
            
        Returns:
            Tuple of (feature_indices, feature_values, labels)
        """
        # Load fold data
        indices_path = self.config.get_part_path(fold_num, self.config.train_i_npy)
        values_path = self.config.get_part_path(fold_num, self.config.train_x_npy)
        labels_path = self.config.get_part_path(fold_num, self.config.train_y_npy)
        
        feature_indices = np.load(indices_path)
        feature_values = np.load(values_path)
        labels = np.load(labels_path)
        
        return feature_indices, feature_values, labels
    
    def get_scaled_fold_data(self, fold_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get scaled data for a specific fold.
        
        Args:
            fold_num: Fold number (1-indexed)
            
        Returns:
            Tuple of (feature_indices, scaled_feature_values, labels)
        """
        # Try to load scaled data first
        indices_path = self.config.get_part_path(fold_num, self.config.train_i_npy)
        scaled_values_path = self.config.get_part_path(fold_num, self.config.train_x2_npy)
        labels_path = self.config.get_part_path(fold_num, self.config.train_y_npy)
        
        if os.path.exists(scaled_values_path):
            feature_indices = np.load(indices_path)
            feature_values = np.load(scaled_values_path)
            labels = np.load(labels_path)
        else:
            # Fall back to original data
            if self.config.verbose > 0:
                print(f"Scaled data not found for fold {fold_num}, using original data")
            feature_indices, feature_values, labels = self.get_fold_data(fold_num)
        
        return feature_indices, feature_values, labels
    
    @classmethod
    def create_splits_from_config(cls, config: DataConfig) -> 'StratifiedDataSplitter':
        """
        Create splits from configuration.
        
        Args:
            config: Data configuration
            
        Returns:
            Configured splitter instance
        """
        splitter = cls(config)
        
        # Load processed data
        features, labels = splitter.load_processed_data()
        
        # Create splits
        fold_indices = splitter.create_stratified_splits(features, labels, save_splits=True)
        
        # Save indices data
        splitter.save_indices_data(fold_indices)
        
        return splitter


def create_stratified_splits(config: DataConfig, nrows: Optional[int] = None) -> StratifiedDataSplitter:
    """
    Convenience function to create stratified splits.
    
    Args:
        config: Data configuration
        nrows: Maximum number of rows to process
        
    Returns:
        Configured splitter instance
    """
    splitter = StratifiedDataSplitter(config)
    
    # Load data
    features, labels = splitter.load_processed_data(nrows=nrows)
    
    # Create splits
    splitter.create_stratified_splits(features, labels, save_splits=True)
    
    # Save indices
    splitter.save_indices_data()
    
    return splitter


class DataScalerUtility:
    """
    Utility class for scaling features across folds.
    """
    
    @staticmethod
    def scale_folds(config: DataConfig,
                   numerical_columns: List[int],
                   scale_method: str = 'log') -> None:
        """
        Scale numerical features across all folds.
        
        Args:
            config: Data configuration
            numerical_columns: List of numerical column indices
            scale_method: Scaling method ('log', 'minmax', 'standard')
        """
        if config.verbose > 0:
            print(f"Scaling {len(numerical_columns)} numerical columns using {scale_method}")
            
        for fold_num in range(1, config.num_splits + 1):
            if config.verbose > 0:
                print(f"Scaling fold {fold_num}")
                
            # Load original data
            values_path = config.get_part_path(fold_num, config.train_x_npy)
            if not os.path.exists(values_path):
                if config.verbose > 0:
                    print(f"Skipping fold {fold_num} - data not found")
                continue
                
            data = np.load(values_path)
            
            # Apply scaling
            if scale_method == 'log':
                DataScalerUtility._apply_log_scaling(data, numerical_columns, config.verbose)
            elif scale_method == 'minmax':
                scaler = MinMaxScaler()
                data[:, numerical_columns] = scaler.fit_transform(data[:, numerical_columns])
            elif scale_method == 'standard':
                scaler = StandardScaler()
                data[:, numerical_columns] = scaler.fit_transform(data[:, numerical_columns])
                # Add 1 to avoid negative values
                data[:, numerical_columns] += 1
            
            # Save scaled data
            scaled_path = config.get_part_path(fold_num, config.train_x2_npy)
            np.save(scaled_path, data)
    
    @staticmethod
    def _apply_log_scaling(data: np.ndarray, 
                          numerical_columns: List[int],
                          verbose: int = 0) -> None:
        """
        Apply logarithmic scaling as in original implementation.
        
        Args:
            data: Data array to scale in-place
            numerical_columns: Column indices to scale
            verbose: Verbosity level
        """
        import math
        
        for i, row in enumerate(data):
            if verbose > 0 and i % 100000 == 0:
                print(f"  Scaling row {i}")
                
            for col in numerical_columns:
                value = row[col]
                if value > 2:
                    data[i, col] = int(math.log(float(value))**2)
