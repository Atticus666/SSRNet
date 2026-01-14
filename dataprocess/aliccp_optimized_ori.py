"""Ali-CCP (Alibaba Click and Conversion Prediction) dataset processor.

This module provides optimized data processing for the Ali-CCP dataset,
which includes multi-task learning with click and purchase predictions.

Dataset: https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408
"""

import os
import re
import random
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Optional, Dict, List
from collections import defaultdict
from tqdm import tqdm

from .base import BaseDataProcessor
from .config import AlicppConfig


class AliccpProcessor(BaseDataProcessor):
    """Processor for Ali-CCP dataset with multi-task learning support."""
    
    def __init__(self, config: Optional[AlicppConfig] = None):
        """Initialize Ali-CCP processor.
        
        Args:
            config: Ali-CCP configuration object
        """
        if config is None:
            config = AlicppConfig()
        super().__init__(config)
        
        # Ali-CCP specific settings
        self.use_columns = config.use_columns
        self.field_size = len(self.use_columns)
        self.min_category_count = config.min_category_count
        self.dev_split_ratio = config.dev_split_ratio
        
        # Feature vocabulary
        self.feature_vocab = {}
        self.feature_size = 0
        
        if self.config.verbose > 0:
            print(f"AliccpProcessor initialized")
            print(f"  Using {self.field_size} features: {self.use_columns}")
            print(f"  Min category count: {self.min_category_count}")
            print(f"  Dev split ratio: {self.dev_split_ratio}")
    
    def load_raw_data(self, nrows: Optional[int] = None, 
                     mode: str = 'train') -> Tuple[pd.DataFrame, np.ndarray]:
        """Load raw Ali-CCP data from skeleton and common features files.
        
        Args:
            nrows: Maximum number of rows to load
            mode: 'train' or 'test'
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        if mode == 'train':
            skeleton_file = self.config.skeleton_file_train
            common_feat_file = self.config.common_feat_file_train
        else:
            skeleton_file = self.config.skeleton_file_test
            common_feat_file = self.config.common_feat_file_test
        
        if not os.path.exists(skeleton_file):
            raise FileNotFoundError(f"Skeleton file not found: {skeleton_file}")
        if not os.path.exists(common_feat_file):
            raise FileNotFoundError(f"Common features file not found: {common_feat_file}")
        
        if self.config.verbose > 0:
            print(f"Loading Ali-CCP {mode} data...")
            print(f"  Skeleton file: {skeleton_file}")
            print(f"  Common features file: {common_feat_file}")
        
        # Step 1: Load common features into dictionary
        if self.config.verbose > 0:
            print("Loading common features...")
        
        common_feat_dict = {}
        with open(common_feat_file, 'r') as fr:
            for line_idx, line in enumerate(fr):
                if nrows and line_idx >= nrows:
                    break
                    
                line_list = line.strip().split(',')
                if len(line_list) < 3:
                    continue
                
                # Parse key-value pairs using delimiters
                kv = np.array(re.split('\x01|\x02|\x03', line_list[2]))
                if len(kv) < 3:
                    continue
                    
                key = kv[range(0, len(kv), 3)]
                value = kv[range(1, len(kv), 3)]
                feat_dict = dict(zip(key, value))
                common_feat_dict[line_list[0]] = feat_dict
                
                if self.config.verbose > 0 and (line_idx + 1) % 100000 == 0:
                    print(f"  Loaded {line_idx + 1:,} common features")
        
        if self.config.verbose > 0:
            print(f"Loaded {len(common_feat_dict):,} common feature records")
        
        # Step 2: Load skeleton data and join with common features
        if self.config.verbose > 0:
            print("Loading and joining skeleton data with common features...")
        
        features_list = []
        labels_list = []
        
        with open(skeleton_file, 'r') as fr:
            for line_idx, line in enumerate(fr):
                if nrows and line_idx >= len(common_feat_dict):
                    break
                
                line_list = line.strip().split(',')
                if len(line_list) < 6:
                    continue
                
                # Skip samples with inconsistent labels (click=0, purchase=1)
                click_label = line_list[1]
                purchase_label = line_list[2]
                if click_label == '0' and purchase_label == '1':
                    continue
                
                # Parse skeleton features
                kv = np.array(re.split('\x01|\x02|\x03', line_list[5]))
                if len(kv) < 3:
                    continue
                    
                key = kv[range(0, len(kv), 3)]
                value = kv[range(1, len(kv), 3)]
                feat_dict = dict(zip(key, value))
                
                # Join with common features
                user_id = line_list[3]
                if user_id in common_feat_dict:
                    feat_dict.update(common_feat_dict[user_id])
                
                # Extract features based on use_columns
                feature_row = {}
                for col in self.use_columns:
                    feature_row[col] = feat_dict.get(col, '0')
                
                features_list.append(feature_row)
                
                # Extract labels: [click, purchase]
                labels_list.append([int(click_label), int(purchase_label)])
                
                if self.config.verbose > 0 and (line_idx + 1) % 100000 == 0:
                    print(f"  Processed {line_idx + 1:,} samples")
        
        # Convert to DataFrame and array
        features_df = pd.DataFrame(features_list)
        labels = np.array(labels_list, dtype=np.int32)
        
        if self.config.verbose > 0:
            print(f"Loaded {len(features_df):,} samples with {features_df.shape[1]} features")
            print(f"Click distribution: {np.bincount(labels[:, 0])}")
            print(f"Purchase distribution: {np.bincount(labels[:, 1])}")
        
        return features_df, labels
    
    def preprocess_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features into index and value arrays.
        
        Args:
            data: Raw feature dataframe
            
        Returns:
            Tuple of (feature_indices, feature_values)
        """
        num_samples = len(data)
        
        if self.config.verbose > 0:
            print(f"Preprocessing {num_samples:,} samples...")
        
        # Build vocabulary first
        self._build_vocabulary(data)
        
        # Initialize arrays
        feature_indices = np.zeros((num_samples, self.field_size), dtype=np.int32)
        feature_values = np.ones((num_samples, self.field_size), dtype=np.float32)
        
        # Encode categorical features
        if self.config.verbose > 0:
            print("Encoding categorical features...")
        
        for field_idx, col_name in enumerate(self.use_columns):
            column_data = data[col_name].astype(str)
            vocab = self.feature_vocab[col_name]
            
            # Calculate feature offset for this field
            feature_offset = sum(len(self.feature_vocab[c]) for c in self.use_columns[:field_idx])
            
            # Map values to indices
            def map_to_vocab(value):
                return vocab.get(value, vocab.get('<RARE>', 0))
            
            vocab_indices = column_data.apply(map_to_vocab).values
            feature_indices[:, field_idx] = vocab_indices + feature_offset
            # All categorical features have value 1.0
            feature_values[:, field_idx] = 1.0
        
        if self.config.verbose > 0:
            print("Feature preprocessing completed")
        
        return feature_indices, feature_values
    
    def _build_vocabulary(self, data: pd.DataFrame) -> None:
        """Build vocabulary for categorical features with frequency filtering.
        
        Args:
            data: Input feature dataframe
        """
        if self.config.verbose > 0:
            print("Building feature vocabulary...")
        
        self.feature_vocab = {}
        self.feature_size_per_field = []
        
        for col_name in self.use_columns:
            column_data = data[col_name].astype(str)
            value_counts = column_data.value_counts()
            
            # Create vocabulary with rare value handling
            vocab = {}
            
            # Index 0 is reserved for unknown/missing values
            vocab['<RARE>'] = 0
            
            # Assign indices to frequent values (count > min_category_count)
            next_index = 1
            for value, count in value_counts.items():
                if value in ['', 'nan', 'None', '0'] or pd.isna(value):
                    # Missing/default values map to rare index
                    vocab[str(value)] = 0
                elif count > self.min_category_count:
                    # Frequent values get unique indices
                    vocab[str(value)] = next_index
                    next_index += 1
                else:
                    # Rare values map to rare index
                    vocab[str(value)] = 0
            
            self.feature_vocab[col_name] = vocab
            self.feature_size_per_field.append(next_index)
            
            if self.config.verbose > 1:
                frequent_count = sum(1 for count in value_counts.values() 
                                   if count > self.min_category_count)
                print(f"  Feature {col_name}: {frequent_count} frequent categories, "
                      f"vocab size: {next_index}")
        
        # Calculate total feature size
        self.feature_size = sum(self.feature_size_per_field)
        
        if self.config.verbose > 0:
            print(f"Built vocabulary with {self.feature_size:,} total dimensions")
            print(f"Feature sizes per field: {self.feature_size_per_field}")
    
    def save_processed_data(self,
                          feature_indices: np.ndarray,
                          feature_values: np.ndarray,
                          labels: np.ndarray,
                          output_dir: Optional[str] = None,
                          split_dev: bool = True) -> None:
        """Save processed data in standard format.
        
        Args:
            feature_indices: Feature index array
            feature_values: Feature value array
            labels: Label array (Nx2 for click and purchase)
            output_dir: Output directory (uses config.data_path if None)
            split_dev: Whether to split train/dev sets
        """
        if output_dir is None:
            output_dir = self.config.data_path
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.config.verbose > 0:
            print(f"Saving processed data to {output_dir}")
        
        # Split train/dev if requested
        if split_dev:
            num_samples = len(feature_indices)
            indices = np.arange(num_samples)
            
            # Set random seed for reproducibility
            np.random.seed(self.config.random_seed)
            np.random.shuffle(indices)
            
            # Split 90% train, 10% dev
            split_idx = int(num_samples * (1 - self.dev_split_ratio))
            train_indices = indices[:split_idx]
            dev_indices = indices[split_idx:]
            
            # Save train set
            train_dir = os.path.join(output_dir, 'train')
            os.makedirs(train_dir, exist_ok=True)
            self._save_data_split(
                feature_indices[train_indices],
                feature_values[train_indices],
                labels[train_indices],
                train_dir,
                'train'
            )
            
            # Save dev set
            dev_dir = os.path.join(output_dir, 'dev')
            os.makedirs(dev_dir, exist_ok=True)
            self._save_data_split(
                feature_indices[dev_indices],
                feature_values[dev_indices],
                labels[dev_indices],
                dev_dir,
                'dev'
            )
        else:
            # Save all data without splitting
            self._save_data_split(
                feature_indices,
                feature_values,
                labels,
                output_dir,
                'all'
            )
        
        # Save vocabulary
        vocab_path = os.path.join(output_dir, 'ctrcvr_enum.pkl')
        joblib.dump(self.feature_vocab, vocab_path, compress=3)
        
        # Save feature size
        feature_size_path = os.path.join(output_dir, 'feature_size.npy')
        np.save(feature_size_path, np.array([self.feature_size]))
        
        if self.config.verbose > 0:
            print(f"Saved processed data")
            print(f"  Feature size: {self.feature_size:,}")
            print(f"  Vocabulary: {vocab_path}")
    
    def _save_data_split(self,
                        feature_indices: np.ndarray,
                        feature_values: np.ndarray,
                        labels: np.ndarray,
                        output_dir: str,
                        split_name: str) -> None:
        """Save a single data split.
        
        Args:
            feature_indices: Feature index array
            feature_values: Feature value array
            labels: Label array
            output_dir: Output directory
            split_name: Name of the split (train/dev/all)
        """
        # Save feature indices
        indices_path = os.path.join(output_dir, 'train_i.txt')
        with open(indices_path, 'w') as f:
            for i in range(len(feature_indices)):
                indices_line = ' '.join(map(str, feature_indices[i]))
                f.write(indices_line + '\n')
        
        # Save feature values
        values_path = os.path.join(output_dir, 'train_x.txt')
        with open(values_path, 'w') as f:
            for i in range(len(feature_values)):
                values_line = ' '.join(map(str, feature_values[i]))
                f.write(values_line + '\n')
        
        # Save labels (click and purchase)
        # Save click labels
        click_labels_path = os.path.join(output_dir, 'train_y_click.txt')
        np.savetxt(click_labels_path, labels[:, 0], fmt='%d')
        
        # Save purchase labels
        purchase_labels_path = os.path.join(output_dir, 'train_y_purchase.txt')
        np.savetxt(purchase_labels_path, labels[:, 1], fmt='%d')
        
        # Also save combined labels for compatibility
        labels_path = os.path.join(output_dir, 'train_y.txt')
        with open(labels_path, 'w') as f:
            for i in range(len(labels)):
                f.write(f"{labels[i, 0]} {labels[i, 1]}\n")
        
        if self.config.verbose > 0:
            print(f"  {split_name}: {len(labels):,} samples")
            print(f"    Click: {np.bincount(labels[:, 0])}")
            print(f"    Purchase: {np.bincount(labels[:, 1])}")
    
    def process_dataset(self,
                       nrows: Optional[int] = None,
                       save: bool = True,
                       split_dev: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process the complete Ali-CCP dataset pipeline.
        
        Args:
            nrows: Maximum number of rows to process
            save: Whether to save processed data
            split_dev: Whether to split train/dev sets
            
        Returns:
            Tuple of (feature_indices, feature_values, labels)
        """
        # Load and process training data
        features, labels = self.load_raw_data(nrows=nrows, mode='train')
        feature_indices, feature_values = self.preprocess_features(features)
        
        if save:
            self.save_processed_data(
                feature_indices,
                feature_values,
                labels,
                split_dev=split_dev
            )
        
        return feature_indices, feature_values, labels
    
    def process_test_data(self,
                         nrows: Optional[int] = None,
                         save: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process the Ali-CCP test dataset.
        
        Args:
            nrows: Maximum number of rows to process
            save: Whether to save processed data
            
        Returns:
            Tuple of (feature_indices, feature_values, labels)
        """
        if self.config.verbose > 0:
            print("\nProcessing test data...")
        
        # Load test data
        features, labels = self.load_raw_data(nrows=nrows, mode='test')
        
        # Use existing vocabulary (built from training data)
        if not self.feature_vocab:
            raise ValueError("Vocabulary not built. Please process training data first.")
        
        # Initialize arrays
        num_samples = len(features)
        feature_indices = np.zeros((num_samples, self.field_size), dtype=np.int32)
        feature_values = np.ones((num_samples, self.field_size), dtype=np.float32)
        
        # Encode features using training vocabulary with actual values
        for field_idx, col_name in enumerate(self.use_columns):
            column_data = features[col_name].astype(str)
            vocab = self.feature_vocab[col_name]
            
            feature_offset = sum(len(self.feature_vocab[c]) for c in self.use_columns[:field_idx])
            
            # Parse feature_id and feature_value from "feature_id:feature_value" format
            parsed_data = column_data.str.split(':', n=1, expand=True)
            if parsed_data.shape[1] == 2:
                feature_ids = parsed_data[0].astype(str)
                feature_vals = pd.to_numeric(parsed_data[1], errors='coerce').fillna(1.0)
            else:
                # Fallback for malformed data
                feature_ids = column_data
                feature_vals = pd.Series([1.0] * len(column_data))
            
            def map_to_vocab(value):
                return vocab.get(value, vocab.get('<RARE>', 0))
            
            vocab_indices = feature_ids.apply(map_to_vocab).values
            feature_indices[:, field_idx] = vocab_indices + feature_offset
            
            # Use actual feature values
            feature_values[:, field_idx] = feature_vals.astype(np.float32)
        
        if save:
            test_dir = os.path.join(self.config.data_path, 'test')
            os.makedirs(test_dir, exist_ok=True)
            self._save_data_split(
                feature_indices,
                feature_values,
                labels,
                test_dir,
                'test'
            )
        
        return feature_indices, feature_values, labels
    
    @classmethod
    def create_from_source(cls,
                          source_path: str,
                          output_path: str,
                          nrows: Optional[int] = None,
                          process_test: bool = True) -> 'AliccpProcessor':
        """Create processor and process data from source files.
        
        Args:
            source_path: Path to source data directory
            output_path: Output directory for processed data
            nrows: Maximum number of rows to process
            process_test: Whether to also process test data
            
        Returns:
            Configured AliccpProcessor instance
        """
        config = AlicppConfig()
        config.source_path = source_path
        config.data_path = output_path
        
        # Update file paths relative to source_path
        config.skeleton_file_train = os.path.join(source_path, 'sample_skeleton_train.csv')
        config.skeleton_file_test = os.path.join(source_path, 'sample_skeleton_test.csv')
        config.common_feat_file_train = os.path.join(source_path, 'common_features_train.csv')
        config.common_feat_file_test = os.path.join(source_path, 'common_features_test.csv')
        
        processor = cls(config)
        
        # Process training data
        processor.process_dataset(nrows=nrows, save=True, split_dev=True)
        
        # Process test data if requested
        if process_test:
            processor.process_test_data(nrows=nrows, save=True)
        
        return processor


def preprocess_aliccp_dataset(source_path: str,
                              output_path: str,
                              nrows: Optional[int] = None,
                              process_test: bool = True,
                              verbose: int = 1) -> int:
    """
    Convenience function to preprocess Ali-CCP dataset.
    
    Args:
        source_path: Path to raw Ali-CCP data directory
        output_path: Output directory for processed data
        nrows: Maximum number of rows to process (None for all)
        process_test: Whether to also process test data
        verbose: Verbosity level (0=quiet, 1=normal, 2=detailed)
        
    Returns:
        Total number of features after processing
    """
    processor = AliccpProcessor.create_from_source(
        source_path=source_path,
        output_path=output_path,
        nrows=nrows,
        process_test=process_test
    )
    
    if verbose > 0:
        print(f"\nFinal feature size: {processor.feature_size:,}")
    
    return processor.feature_size
