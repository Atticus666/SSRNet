"""
Configuration settings for data processing.

This module provides centralized configuration management for different datasets
and processing parameters.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration class for data processing parameters."""
    
    # General settings
    data_path: str = './data/'
    random_seed: int = 2018
    num_splits: int = 10
    debug: bool = False
    verbose: int = 1
    
    # File names
    train_i_file: str = 'train_i.txt'
    train_x_file: str = 'train_x.txt'  
    train_y_file: str = 'train_y.txt'
    
    # Numpy file names
    train_i_npy: str = 'train_i.npy'
    train_x_npy: str = 'train_x.npy'
    train_x2_npy: str = 'train_x2.npy'  # scaled version
    train_y_npy: str = 'train_y.npy'
    feature_size_file: str = 'feature_size.npy'
    fold_index_file: str = 'fold_index.npy'
    
    def __post_init__(self):
        """Initialize derived paths after object creation."""
        self.train_i_path = os.path.join(self.data_path, self.train_i_file)
        self.train_x_path = os.path.join(self.data_path, self.train_x_file)
        self.train_y_path = os.path.join(self.data_path, self.train_y_file)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)
    
    def get_part_path(self, part_id: int, filename: str) -> str:
        """Get path for a specific data part."""
        part_dir = os.path.join(self.data_path, f'part{part_id}')
        os.makedirs(part_dir, exist_ok=True)
        return os.path.join(part_dir, filename)


    # 缩放相关配置
    force_regenerate: bool = False  # 强制重新生成缩放数据
    save_text_format: bool = False  # 是否保存文本格式
    default_scale_method: str = 'log'  # 默认缩放方法
    
    # 数值特征列配置
    numerical_columns: Optional[list[int]] = None
    
    def get_scaling_config(self) -> Dict[str, Any]:
        """获取缩放配置"""
        return {
            'method': self.default_scale_method,
            'numerical_columns': self.numerical_columns,
            'force_regenerate': self.force_regenerate
        }

# Dataset-specific configurations
class CriteoConfig(DataConfig):
    """Configuration specific to Criteo dataset."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_file = kwargs.get('source_file', './data/Criteo/train_examples.txt')
        self.field_size = 39
        self.num_numerical_features = 13
        self.num_categorical_features = 26
        self.min_category_count = 5  # Minimum count for category to get unique index
        self.numerical_columns = list(range(13))  # Criteo前13列为数值特征
        self.default_scale_method = 'log'  # Criteo使用对数缩放
        self.num_buckets_per_feature = kwargs.get('num_buckets_per_feature', 100)  # 每个数值特征的分桶数
        self.use_numerical_discretization = kwargs.get('use_numerical_discretization', False)  # 是否启用数值离散化
        

class AvazuConfig(DataConfig):
    """Configuration specific to Avazu dataset."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_file = kwargs.get('source_file', './data/Avazu/train')
        self.field_size = 23  # 24 total - 1 label
        self.label_index = 1
        self.total_columns = 24
        self.min_category_counts = [5] * 24  # Minimum counts per feature
        self.numerical_columns = []  # Avazu全部为分类特征
        self.default_scale_method = None
        

class KDD2012Config(DataConfig):
    """Configuration specific to KDD2012 dataset."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_file = kwargs.get('source_file', './data/KDD2012/training.txt')
        self.field_size = 11  # Adjust based on actual KDD2012 structure
        self.numerical_columns = list(range(3))  # 根据KDD2012实际情况调整
        self.default_scale_method = 'standard'


class CriteoDiscConfig(DataConfig):
    """Configuration specific to Criteo dataset with discretized numerical features."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_file = kwargs.get('source_file', '/data/oss_bucket_0/ssrnet/data/Criteo_disc/train_examples.txt')
        self.field_size = 39
        self.num_numerical_features = 13
        self.num_categorical_features = 26
        self.min_category_count = 5  # Minimum count for category to get unique index
        self.numerical_columns = list(range(13))  # Criteo前13列为数值特征
        self.default_scale_method = 'log'  # Criteo使用对数缩放
        self.num_buckets_per_feature = kwargs.get('num_buckets_per_feature', 100)  # 每个数值特征的分桶数
        # 离散化数据集默认启用数值离散化
        self.use_numerical_discretization = kwargs.get('use_numerical_discretization', True)


class AliccpConfig(DataConfig):
    """Configuration specific to Ali-CCP (Alibaba Click and Conversion Prediction) dataset."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Data paths
        self.data_path = kwargs.get('data_path', './Aliccp/')
        self.source_path = kwargs.get('source_path', './data/Aliccp/')
        self.skeleton_file_train = kwargs.get('skeleton_file_train', 
                                             'data/sample_skeleton_train.csv')
        self.skeleton_file_test = kwargs.get('skeleton_file_test',
                                            'data/sample_skeleton_test.csv')
        self.common_feat_file_train = kwargs.get('common_feat_file_train',
                                                 'data/common_features_train.csv')
        self.common_feat_file_test = kwargs.get('common_feat_file_test',
                                               'data/common_features_test.csv')
        
        # Feature configuration
        self.use_columns = kwargs.get('use_columns', [
            '101', '121', '122', '124', '125', '126', '127', '128', '129', '205',
            '206', '207', '216', '508', '509', '702', '853', '301'
        ])
        self.field_size = len(self.use_columns)  # 18 features
        
        # All features are categorical in Ali-CCP
        self.num_numerical_features = 0
        self.num_categorical_features = self.field_size
        self.numerical_columns = []
        
        # Preprocessing parameters
        self.min_category_count = kwargs.get('min_category_count', 5)
        self.dev_split_ratio = kwargs.get('dev_split_ratio', 0.1)  # 10% for dev set
        self.use_log_scaling = kwargs.get('use_log_scaling', True)  # Apply log scaling to feature values
        
        # Multi-task labels
        self.has_click_label = True
        self.has_purchase_label = True
        self.label_columns = ['click', 'purchase']
        
        # File format
        self.delimiter_level1 = '\x01'  # Primary delimiter in feature string
        self.delimiter_level2 = '\x02'  # Secondary delimiter 
        self.delimiter_level3 = '\x03'  # Tertiary delimiter
