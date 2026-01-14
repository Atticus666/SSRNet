"""
Data processing module for AutoInt implementation.

This module provides optimized data processing utilities for various CTR datasets
including Criteo, Avazu, KDD2012, and Ali-CCP.
"""

from .base import BaseDataProcessor
from .criteo_optimized import CriteoProcessor  
from .avazu_optimized import AvazuProcessor
from .kdd2012_optimized import KDD2012Processor
from .aliccp_optimized import AliccpProcessor
from .kfold_split import StratifiedDataSplitter
from .config import DataConfig, CriteoConfig, AvazuConfig, KDD2012Config, CriteoDiscConfig, AliccpConfig  

__all__ = [
    'BaseDataProcessor',
    'CriteoProcessor', 
    'AvazuProcessor',
    'KDD2012Processor',
    'AliccpProcessor',
    'StratifiedDataSplitter',
    'DataConfig',
    'CriteoConfig',     
    'AvazuConfig',       
    'KDD2012Config',
    'CriteoDiscConfig',
    'AliccpConfig'      
]
