"""
Metrics calculation utilities for model.

This module provides comprehensive metrics calculation for CTR prediction tasks,
including AUC, log loss, and other relevant evaluation metrics.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score
from typing import Tuple, Dict, Any, Optional, List
import logging


class MetricsCalculator:
    """
    Comprehensive metrics calculator for CTR prediction tasks.
    
    Provides calculation of various metrics including AUC, log loss,
    accuracy, precision, and recall for binary classification.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Threshold for binary classification metrics
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.labels = []
        self.losses = []
    
    def update(self, 
               y_true: np.ndarray, 
               y_pred: np.ndarray, 
               loss: Optional[float] = None):
        """
        Update metrics with new batch of predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            loss: Optional loss value
        """
        # Flatten arrays if needed
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        self.labels.extend(y_true_flat)
        self.predictions.extend(y_pred_flat)
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute_auc(self, 
                   y_true: Optional[np.ndarray] = None,
                   y_pred: Optional[np.ndarray] = None) -> float:
        """
        Compute Area Under the ROC Curve.
        
        Args:
            y_true: True labels (uses accumulated if None)
            y_pred: Predicted probabilities (uses accumulated if None)
            
        Returns:
            AUC score
        """
        if y_true is None or y_pred is None:
            if not self.labels or not self.predictions:
                return 0.0
            y_true = np.array(self.labels)
            y_pred = np.array(self.predictions)
        
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError as e:
            logging.warning(f"AUC calculation failed: {e}")
            return 0.0
    
    def compute_logloss(self,
                       y_true: Optional[np.ndarray] = None,
                       y_pred: Optional[np.ndarray] = None) -> float:
        """
        Compute logarithmic loss.
        
        Args:
            y_true: True labels (uses accumulated if None)
            y_pred: Predicted probabilities (uses accumulated if None)
            
        Returns:
            Log loss value
        """
        if y_true is None or y_pred is None:
            if not self.labels or not self.predictions:
                return float('inf')
            y_true = np.array(self.labels)
            y_pred = np.array(self.predictions)
        
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        try:
            return log_loss(y_true, y_pred_clipped)
        except ValueError as e:
            logging.warning(f"Log loss calculation failed: {e}")
            return float('inf')
    
    def compute_accuracy(self,
                        y_true: Optional[np.ndarray] = None,
                        y_pred: Optional[np.ndarray] = None) -> float:
        """
        Compute classification accuracy.
        
        Args:
            y_true: True labels (uses accumulated if None)
            y_pred: Predicted probabilities (uses accumulated if None)
            
        Returns:
            Accuracy score
        """
        if y_true is None or y_pred is None:
            if not self.labels or not self.predictions:
                return 0.0
            y_true = np.array(self.labels)
            y_pred = np.array(self.predictions)
        
        # Convert predictions to binary
        y_pred_binary = (y_pred >= self.threshold).astype(int)
        
        try:
            return accuracy_score(y_true, y_pred_binary)
        except ValueError as e:
            logging.warning(f"Accuracy calculation failed: {e}")
            return 0.0
    
    def compute_precision(self,
                         y_true: Optional[np.ndarray] = None,
                         y_pred: Optional[np.ndarray] = None) -> float:
        """
        Compute precision score.
        
        Args:
            y_true: True labels (uses accumulated if None)
            y_pred: Predicted probabilities (uses accumulated if None)
            
        Returns:
            Precision score
        """
        if y_true is None or y_pred is None:
            if not self.labels or not self.predictions:
                return 0.0
            y_true = np.array(self.labels)
            y_pred = np.array(self.predictions)
        
        y_pred_binary = (y_pred >= self.threshold).astype(int)
        
        try:
            return precision_score(y_true, y_pred_binary, zero_division=0)
        except ValueError as e:
            logging.warning(f"Precision calculation failed: {e}")
            return 0.0
    
    def compute_recall(self,
                      y_true: Optional[np.ndarray] = None,
                      y_pred: Optional[np.ndarray] = None) -> float:
        """
        Compute recall score.
        
        Args:
            y_true: True labels (uses accumulated if None)
            y_pred: Predicted probabilities (uses accumulated if None)
            
        Returns:
            Recall score
        """
        if y_true is None or y_pred is None:
            if not self.labels or not self.predictions:
                return 0.0
            y_true = np.array(self.labels)
            y_pred = np.array(self.predictions)
        
        y_pred_binary = (y_pred >= self.threshold).astype(int)
        
        try:
            return recall_score(y_true, y_pred_binary, zero_division=0)
        except ValueError as e:
            logging.warning(f"Recall calculation failed: {e}")
            return 0.0
    
    def compute_f1_score(self,
                        y_true: Optional[np.ndarray] = None,
                        y_pred: Optional[np.ndarray] = None) -> float:
        """
        Compute F1 score.
        
        Args:
            y_true: True labels (uses accumulated if None)
            y_pred: Predicted probabilities (uses accumulated if None)
            
        Returns:
            F1 score
        """
        precision = self.compute_precision(y_true, y_pred)
        recall = self.compute_recall(y_true, y_pred)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def compute_all_metrics(self,
                           y_true: Optional[np.ndarray] = None,
                           y_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            y_true: True labels (uses accumulated if None)
            y_pred: Predicted probabilities (uses accumulated if None)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'auc': self.compute_auc(y_true, y_pred),
            'logloss': self.compute_logloss(y_true, y_pred),
            'accuracy': self.compute_accuracy(y_true, y_pred),
            'precision': self.compute_precision(y_true, y_pred),
            'recall': self.compute_recall(y_true, y_pred),
            'f1': self.compute_f1_score(y_true, y_pred)
        }
        
        # Add average loss if available
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
        
        return metrics
    
    def get_sample_count(self) -> int:
        """Get number of accumulated samples."""
        return len(self.labels)
    
    def get_positive_ratio(self) -> float:
        """Get ratio of positive samples."""
        if not self.labels:
            return 0.0
        return np.mean(self.labels)


class TensorFlowMetrics:
    """
    TensorFlow-based metrics for integration with TF training loops.
    """
    
    @staticmethod
    def create_auc_metric(name: str = 'auc') -> tf.keras.metrics.AUC:
        """
        Create AUC metric.
        
        Args:
            name: Metric name
            
        Returns:
            AUC metric instance
        """
        return tf.keras.metrics.AUC(name=name)
    
    @staticmethod
    def create_binary_accuracy_metric(name: str = 'binary_accuracy', 
                                    threshold: float = 0.5) -> tf.keras.metrics.BinaryAccuracy:
        """
        Create binary accuracy metric.
        
        Args:
            name: Metric name
            threshold: Classification threshold
            
        Returns:
            Binary accuracy metric instance
        """
        return tf.keras.metrics.BinaryAccuracy(name=name, threshold=threshold)
    
    @staticmethod
    def create_precision_metric(name: str = 'precision',
                              threshold: float = 0.5) -> tf.keras.metrics.Precision:
        """
        Create precision metric.
        
        Args:
            name: Metric name
            threshold: Classification threshold
            
        Returns:
            Precision metric instance
        """
        return tf.keras.metrics.Precision(name=name, thresholds=threshold)
    
    @staticmethod
    def create_recall_metric(name: str = 'recall',
                           threshold: float = 0.5) -> tf.keras.metrics.Recall:
        """
        Create recall metric.
        
        Args:
            name: Metric name
            threshold: Classification threshold
            
        Returns:
            Recall metric instance
        """
        return tf.keras.metrics.Recall(name=name, thresholds=threshold)
    
    @staticmethod
    def create_all_metrics(threshold: float = 0.5) -> List[tf.keras.metrics.Metric]:
        """
        Create all standard metrics.
        
        Args:
            threshold: Classification threshold
            
        Returns:
            List of metric instances
        """
        return [
            TensorFlowMetrics.create_auc_metric(),
            TensorFlowMetrics.create_binary_accuracy_metric(threshold=threshold),
            TensorFlowMetrics.create_precision_metric(threshold=threshold),
            TensorFlowMetrics.create_recall_metric(threshold=threshold)
        ]


def evaluate_predictions(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        threshold: float = 0.5,
                        verbose: bool = True) -> Dict[str, float]:
    """
    Convenience function to evaluate predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        verbose: Whether to print results
        
    Returns:
        Dictionary containing all metrics
    """
    calculator = MetricsCalculator(threshold=threshold)
    metrics = calculator.compute_all_metrics(y_true, y_pred)
    
    if verbose:
        print("Evaluation Results:")
        print("-" * 40)
        for name, value in metrics.items():
            if name == 'logloss':
                print(f"{name.upper():>12}: {value:.6f}")
            else:
                print(f"{name.upper():>12}: {value:.4f}")
        print("-" * 40)
    
    return metrics


def compute_ctr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute primary CTR metrics (AUC and LogLoss).
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        
    Returns:
        Tuple of (AUC, LogLoss)
    """
    calculator = MetricsCalculator()
    auc = calculator.compute_auc(y_true, y_pred)
    logloss = calculator.compute_logloss(y_true, y_pred)
    
    return auc, logloss
