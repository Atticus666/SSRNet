"""
Callback utilities for training.

This module provides custom callbacks for training monitoring,
early stopping, and model checkpointing.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional
import logging


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    """
    Custom early stopping callback with enhanced functionality.
    
    Monitors validation metrics and stops training when no improvement
    is observed for a specified number of epochs.
    """
    
    def __init__(self,
                 monitor: str = 'val_auc',
                 patience: int = 5,
                 mode: str = 'max',
                 min_delta: float = 0.0001,
                 restore_best_weights: bool = True,
                 verbose: int = 1):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
            verbose: Verbosity level
        """
        super().__init__()
        
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        elif mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Reset callback state at training start."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if self.mode == 'max':
            self.best = -np.inf
        else:
            self.best = np.inf
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check for early stopping at epoch end."""
        if logs is None:
            return
            
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose > 0:
                print(f"Warning: Monitor metric '{self.monitor}' not found in logs")
            return
        
        # Check if current value is better than best
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    print("Restoring model weights from best epoch")
                self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Print early stopping information at training end."""
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Early stopping at epoch {self.stopped_epoch + 1}")
            print(f"Best {self.monitor}: {self.best:.6f}")


class ModelCheckpointCallback(tf.keras.callbacks.Callback):
    """
    Custom model checkpoint callback with flexible saving options.
    
    Saves model checkpoints based on metric improvements with
    support for different file formats and naming schemes.
    """
    
    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_auc',
                 mode: str = 'max',
                 save_best_only: bool = True,
                 save_weights_only: bool = True,
                 verbose: int = 1):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save model checkpoints
            monitor: Metric to monitor for best model
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            save_best_only: Whether to save only the best model
            save_weights_only: Whether to save only weights or full model
            verbose: Verbosity level
        """
        super().__init__()
        
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        elif mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Save model checkpoint if conditions are met."""
        if logs is None:
            return
            
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose > 0:
                print(f"Warning: Monitor metric '{self.monitor}' not found in logs")
            return
        
        # Determine if we should save
        should_save = False
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
                should_save = True
                if self.verbose > 0:
                    print(f"New best {self.monitor}: {current:.6f} - saving model")
        else:
            should_save = True
        
        # Save model if conditions met
        if should_save:
            try:
                if self.save_weights_only:
                    self.model.save_weights(self.filepath)
                else:
                    self.model.save(self.filepath)
                    
                if self.verbose > 0 and not self.save_best_only:
                    print(f"Saved model to {self.filepath}")
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to save model: {e}")


class MetricsLoggerCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for detailed metrics logging.
    
    Logs training and validation metrics with timestamps
    and additional statistics.
    """
    
    def __init__(self, 
                 log_file: Optional[str] = None,
                 verbose: int = 1):
        """
        Initialize metrics logger callback.
        
        Args:
            log_file: Optional file to save logs
            verbose: Verbosity level
        """
        super().__init__()
        
        self.log_file = log_file
        self.verbose = verbose
        self.epoch_logs = []
        
        if log_file:
            # Create log directory if needed
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Initialize log file with headers
            with open(log_file, 'w') as f:
                f.write("epoch,train_loss,train_auc,val_loss,val_auc,timestamp\n")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log metrics at epoch end."""
        if logs is None:
            return
            
        import time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Extract key metrics
        train_loss = logs.get('loss', 0.0)
        train_auc = logs.get('auc', 0.0)
        val_loss = logs.get('val_loss', 0.0)
        val_auc = logs.get('val_auc', 0.0)
        
        # Store epoch log
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'timestamp': timestamp
        }
        self.epoch_logs.append(epoch_log)
        
        # Print to console if verbose
        if self.verbose > 0:
            print(f"Epoch {epoch + 1:3d} | "
                  f"Loss: {train_loss:.4f} | "
                  f"AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f}")
        
        # Write to log file if specified
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(f"{epoch + 1},{train_loss:.6f},{train_auc:.6f},"
                           f"{val_loss:.6f},{val_auc:.6f},{timestamp}\n")
            except Exception as e:
                logging.warning(f"Failed to write to log file: {e}")
    
    def get_logs(self) -> list:
        """Get all epoch logs."""
        return self.epoch_logs.copy()


class LearningRateSchedulerCallback(tf.keras.callbacks.Callback):
    """
    Custom learning rate scheduler with multiple strategies.
    """
    
    def __init__(self,
                 strategy: str = 'exponential',
                 initial_lr: float = 0.001,
                 decay_rate: float = 0.95,
                 decay_steps: int = 1,
                 min_lr: float = 1e-6,
                 patience: int = 3,
                 factor: float = 0.5,
                 monitor: str = 'val_loss',
                 verbose: int = 1):
        """
        Initialize learning rate scheduler.
        
        Args:
            strategy: Scheduling strategy ('exponential', 'step', 'plateau')
            initial_lr: Initial learning rate
            decay_rate: Decay rate for exponential/step decay
            decay_steps: Steps between rate updates
            min_lr: Minimum learning rate
            patience: Epochs to wait before reducing LR (for plateau)
            factor: Factor to reduce LR by (for plateau)
            monitor: Metric to monitor (for plateau)
            verbose: Verbosity level
        """
        super().__init__()
        
        self.strategy = strategy
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.monitor = monitor
        self.verbose = verbose
        
        # For plateau strategy
        self.wait = 0
        self.best = np.inf if 'loss' in monitor else -np.inf
        self.monitor_op = np.less if 'loss' in monitor else np.greater
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Update learning rate based on strategy."""
        if logs is None:
            return
            
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        if self.strategy == 'exponential':
            new_lr = self.initial_lr * (self.decay_rate ** (epoch // self.decay_steps))
        elif self.strategy == 'step':
            new_lr = current_lr * self.decay_rate if (epoch + 1) % self.decay_steps == 0 else current_lr
        elif self.strategy == 'plateau':
            current_metric = logs.get(self.monitor)
            if current_metric is None:
                return
                
            if self.monitor_op(current_metric, self.best):
                self.best = current_metric
                self.wait = 0
            else:
                self.wait += 1
                
            new_lr = current_lr
            if self.wait >= self.patience:
                new_lr = current_lr * self.factor
                self.wait = 0
                if self.verbose > 0:
                    print(f"Reducing learning rate to {new_lr:.2e}")
        else:
            return
        
        # Apply minimum learning rate constraint
        new_lr = max(new_lr, self.min_lr)
        
        # Update learning rate if changed
        if abs(new_lr - current_lr) > 1e-8:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            
            if self.verbose > 0 and self.strategy != 'plateau':
                print(f"Learning rate updated: {current_lr:.2e} -> {new_lr:.2e}")


def create_default_callbacks(save_path: str,
                            monitor: str = 'val_auc',
                            patience: int = 5,
                            verbose: int = 1) -> list:
    """
    Create default set of callbacks for training.
    
    Args:
        save_path: Path to save model checkpoints
        monitor: Metric to monitor
        patience: Early stopping patience
        verbose: Verbosity level
        
    Returns:
        List of callback instances
    """
    callbacks = [
        EarlyStoppingCallback(
            monitor=monitor,
            patience=patience,
            mode='max' if 'auc' in monitor else 'min',
            verbose=verbose
        ),
        ModelCheckpointCallback(
            filepath=os.path.join(save_path, 'best_model'),
            monitor=monitor,
            mode='max' if 'auc' in monitor else 'min',
            verbose=verbose
        )
    ]
    
    # Add learning rate scheduler
    callbacks.append(
        LearningRateSchedulerCallback(
            strategy='plateau',
            monitor='val_loss',
            patience=patience // 2,
            verbose=verbose
        )
    )
    
    return callbacks
