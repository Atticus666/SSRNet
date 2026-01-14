import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple
import json
from time import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_autofis import AutoFISTwoStageTrainer
from dataprocess import DataConfig, CriteoConfig, AvazuConfig, KDD2012Config, CriteoDiscConfig, AliccpConfig
from utils import DataLoader, print_model_profile
from utils import MetricsCalculator


class AutoFISExperiment:

    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = self._create_data_config()
        self.trainer = None
        self.data_loader = None
        
        # Setup logging and saving with unique path (timestamp + PID)
        if args.is_save:
            # Generate unique path with timestamp and process ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pid = os.getpid()
            unique_suffix = f"{timestamp}_pid{pid}"
            self.save_path = os.path.join(args.save_path, unique_suffix)
            
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
            
            if args.verbose > 0:
                print(f"Save path: {self.save_path}")
        else:
            self.save_path = args.save_path
            
        self.metrics_calculator = MetricsCalculator()
        self.results = {
            'train_history': [],
            'val_history': [], 
            'test_results': []
        }
        
    def _create_data_config(self) -> DataConfig:
        """Create appropriate data configuration based on dataset."""
        if self.args.data.lower() == 'criteo':
            return CriteoConfig(
                data_path=os.path.join(self.args.data_path, 'Criteo'),
                verbose=self.args.verbose
            )
        elif self.args.data.lower() == 'criteo_disc':
            return CriteoDiscConfig(
                data_path=os.path.join(self.args.data_path, 'Criteo_disc'),
                verbose=self.args.verbose
            )
        elif self.args.data.lower() == 'avazu':
            return AvazuConfig(
                data_path=os.path.join(self.args.data_path, 'Avazu'),
                verbose=self.args.verbose
            )
        elif self.args.data.lower() == 'aliccp':
            return AliccpConfig(
                data_path=os.path.join(self.args.data_path, 'Aliccp'),
                verbose=self.args.verbose
            )
    
    def _create_data_loader(self) -> DataLoader:
        """Create data loader for the experiment."""
        return DataLoader(
            config=self.config,
            batch_size=self.args.batch_size,
            buffer_size=10000,
            prefetch_size=tf.data.AUTOTUNE
        )
    
    def _get_file_names(self) -> Tuple[str, str, str]:
        """Get file names based on dataset type."""
        if self.args.data.lower() in ['criteo', 'kdd2012']:
            return (self.config.train_i_npy, self.config.train_x2_npy, self.config.train_y_npy)
        else:
            return (self.config.train_i_npy, self.config.train_x_npy, self.config.train_y_npy)
    
    def run_single_experiment(self, run_id: int) -> Tuple[float, float]:
        """Run a single training experiment."""
        # Get feature size and field size
        assert self.data_loader is not None
        feature_size = self.data_loader.get_feature_size()
        field_size = self.data_loader.get_field_size()
        
        # Create trainer
        trainer_save_path = os.path.join(self.save_path, str(run_id)) if self.args.is_save else None
        if trainer_save_path and not os.path.exists(trainer_save_path):
            os.makedirs(trainer_save_path, exist_ok=True)
        
        self.trainer = AutoFISTwoStageTrainer(
            feature_size=feature_size,
            field_size=field_size,
            embedding_size=self.args.embedding_size,
            mlp_width=self.args.mlp_width,
            mlp_depth=self.args.mlp_depth,
            use_bn=self.args.batch_norm > 0,
            dropout_rate=self.args.dropout,
            l2_reg=self.args.l2_reg,
            optimizer=self.args.optimizer_type,
            learning_rate=self.args.learning_rate,
            keep_ratio=self.args.keep_ratio,
            early_stop_patience=self.args.early_stop_patience,
            save_path=trainer_save_path,
            verbose=self.args.verbose
        )
        
        if self.args.verbose > 0:
            print("\nProfiling model...")
            try:
                print_model_profile(
                    self.trainer.model_stage1, 
                    batch_size=self.args.batch_size,
                    field_size=field_size
                )
            except Exception as e:
                print(f"Warning: Could not profile model - {str(e)}")
        
        # Get file names
        file_names = self._get_file_names()
        
        # Load validation data (part 1)
        val_dataset = self.data_loader.load_fold_dataset(1, file_names)
        
        # Two-stage training
        if self.args.verbose > 0:
            print("\n" + "="*80)
            print("Two-Stage Training for AutoFIS ")
            print("="*80)
            print(f"Using two independent models for stage1 and stage2")
            print(f"Stage 1 epochs: {self.args.stage1_epochs}")
            print(f"Stage 2 epochs: {self.args.stage2_epochs}")
            print(f"Keep ratio: {self.args.keep_ratio}")
            print("="*80 + "\n")
        
        # Stage 1: Train with learnable mask (part by part, no validation)
        if self.args.verbose > 0:
            print("Stage 1: Training with learnable interaction mask (no validation)")
        
        for epoch in range(self.args.stage1_epochs):
            if self.args.verbose > 0:
                print(f"\nStage 1 - Epoch {epoch + 1}/{self.args.stage1_epochs}")
            
            for part_id in range(2, 11):
                try:
                    if self.args.verbose > 0:
                        print(f"  Training on part {part_id}...")
                    train_dataset = self.data_loader.load_fold_dataset(part_id, file_names)
                    self.trainer.train_stage1(
                        train_dataset=train_dataset,
                        epochs=1,
                        steps_per_epoch=None
                    )
                except FileNotFoundError:
                    if self.args.verbose > 1:
                        print(f"  Part {part_id} not found, skipping...")
                    continue
        
        # Prune interactions
        self.trainer.prune_interactions()
        
        # Stage 2: Re-train with fixed mask (part by part with validation and early stopping)
        if self.args.verbose > 0:
            print("\nStage 2: Re-training with fixed interaction mask (with early stopping)")
        
        for epoch in range(self.args.stage2_epochs):
            if self.args.verbose > 0:
                print(f"\nStage 2 - Epoch {epoch + 1}/{self.args.stage2_epochs}")
            
            continue_training = True
            
            for part_id in range(2, 11):
                if not continue_training:
                    break
                
                try:
                    if self.args.verbose > 0:
                        print(f"  Training on part {part_id}...")
                    train_dataset = self.data_loader.load_fold_dataset(part_id, file_names)
                    history, val_auc, val_loss = self.trainer.train_stage2(
                        train_dataset=train_dataset,
                        validation_dataset=val_dataset,
                        epochs=1,
                        steps_per_epoch=None
                    )
                    
                    # Check early stopping after each part (returns True to continue, False to stop)
                    continue_training = self.trainer.check_early_stopping(val_auc)
                    
                    if not continue_training:
                        if self.args.verbose > 0:
                            print(f"\nEarly stopping triggered at epoch {epoch + 1}, part {part_id}")
                        break
                    
                except FileNotFoundError:
                    if self.args.verbose > 1:
                        print(f"  Part {part_id} not found, skipping...")
                    continue
            
            # Break outer loop if early stopping
            if not continue_training:
                break
        
        if self.args.verbose > 0:
            print("\n" + "="*80)
            print("Two-Stage Training Completed")
            print("="*80 + "\n")
        
        # Testing
        if self.args.verbose > 0:
            print("\nStarting testing...")
        
        # Load best model weights
        if self.trainer.load_best_weights():
            if self.args.verbose > 0:
                print("Loaded best model weights")
        
        try:
            test_dataset = self.data_loader.load_fold_dataset(1, file_names)
            test_loss, test_auc = self.trainer.evaluate(test_dataset)
            
            if self.args.verbose > 0:
                print(f"Test Results - AUC: {test_auc:.4f}, LogLoss: {test_loss:.4f}")
            
            return test_auc, test_loss
            
        except FileNotFoundError:
            print("Test data (part 1) not found!")
            return 0.0, float('inf')
    
    def run_experiments(self) -> Dict[str, Any]:
        """Run multiple experiments and return aggregated results."""
        # Initialize data loader
        self.data_loader = self._create_data_loader()
        
        auc_list = []
        logloss_list = []
        
        for run in range(self.args.runs):
            if self.args.verbose > 0:
                print("\n" + "="*80)
                print(f"Run {run + 1}/{self.args.runs}")
                print("="*80)
                print(f"feature_size: {self.data_loader.get_feature_size()}")
                print(f"field_size: {self.data_loader.get_field_size()}")
            
            auc, logloss = self.run_single_experiment(run)
            auc_list.append(auc)
            logloss_list.append(logloss)
            
            if self.args.verbose > 0:
                print(f"\nRun {run + 1} completed - AUC: {auc:.4f}, LogLoss: {logloss:.4f}")
        
        # Calculate statistics
        auc_mean = np.mean(auc_list)
        auc_std = np.std(auc_list)
        logloss_mean = np.mean(logloss_list)
        logloss_std = np.std(logloss_list)
        
        results = {
            'auc_mean': float(auc_mean),
            'auc_std': float(auc_std),
            'logloss_mean': float(logloss_mean),
            'logloss_std': float(logloss_std),
            'auc_list': [float(x) for x in auc_list],
            'logloss_list': [float(x) for x in logloss_list]
        }
        
        if self.args.verbose > 0:
            print("\n" + "="*80)
            print("Final Results:")
            print("="*80)
            print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
            print(f"LogLoss: {logloss_mean:.4f} ± {logloss_std:.4f}")
            print("="*80 + "\n")
        
        # Save results
        if self.args.is_save:
            results_path = os.path.join(self.save_path, 'results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {results_path}")
        
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AutoFIS  Training')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='criteo', help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data/', help='Data directory')
    
    # Model parameters
    parser.add_argument('--embedding_size', type=int, default=16, help='Embedding size')
    parser.add_argument('--mlp_width', type=int, default=256, help='MLP width')
    parser.add_argument('--mlp_depth', type=int, default=3, help='MLP depth')
    parser.add_argument('--batch_norm', type=int, default=1, help='Use batch normalization')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--l2_reg', type=float, default=0.0, help='L2 regularization')
    
    # Training parameters
    parser.add_argument('--optimizer_type', type=str, default='adam', help='Optimizer type')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--stage1_epochs', type=int, default=1, help='Stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, default=2, help='Stage 2 epochs')
    parser.add_argument('--keep_ratio', type=float, default=0.5, help='Keep ratio for pruning')
    parser.add_argument('--early_stop_patience', type=int, default=3, help='Early stopping patience (number of parts without improvement)')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    
    # Other parameters
    parser.add_argument('--save_path', type=str, default='../checkpoint/autofis', help='Save path')
    parser.add_argument('--is_save', type=int, default=1, help='Save model')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set GPU
    if args.gpu >= 0:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
            except RuntimeError as e:
                print(e)
    
    # Print configuration
    print("\n" + "="*80)
    print("AutoFIS  Training Configuration")
    print("="*80)
    print(f"Dataset: {args.data}")
    print(f"Embedding Size: {args.embedding_size}")
    print(f"MLP Width: {args.mlp_width}")
    print(f"MLP Depth: {args.mlp_depth}")
    print(f"Batch Normalization: {bool(args.batch_norm)}")
    print(f"Dropout Rate: {args.dropout}")
    print(f"L2 Regularization: {args.l2_reg}")
    print(f"Optimizer: {args.optimizer_type}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Runs: {args.runs}")
    print(f"Two-Stage Training: True")
    print(f"  Stage 1 Epochs: {args.stage1_epochs} (no validation)")
    print(f"  Stage 2 Epochs: {args.stage2_epochs} (with validation & early stopping)")
    print(f"  Keep Ratio: {args.keep_ratio}")
    print(f"  Early Stop Patience: {args.early_stop_patience}")
    print(f"Save Path: {args.save_path}")
    print("="*80 + "\n")
    
    # Run experiment
    experiment = AutoFISExperiment(args)
    results = experiment.run_experiments()
    
    return results


if __name__ == '__main__':
    main()
