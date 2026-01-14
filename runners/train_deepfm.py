import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple
import json
from time import time
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DeepFM, DeepFMTrainer
from dataprocess import DataConfig, CriteoConfig, AvazuConfig, KDD2012Config, CriteoDiscConfig, AliccpConfig
from utils import DataLoader, print_model_profile
from utils import MetricsCalculator


class DeepFMExperiment:
    """
    Main experiment class for DeepFM training and evaluation using Keras fit.
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize experiment with arguments.
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.config = self._create_data_config()
        self.model = None
        self.trainer = None
        self.data_loader = None
        
        # Setup logging and saving
        self.save_path = args.save_path
        if args.is_save and not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            
        # Initialize metrics tracking
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
    
    def _create_model(self, feature_size: int, field_size: int) -> DeepFM:
        """
        Create DeepFM model with specified parameters.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields
            
        Returns:
            Configured DeepFM model
        """
        model = DeepFM(
            feature_size=feature_size,
            field_size=field_size,
            embedding_size=self.args.embedding_size,
            deep_layers=self.args.deep_layers,
            dropout_rates=self.args.dropout_rates,
            l2_reg=self.args.l2_reg,
            use_batch_norm=self.args.batch_norm > 0
        )
        
        return model
    
    def _create_data_loader(self) -> DataLoader:
        """
        Create data loader for the experiment.
        
        Returns:
            Configured data loader
        """
        return DataLoader(
            config=self.config,
            batch_size=self.args.batch_size,
            buffer_size=10000,
            prefetch_size=tf.data.AUTOTUNE
        )
    
    def _get_file_names(self) -> Tuple[str, str, str]:
        """Get file names based on dataset type."""
        if self.args.data.lower() in ['criteo', 'kdd2012']:
            # Use scaled data for Criteo and KDD2012
            return (self.config.train_i_npy, self.config.train_x2_npy, self.config.train_y_npy)
        else:
            # Use original data for Avazu  
            return (self.config.train_i_npy, self.config.train_x_npy, self.config.train_y_npy)
    
    def run_single_experiment(self, run_id: int) -> Tuple[float, float]:
        """
        Run a single training experiment.
        
        Args:
            run_id: Experiment run identifier
            
        Returns:
            Tuple of (test_auc, test_logloss)
        """
        # Get feature size and field size
        assert self.data_loader is not None, "Data loader must be initialized before running experiment"
        feature_size = self.data_loader.get_feature_size()
        field_size = self.data_loader.get_field_size()
        
        # Create model and trainer
        self.model = self._create_model(feature_size, field_size)

        # Print model FLOPs
        if self.args.verbose > 0:
            print("\nProfiling model...")
            try:
                print_model_profile(self.model, batch_size=self.args.batch_size)
            except Exception as e:
                print(f"Warning: Could not profile model - {str(e)}")

        trainer_save_path = os.path.join(self.save_path, str(run_id)) if self.args.is_save else None
        self.trainer = DeepFMTrainer(
            model=self.model,
            optimizer=self.args.optimizer_type,
            learning_rate=self.args.learning_rate,
            learning_rate_wide=self.args.learning_rate_wide,
            patience=3,  # Early stopping patience
            save_path=trainer_save_path,
            verbose=self.args.verbose
        )
        
        # Get file names based on dataset
        file_names = self._get_file_names()
        
        # Load validation data (part 1)
        val_dataset = self.data_loader.load_fold_dataset(1, file_names)
        
        # Training loop
        for epoch in range(self.args.epoch):
            epoch_start_time = time()
            continue_training = True
            
            # Simple tqdm progress bar for parts
            parts_range = range(2, 11)
            if self.args.verbose > 0:
                parts_iter = tqdm(parts_range, 
                                desc=f"Epochs {epoch + 1}/{self.args.epoch}=> Parts", 
                                unit="part",
                                leave=False)
            else:
                parts_iter = parts_range
            
            for part_id in parts_iter:
                if not continue_training:
                    break
                
                # Load training data for this part
                try:
                    train_dataset = self.data_loader.load_fold_dataset(part_id, file_names)
                    
                    # Train for one epoch on this data part using Keras fit
                    continue_training = self.trainer.fit(
                        train_dataset=train_dataset,
                        validation_dataset=val_dataset,
                        epochs=1,
                        steps_per_epoch=None
                    )
                    
                    # Check if early stopping was triggered
                    if not continue_training:
                        if self.args.verbose > 0:
                            print(f"\nEarly stopping triggered at epoch {epoch + 1}, part {part_id}")
                        break
                        
                except FileNotFoundError:
                    if self.args.verbose > 1:
                        print(f"\nPart {part_id} not found, skipping...")
                    continue
            
            epoch_time = time() - epoch_start_time
            if self.args.verbose > 0:
                print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
            
            if not continue_training:
                break
        
        # Testing phase
        if self.args.verbose > 0:
            print("\nStarting testing...")
        
        # Load best model weights
        if self.trainer.load_best_weights():
            if self.args.verbose > 0:
                print("Loaded best model weights")
        
        # Load test data (part 1)
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
        """
        Run multiple experiments and return aggregated results.
        
        Returns:
            Dictionary containing experiment results
        """
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
    parser = argparse.ArgumentParser(description='DeepFM Training')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='criteo', help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data/', help='Data directory')
    
    # Model parameters
    parser.add_argument('--embedding_size', type=int, default=16, help='Embedding size')
    parser.add_argument('--deep_layers', type=int, nargs='+', default=[256, 128, 64], 
                        help='Deep network hidden layer sizes')
    parser.add_argument('--dropout_rates', type=float, nargs='+', default=[0.0, 0.1],
                        help='Dropout rates [embedding_dropout, deep_dropout]')
    parser.add_argument('--batch_norm', type=int, default=0, help='Use batch normalization')
    parser.add_argument('--l2_reg', type=float, default=0.0, help='L2 regularization')
    
    # Training parameters
    parser.add_argument('--optimizer_type', type=str, default='adam', help='Optimizer type')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--learning_rate_wide', type=float, default=0.001, help='Learning rate for wide part')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    
    # Other parameters
    parser.add_argument('--save_path', type=str, default='../checkpoint/deepfm', help='Save path')
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
    print("DeepFM Training Configuration")
    print("="*80)
    print(f"Dataset: {args.data}")
    print(f"Embedding Size: {args.embedding_size}")
    print(f"Deep Layers: {args.deep_layers}")
    print(f"Dropout Rates: {args.dropout_rates}")
    print(f"Batch Normalization: {bool(args.batch_norm)}")
    print(f"L2 Regularization: {args.l2_reg}")
    print(f"Optimizer: {args.optimizer_type}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epoch}")
    print(f"Runs: {args.runs}")
    print(f"Save Path: {args.save_path}")
    print("="*80 + "\n")
    
    # Run experiment
    experiment = DeepFMExperiment(args)
    results = experiment.run_experiments()
    
    return results


if __name__ == '__main__':
    main()
