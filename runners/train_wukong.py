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

from models import Wukong, WukongTrainer
from dataprocess import DataConfig, CriteoConfig, AvazuConfig, KDD2012Config, CriteoDiscConfig, AliccpConfig
from utils import DataLoader, print_model_profile
from utils import MetricsCalculator


class WukongExperiment:
    """
    Main experiment class for WuKong training and evaluation.
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
                data_path='./data/Criteo_disc',
                verbose=self.args.verbose
            )
        elif self.args.data.lower() == 'avazu':
            return AvazuConfig(
                data_path=os.path.join(self.args.data_path, 'Avazu'),
                verbose=self.args.verbose
            )
        elif self.args.data.lower() == 'kdd2012':
            return KDD2012Config(
                data_path=os.path.join(self.args.data_path, 'KDD2012'),
                verbose=self.args.verbose
            )
        elif self.args.data.lower() == 'aliccp':
            return AliccpConfig(
                data_path=os.path.join(self.args.data_path, 'Aliccp'),
                verbose=self.args.verbose
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.args.data}")    
    def _create_model(self, feature_size: int, field_size: int) -> Wukong:
        """
        Create WuKong model with specified parameters.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields
            
        Returns:
            Configured WuKong model
        """
        model = Wukong(
            feature_size=feature_size,
            field_size=field_size,
            embedding_size=self.args.embedding_size,
            num_wukong_layers=self.args.num_wukong_layers,
            lcb_features=self.args.lcb_features,
            fmb_features=self.args.fmb_features,
            fmb_mlp_units=self.args.fmb_mlp_units,
            fmb_mlp_activations=self.args.fmb_mlp_activations,
            fmp_rank_k=self.args.fmp_rank_k,
            mlp_hidden_units=self.args.mlp_hidden_units,
            mlp_hidden_activations=self.args.mlp_hidden_activations,
            mlp_batch_norm=self.args.mlp_batch_norm > 0,
            layer_norm=self.args.layer_norm > 0,
            net_dropout=self.args.dropout_rate,
            l2_reg_embedding=self.args.l2_reg_embedding,
            l2_reg_deep=self.args.l2_reg_deep
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
            # Use scaled data for Criteo, Criteo_disc and KDD2012
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

        # Print model profile
        if self.args.verbose > 0:
            print("\nProfiling model...")
            try:
                print_model_profile(self.model, batch_size=self.args.batch_size)
            except Exception as e:
                print(f"Warning: Could not profile model - {str(e)}")
        
        trainer_save_path = os.path.join(self.save_path, str(run_id)) if self.args.is_save else None
        self.trainer = WukongTrainer(
            model=self.model,
            optimizer=self.args.optimizer_type,
            learning_rate=self.args.learning_rate,
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
                    
                    # Train for one epoch on this data part
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
        
        if self.args.verbose > 0:
            print(f"\nStarting {self.args.run_times} experiment(s)...")
        
        all_test_aucs = []
        all_test_losses = []
        
        for run_id in range(self.args.run_times):
            if self.args.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Run {run_id + 1}/{self.args.run_times}")
                print(f"{'='*60}")
            
            test_auc, test_loss = self.run_single_experiment(run_id)
            all_test_aucs.append(test_auc)
            all_test_losses.append(test_loss)
            
            self.results['test_results'].append({
                'run_id': run_id,
                'test_auc': float(test_auc),
                'test_loss': float(test_loss)
            })
        
        # Calculate statistics
        mean_auc = np.mean(all_test_aucs)
        std_auc = np.std(all_test_aucs)
        mean_loss = np.mean(all_test_losses)
        std_loss = np.std(all_test_losses)
        
        if self.args.verbose > 0:
            print(f"\n{'='*60}")
            print("Final Results Summary")
            print(f"{'='*60}")
            print(f"Test AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
            print(f"Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
            print(f"{'='*60}")
        
        # Save results
        if self.args.is_save:
            results_file = os.path.join(self.save_path, 'results.json')
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            if self.args.verbose > 0:
                print(f"Results saved to {results_file}")
        
        return {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'all_results': self.results
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train WuKong model for CTR prediction')
    
    # Dataset arguments
    parser.add_argument('--data', type=str, default='criteo',
                        choices=['criteo', 'avazu', 'kdd2012', 'criteo_disc', 'aliccp'],
                        help='Dataset name (criteo, avazu, kdd2012, criteo_disc, aliccp)')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='Path to data directory')
    
    # Model arguments - Embedding
    
    # Model arguments - Embedding
    parser.add_argument('--embedding_size', type=int, default=16,
                        help='Embedding dimension')
    
    # Model arguments - WuKong layers
    parser.add_argument('--num_wukong_layers', type=int, default=3,
                        help='Number of WuKong layers')
    parser.add_argument('--lcb_features', type=int, default=40,
                        help='Number of LCB features')
    parser.add_argument('--fmb_features', type=int, default=40,
                        help='Number of FMB features')
    parser.add_argument('--fmb_mlp_units', type=int, nargs='+', default=[32, 32],
                        help='Hidden units for FMB MLP')
    parser.add_argument('--fmb_mlp_activations', type=str, default='relu',
                        help='Activation for FMB MLP')
    parser.add_argument('--fmp_rank_k', type=int, default=8,
                        help='Rank for FM projection')
    
    # Model arguments - Output MLP
    parser.add_argument('--mlp_hidden_units', type=int, nargs='+', default=[32, 32],
                        help='Hidden units for output MLP')
    parser.add_argument('--mlp_hidden_activations', type=str, default='relu',
                        help='Activation for output MLP')
    parser.add_argument('--mlp_batch_norm', type=int, default=1,
                        help='Whether to use batch normalization in output MLP (0/1)')
    parser.add_argument('--layer_norm', type=int, default=1,
                        help='Whether to use layer normalization (0/1)')
    
    # Regularization arguments
    parser.add_argument('--l2_reg_embedding', type=float, default=0.0,
                        help='L2 regularization for embeddings')
    parser.add_argument('--l2_reg_deep', type=float, default=0.0,
                        help='L2 regularization for deep network')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--epoch', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer_type', type=str, default='adam',
                        help='Optimizer type (adam, rmsprop, sgd)')
    
    # Experiment arguments
    parser.add_argument('--run_times', type=int, default=1,
                        help='Number of experiment runs')
    parser.add_argument('--save_path', type=str, default='./checkpoint/wukong_experiment/',
                        help='Path to save model checkpoints')
    parser.add_argument('--is_save', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to save model')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0, 1, 2)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Print configuration
    if args.verbose > 0:
        print("\n" + "="*60)
        print("WuKong Training Configuration")
        print("="*60)
        print(f"Dataset: {args.data}")
        print(f"Data path: {args.data_path}")
        print(f"\nModel Architecture:")
        print(f"  Embedding size: {args.embedding_size}")
        print(f"  Number of WuKong layers: {args.num_wukong_layers}")
        print(f"  LCB features: {args.lcb_features}")
        print(f"  FMB features: {args.fmb_features}")
        print(f"  FMB MLP units: {args.fmb_mlp_units}")
        print(f"  FMP rank k: {args.fmp_rank_k}")
        print(f"  Output MLP units: {args.mlp_hidden_units}")
        print(f"  Layer normalization: {args.layer_norm > 0}")
        print(f"  MLP batch normalization: {args.mlp_batch_norm > 0}")
        print(f"\nTraining:")
        print(f"  Dropout rate: {args.dropout_rate}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Epochs: {args.epoch}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Optimizer: {args.optimizer_type}")
        print(f"  L2 reg (embedding): {args.l2_reg_embedding}")
        print(f"  L2 reg (deep): {args.l2_reg_deep}")
        print(f"\nExperiment:")
        print(f"  Run times: {args.run_times}")
        print(f"  Save path: {args.save_path}")
        print("="*60 + "\n")
    
    # Run experiments
    experiment = WukongExperiment(args)
    results = experiment.run_experiments()
    
    return results


if __name__ == '__main__':
    main()
