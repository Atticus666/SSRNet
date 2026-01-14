import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple
import json
from time import time
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DCNV2, DCNV2Trainer
from dataprocess import DataConfig, CriteoConfig, AvazuConfig, KDD2012Config, CriteoDiscConfig, AliccpConfig
from utils import DataLoader, print_model_profile
from utils import MetricsCalculator

class DCNV2Experiment:
    """
    Main experiment class for DCN v2 training and evaluation.
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
    def _create_model(self, feature_size: int, field_size: int) -> DCNV2:
        """
        Create DCN V2 model with specified parameters.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields
            
        Returns:
            Configured DCN V2 model
        """
        model = DCNV2(
            feature_size=feature_size,
            field_size=field_size,
            embedding_size=self.args.embedding_size,
            cross_layers=self.args.cross_layers,
            deep_layers=self.args.deep_layers,
            is_stacked=self.args.is_stacked,
            use_low_rank_mixture=self.args.use_low_rank_mixture,
            low_rank=self.args.low_rank,
            num_experts=self.args.num_experts,
            l2_reg_embedding=self.args.l2_reg_embedding,
            l2_reg_cross=self.args.l2_reg_cross,
            l2_reg_deep=self.args.l2_reg_deep,
            dropout_rate=self.args.dropout_rate,
            use_batch_norm=bool(self.args.batch_norm)
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

        if self.args.verbose > 0:
            print("\nProfiling model...")
            try:
                print_model_profile(self.model, batch_size=self.args.batch_size)
            except Exception as e:
                print(f"Warning: Could not profile model - {str(e)}")
        
        trainer_save_path = os.path.join(self.save_path, str(run_id)) if self.args.is_save else None
        self.trainer = DCNV2Trainer(
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
                    self.trainer.fit(
                        train_dataset=train_dataset,
                        validation_dataset=val_dataset,
                        epochs=1,
                        steps_per_epoch=None
                    )
                    
                    # Check if early stopping was triggered
                    if self.trainer.wait >= self.trainer.patience:
                        continue_training = False
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
        
        test_aucs = []
        test_losses = []
        
        # Run multiple experiments
        for run_id in range(1, self.args.run_times + 1):
            if self.args.verbose > 0:
                print(f"\n{'='*50}")
                print(f"Run {run_id}/{self.args.run_times}")
                print(f"{'='*50}")
            
            try:
                test_auc, test_loss = self.run_single_experiment(run_id)
                test_aucs.append(test_auc)
                test_losses.append(test_loss)
                
                if self.args.verbose > 0:
                    print(f"Run {run_id} - AUC: {test_auc:.4f}, LogLoss: {test_loss:.4f}")
            
            except Exception as e:
                print(f"Run {run_id} failed with error: {e}")
                continue
        
        # Calculate statistics
        if test_aucs:
            results = {
                'test_aucs': test_aucs,
                'test_losses': test_losses,
                'avg_auc': np.mean(test_aucs) if test_aucs else 0.0,
                'std_auc': np.std(test_aucs) if test_aucs else 0.0,
                'avg_loss': np.mean(test_losses) if test_losses else float('inf'),
                'std_loss': np.std(test_losses) if test_losses else 0.0,
                'args': vars(self.args)
            }
            
            if self.args.verbose > 0:
                print(f"\n{'='*60}")
                print("FINAL RESULTS")
                print(f"{'='*60}")
                print(f"Test AUC: {results['avg_auc']:.4f} ± {results['std_auc']:.4f}")
                print(f"Test LogLoss: {results['avg_loss']:.4f} ± {results['std_loss']:.4f}")
                print(f"Successful runs: {len(test_aucs)}/{self.args.run_times}")
        else:
            results = {'error': 'All runs failed'}
            
        return results

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DCN v2 Training Script')
    
    # Dataset parameters
    parser.add_argument('--data', type=str, required=True,
                       choices=['criteo', 'avazu', 'kdd2012', 'criteo_disc', 'aliccp'],
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data/',
                       help='Path to data directory')
    
    # Model architecture parameters
    parser.add_argument('--embedding_size', type=int, default=16,
                       help='Embedding dimension')
    parser.add_argument('--cross_layers', type=int, default=3,
                       help='Number of cross layers')
    parser.add_argument('--deep_layers', nargs='+', type=int,
                       default=[256, 256],
                       help='Deep network hidden layer dimensions')
    parser.add_argument('--is_stacked', action='store_true',
                       help='Use stacked architecture instead of parallel')
    parser.add_argument('--use_low_rank_mixture', action='store_true',
                       help='Use low-rank mixture of experts in cross network')
    parser.add_argument('--low_rank', type=int, default=32,
                       help='Low-rank dimension for mixture of experts')
    parser.add_argument('--num_experts', type=int, default=4,
                       help='Number of expert networks in MOE')
    
    # Training parameters
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--epoch', type=int, default=15,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer_type', type=str, default='adam',
                       choices=['adam', 'rmsprop', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--l2_reg_embedding', type=float, default=1e-6,
                       help='L2 regularization for embeddings')
    parser.add_argument('--l2_reg_cross', type=float, default=1e-6,
                       help='L2 regularization for cross network')
    parser.add_argument('--l2_reg_deep', type=float, default=1e-6,
                       help='L2 regularization for deep network')
    
    # Training configuration
    parser.add_argument('--batch_norm', type=int, default=0,
                       help='Whether to use batch normalization')
    parser.add_argument('--random_seed', type=int, default=2018, help='Random seed')
    parser.add_argument('--run_times', type=int, default=1,
                       help='Number of experiment runs')
    
    # Saving and logging
    parser.add_argument('--save_path', type=str, default='./logs',
                       help='Path to save logs and models')
    parser.add_argument('--is_save', type=str, default='False', help='Whether to save models')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')
    
    args = parser.parse_args()
    
    # Process string arguments - convert to appropriate types
    args.is_save = args.is_save.lower() in ['true', 't', '1', 'yes']
    
    return args

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    if args.verbose > 0:
        print("DCN v2 Training Configuration:")
        print("=" * 50)
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        print("=" * 50)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Setup GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("Using CPU")
    
    # Create and run experiment
    experiment = DCNV2Experiment(args)
    results = experiment.run_experiments()
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test AUCs: {results['test_aucs']}")
    print(f"Test Losses: {results['test_losses']}")
    print(f"Average AUC: {results['avg_auc']:.4f} ± {results['std_auc']:.4f}")
    print(f"Average Loss: {results['avg_loss']:.4f} ± {results['std_loss']:.4f}")
    print("=" * 60)
    
    # Save results
    if args.is_save:
        results_path = os.path.join(args.save_path, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")

if __name__ == '__main__':
    main()