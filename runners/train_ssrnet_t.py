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

from models.model_ssrnet_t import SSRNetT, SSRNetTTrainer
from dataprocess import DataConfig, CriteoConfig, AvazuConfig, KDD2012Config, CriteoDiscConfig, AliccpConfig
from utils import DataLoader, print_model_profile
from utils import MetricsCalculator

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*24)]  # 24GB
        )
        print(f"Found {len(gpus)} GPU(s), memory growth with limitation enabled")

    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


class SSRNetTExperiment:
    """
    Main experiment class for SSRNet-T training and evaluation using Keras fit.
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
    
    def _create_model(self, feature_size: int, field_size: int) -> SSRNetT:
        """
        Create SSRNetT model with specified parameters.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields
            
        Returns:
            Configured SSRNetT model
        """
        model = SSRNetT(
            feature_size=feature_size,
            field_size=field_size,
            embedding_size=self.args.embedding_size,
            tokennum_list=self.args.tokennum_list,
            hidden_unit_list=self.args.hidden_unit_list,
            top_k_list=self.args.top_k_list,
            out_unit_list=self.args.out_unit_list,
            alpha_inits=self.args.alpha_inits,
            scale_inits=self.args.scale_inits,
            has_wide=self.args.has_wide > 0,
            deep_layers=self.args.deep_layers,
            dropout_rates=self.args.dropout_rates,
            l2_reg=self.args.l2_reg,
            use_batch_norm=self.args.batch_norm > 0,
            use_ssr_linear=self.args.use_ssr_linear,
            use_block_mean_pooling=self.args.use_block_mean_pooling,
            use_ssrblock_residual=self.args.use_ssrblock_residual,
            use_block_ln=self.args.use_block_ln,
            use_gate=self.args.use_gate,
            use_block_dense=self.args.use_block_dense,
            iterations=self.args.iterations,
            block_version=self.args.block_version
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
        self.trainer = SSRNetTTrainer(
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
        
        # Close monitoring callback's file writer after all training is complete
        self.trainer.monitoring_callback.close()
        
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
        Run multiple experiments and aggregate results.
        
        Returns:
            Dictionary with aggregated results
        """
        # Initialize data loader
        self.data_loader = self._create_data_loader()
        
        test_aucs = []
        test_loglosses = []
        
        for run_id in range(1, self.args.num_runs + 1):
            if self.args.verbose > 0:
                print(f"\n{'='*50}")
                print(f"Starting experiment run {run_id}/{self.args.num_runs}")
                print(f"{'='*50}\n")
            
            test_auc, test_logloss = self.run_single_experiment(run_id)
            test_aucs.append(test_auc)
            test_loglosses.append(test_logloss)
            
            if self.args.verbose > 0:
                print(f"Run {run_id} completed - AUC: {test_auc:.4f}, Loss: {test_logloss:.4f}")
        
        # Aggregate results
        self.results['test_aucs'] = test_aucs
        self.results['test_losses'] = test_loglosses
        self.results['mean_auc'] = np.mean(test_aucs)
        self.results['std_auc'] = np.std(test_aucs)
        self.results['mean_loss'] = np.mean(test_loglosses)
        self.results['std_loss'] = np.std(test_loglosses)
        
        # Save results
        if self.args.is_save:
            results_path = os.path.join(self.save_path, 'results.json')
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            if self.args.verbose > 0:
                print(f"\nResults saved to: {results_path}")
        
        return self.results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SSRNet-T model")
    
    # Data parameters
    parser.add_argument('--data', type=str, default='criteo',
                        choices=['criteo', 'avazu', 'kdd2012', 'criteo_disc', 'aliccp'],
                        help='Dataset name: criteo, avazu, kdd2012, criteo_disc, aliccp')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='Path to data directory')
    
    # Model parameters
    parser.add_argument('--block_version', type=str, default="t1", help='Block version')
    parser.add_argument('--use_ssr_linear', type=lambda x: x.lower() == 'true', default=False,
                        help='Whether to use SSR-Linear (True or False)')
    parser.add_argument('--use_block_mean_pooling', type=lambda x: x.lower() == 'true', default=False,
                        help='Whether to use block mean pooling (True or False)')
    parser.add_argument('--use_ssrblock_residual', type=lambda x: x.lower() == 'true', default=False,
                        help='Whether to use SSR-Block residual (True or False)')
    parser.add_argument('--use_block_ln', type=lambda x: x.lower() == 'true', default=False,
                        help='Whether to use block LN (True or False)')
    parser.add_argument('--use_gate', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to use gate (True or False)')
    parser.add_argument('--embedding_size', type=int, default=16,
                        help='Embedding dimension')
    parser.add_argument('--tokennum_list', type=int, nargs='+', default=[8, 8],
                        help='List of token numbers for each SSR-T block')
    parser.add_argument('--hidden_unit_list', type=int, nargs='+', default=[64, 128],
                        help='List of hidden units for each SSR-T block')
    parser.add_argument('--top_k_list', type=int, nargs='+', default=[64, 128],
                        help='List of top-k values for each SSR-T block')
    parser.add_argument('--out_unit_list', type=int, nargs='+', default=[64, 128],
                        help='List of output units for each SSR-T block')
    parser.add_argument('--use_block_dense', type=lambda x: x.lower() == 'true', default=False,
                        help='Whether to use block dense (True or False)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations')
    parser.add_argument('--alpha_inits', type=float, nargs='+', default=[0.0, 0.0],
                        help='Initial values for alpha')
    parser.add_argument("--scale_inits", type=float, nargs='+', default=[1.0, 1.0],
                        help='Initial values for scale')
    parser.add_argument('--has_wide', type=int, default=0,
                        help='Whether to use wide (linear) part (1: yes, 0: no)')
    parser.add_argument('--deep_layers', type=int, nargs='*', default=None,
                        help='Hidden dimensions for deep network')
    parser.add_argument('--dropout_rates', type=float, nargs='+', default=[0.0, 0.0, 0.1],
                        help='Dropout rates [attention, embedding, deep]')
    parser.add_argument('--l2_reg', type=float, default=0.0,
                        help='L2 regularization coefficient')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to use batch normalization (1: yes, 0: no)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--epoch', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--optimizer_type', type=str, default='adam',
                        help='Optimizer type: adam, sgd, rmsprop')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--learning_rate_wide', type=float, default=0.001,
                        help='Learning rate for wide part (kept for compatibility)')
    
    # Other parameters
    parser.add_argument('--random_seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--save_path', type=str, default='./checkpoint/ssrnet_t',
                        help='Path to save model checkpoints')
    parser.add_argument('--is_save', type=int, default=1,
                        help='Whether to save model (1: yes, 0: no)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of experimental runs')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Print configuration
    if args.verbose > 0:
        print("\n" + "="*60)
        print("SSRNet-T Training Configuration")
        print("="*60)
        print(f"Dataset: {args.data}")
        print(f"Data path: {args.data_path}")
        print(f"Batch size: {args.batch_size}")
        print(f"Block version: {args.block_version}")
        print(f"Use SSR-Linear: {args.use_ssr_linear}")
        print(f"Use block mean pooling: {args.use_block_mean_pooling}")
        print(f"Use SSR-Block residual: {args.use_ssrblock_residual}")
        print(f"Use block LN: {args.use_block_ln}")
        print(f"Use gate: {args.use_gate}")
        print(f"Iterations: {args.iterations}")
        print(f"Embedding size: {args.embedding_size}")
        print(f"Token number list: {args.tokennum_list}")
        print(f"Hidden unit list: {args.hidden_unit_list}")
        print(f"Top-k list: {args.top_k_list}")
        print(f"Use block dense: {args.use_block_dense}")
        print(f"Output unit list: {args.out_unit_list}")
        print(f"Alpha inits: {args.alpha_inits}")
        print(f"Scale inits: {args.scale_inits}")
        print(f"Has wide: {args.has_wide}")
        print(f"Deep layers: {args.deep_layers}")
        print(f"Dropout rates: {args.dropout_rates}")
        print(f"Epochs: {args.epoch}")
        print(f"Optimizer: {args.optimizer_type}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"L2 regularization: {args.l2_reg}")
        print(f"Random seed: {args.random_seed}")
        print(f"Number of runs: {args.num_runs}")
        print("="*60 + "\n")
    
    # Create experiment
    experiment = SSRNetTExperiment(args)
    
    # Run experiments
    results = experiment.run_experiments()
    
    # Print final results
    if args.verbose > 0:
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Test AUCs: {results['test_aucs']}")
        print(f"Test Losses: {results['test_losses']}")
        print(f"Average AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
        print(f"Average Loss: {results['mean_loss']:.4f} ± {results['std_loss']:.4f}")
        print("="*60)


if __name__ == '__main__':
    main()
