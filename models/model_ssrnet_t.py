import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import roc_auc_score, log_loss
from typing import Optional, List, Tuple, Dict, Any
import os
import sys
from time import time
from tqdm import tqdm


from .ssrnet import (
    StructuralSparseBlockT18,
    StructuralSparseBlockT18a,
    StructuralSparseBlockT21
)

from .ssrnet.monitoring_callback import SSRNetMonitoringCallback

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class SSRNetT(keras.Model):
    def __init__(self,
                 feature_size: int,
                 field_size: int,
                 embedding_size: int = 16,
                 tokennum_list: List[int] = [8, 8],
                 hidden_unit_list: List[int] = [256, 256],
                 out_unit_list: List[int] = [128, 128],
                 top_k_list: List[int] = [128, 128],
                 iterations: int = 5,
                 alpha_inits: List[float] = [0.0, 0.0],
                 scale_inits: List[float] = [1.0, 1.0],
                 use_gate: bool = True,
                 use_block_dense: bool = False,
                 has_wide: bool = False,
                 deep_layers: Optional[List[int]] = None,
                 dropout_rates: Optional[List[float]] = None,
                 l2_reg: float = 0.0,
                 use_batch_norm: bool = False,
                 use_ssr_linear: bool = False,
                 use_block_mean_pooling: bool = False,
                 use_ssrblock_residual: bool = False,
                 block_version: str = "t1",
                 use_block_ln: bool = False,
                 **kwargs):
        """
        Initialize SSRNet-T model.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields
            embedding_size: Embedding dimension
            tokennum_list: List of token numbers for each SSR-T block
            hidden_unit_list: List of hidden units for each SSR-T block
            top_k_list: List of top-k values for each SSR-T block
            has_wide: Whether to include wide (linear) part
            deep_layers: Hidden dimensions for deep network
            dropout_rates: Dropout rates [unused, embedding, deep]
            l2_reg: L2 regularization coefficient
            use_batch_norm: Whether to use batch normalization
            block_version: Version of SSR-T block
        """
        super(SSRNetT, self).__init__(**kwargs)
        
        # Model configuration
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.tokennum_list = tokennum_list
        self.hidden_unit_list = hidden_unit_list
        self.out_unit_list = out_unit_list
        self.top_k_list = top_k_list
        self.has_wide = has_wide
        self.deep_layers = deep_layers or []
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        self.block_version = block_version
        self.m_samples_list = [256, 512]
        self.use_ssr_linear = use_ssr_linear
        self.alpha_inits = alpha_inits
        self.scale_inits = scale_inits
        self.use_gate = use_gate 
        self.iterations = iterations
        self.use_ssrblock_residual = use_ssrblock_residual
        self.use_block_mean_pooling = use_block_mean_pooling
        self.use_block_dense = use_block_dense
        self.use_block_ln = use_block_ln
        
        # Set default dropout rates
        if dropout_rates is None:
            self.dropout_rates = [0.0, 0.0, 0.1]  # [attention, embedding, deep]
        else:
            self.dropout_rates = dropout_rates
        
        # Build model components
        self._build_model()
    
    def _build_model(self):
        """Build model components."""
        # Embedding layers
        self.feature_embedding = layers.Embedding(
            input_dim=self.feature_size,
            output_dim=self.embedding_size,
            embeddings_regularizer=None,
            name='feature_embeddings'
        )
        
        self.embedding_bn = layers.BatchNormalization(name='embedding_bn')

        if self.has_wide:
            self.feature_bias = layers.Embedding(
                input_dim=self.feature_size,
                output_dim=1,
                name='feature_bias'
            )
        
        # Dropout layers
        self.embedding_dropout = layers.Dropout(self.dropout_rates[1])
        
        # SSR-T blocks (using loop for flexible multi-layer stacking)
        self.ssr_blocks = []
        self.block_lns = []
        for i in range(len(self.tokennum_list)):
            if i == 0:
                input_dim = self.field_size * self.embedding_size
            else:
                # Previous block output: tokennum * out_unit (not top_k!)
                if self.use_block_mean_pooling:
                    input_dim = self.out_unit_list[i - 1]
                else:
                    input_dim = self.tokennum_list[i - 1] * self.out_unit_list[i - 1]
            if self.block_version == "t18":
                ssr_block = StructuralSparseBlockT18(
                    len_embeddings=input_dim,
                    tokennum=self.tokennum_list[i],
                    hidden_unit=self.hidden_unit_list[i],
                    top_k=self.top_k_list[i],
                    out_unit=self.out_unit_list[i],
                    alpha_init=self.alpha_inits[i],
                    scale_init=self.scale_inits[i],
                    use_gate=self.use_gate,
                    use_block_dense=self.use_block_dense,
                    iterations=self.iterations,
                    dropout_rates=self.dropout_rates,
                    l2_reg=self.l2_reg,
                    stage_name=f"ssr_block_t_{i+1}"
                )  
            elif self.block_version == "t18a":
                ssr_block = StructuralSparseBlockT18a(
                    len_embeddings=input_dim,
                    tokennum=self.tokennum_list[i],
                    hidden_unit=self.hidden_unit_list[i],
                    top_k=self.top_k_list[i],
                    out_unit=self.out_unit_list[i],
                    alpha_init=self.alpha_inits[i],
                    scale_init=self.scale_inits[i],
                    use_gate=self.use_gate,
                    use_block_dense=self.use_block_dense,
                    iterations=self.iterations,
                    dropout_rates=self.dropout_rates,
                    l2_reg=self.l2_reg,
                    stage_name=f"ssr_block_t_{i+1}"
                )
            elif self.block_version == "t21":
                ssr_block = StructuralSparseBlockT21(
                    len_embeddings=input_dim,
                    tokennum=self.tokennum_list[i],
                    hidden_unit=self.hidden_unit_list[i],
                    top_k=self.top_k_list[i],
                    out_unit=self.out_unit_list[i],
                    alpha_init=self.alpha_inits[i],
                    scale_init=self.scale_inits[i],
                    use_gate=self.use_gate,
                    use_block_dense=self.use_block_dense,
                    iterations=self.iterations,
                    dropout_rates=self.dropout_rates,
                    l2_reg=self.l2_reg,
                    stage_name=f"ssr_block_t_{i+1}"
                )              
            else:
                ssr_block = StructuralSparseBlockT21(
                    len_embeddings=input_dim,
                    tokennum=self.tokennum_list[i],
                    hidden_unit=self.hidden_unit_list[i],
                    top_k=self.top_k_list[i],
                    out_unit=self.out_unit_list[i],
                    alpha_init=self.alpha_inits[i],
                    scale_init=self.scale_inits[i],
                    use_gate=self.use_gate,
                    use_block_dense=self.use_block_dense,
                    iterations=self.iterations,
                    dropout_rates=self.dropout_rates,
                    l2_reg=self.l2_reg,
                    stage_name=f"ssr_block_t_{i+1}"
                )
             
            self.ssr_blocks.append(ssr_block)
            self.block_lns.append(layers.LayerNormalization(name=f"block_ln_{i}", axis=-1, epsilon=1e-6))

            if self.use_ssr_linear:
                last_tokennum = self.tokennum_list[-1]
                last_out_unit = self.out_unit_list[-1]
                
                reduction_ratio = 4  # SENet compression ratio
                self.senet_fc1 = layers.Dense(
                    max(last_tokennum // reduction_ratio, 1),
                    activation='relu',
                    name='senet_fc1'
                )
                self.senet_fc2 = layers.Dense(
                    last_tokennum,
                    activation='sigmoid',
                    name='senet_fc2'
                )
    
        # Deep network (optional)
        if self.deep_layers:
            self.deep_network = []
            for i, units in enumerate(self.deep_layers):
                self.deep_network.append(layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
                    name=f'deep_layer_{i}'
                ))
                if self.use_batch_norm:
                    self.deep_network.append(layers.BatchNormalization(name=f'bn_{i}'))
                self.deep_network.append(layers.Dropout(self.dropout_rates[2], name=f'dropout_{i}'))
            
            self.deep_output = layers.Dense(1, name='deep_prediction')
        else:
            print("==========No deep network specified, using SSRNet-T only==========")
        
        # Final prediction layer
        self.prediction_layer = layers.Dense(1, activation='sigmoid', name='prediction')

    def call(self, inputs, training=None):

        feat_index = inputs['feat_index']  # [batch_size, field_size]
        feat_value = inputs['feat_value']  # [batch_size, field_size]
        
        # Embedding lookup
        embeddings = self.feature_embedding(feat_index)  # [batch_size, field_size, embedding_size]
        feat_value = tf.expand_dims(feat_value, axis=-1)  # [batch_size, field_size, 1]
        embeddings = embeddings * feat_value  # Element-wise multiplication
        embeddings = self.embedding_dropout(embeddings, training=training)
        
        # Wide part (linear)
        wide_output = None
        if self.has_wide:
            wide_logits = self.feature_bias(feat_index)  # [batch_size, field_size, 1]
            wide_output = tf.reduce_sum(wide_logits * feat_value, axis=1)  # [batch_size, 1]
        
        # SSR-T part (using loop for flexible multi-layer stacking)
        ssr_hidden = self.embedding_bn(embeddings, training=training)
        for idx, ssr_block in enumerate(self.ssr_blocks):
            ssr_out = ssr_block(ssr_hidden, training=training)
            if self.use_block_mean_pooling:
                ssr_out = tf.reduce_mean(ssr_out, axis=1, keepdims=False) # [B, b_matrix, col] -> [B, col]  
            else:
                # Flatten for next block: [B, tokennum, col] -> [B, tokennum * col]
                ssr_out = tf.reshape(ssr_out, [tf.shape(ssr_hidden)[0], -1])
            if idx > 0 and self.use_ssrblock_residual:
                ssr_out = ssr_hidden + ssr_out
            if self.use_block_ln:
                ssr_out = self.block_lns[idx](ssr_out, training=training)

            ssr_hidden = ssr_out
        
        if not self.use_block_mean_pooling:
            last_tokennum = self.tokennum_list[-1]
            last_out_unit = self.out_unit_list[-1]
            ssr_hidden_reshaped = tf.reshape(ssr_hidden, [-1, last_tokennum, last_out_unit])

            if self.use_ssr_linear:
                print("==========Using SSR Senet ==========")
                squeeze = tf.reduce_mean(ssr_hidden_reshaped, axis=2)  # [B, tokennum]
                
                # Excitation: FC -> ReLU -> FC -> Sigmoid
                excitation = self.senet_fc1(squeeze)
                excitation = self.senet_fc2(excitation)
                excitation = tf.expand_dims(excitation, axis=-1)  # [B, tokennum, 1]
                
                # Scale and aggregate
                scaled_tokens = ssr_hidden_reshaped * excitation
                ssr_output = tf.reduce_sum(scaled_tokens, axis=1, keepdims=False)  # [B, out_unit]
            else:
                ssr_output = tf.reduce_mean(ssr_hidden_reshaped, axis=1, keepdims=False)  # [B, last_out_unit]
        else:
            ssr_output = ssr_hidden
        
        # Deep part (optional)
        deep_output = None
        if self.deep_layers:
            deep_input = tf.reshape(embeddings, [-1, self.field_size * self.embedding_size])
            deep_hidden = deep_input
            for layer in self.deep_network:
                deep_hidden = layer(deep_hidden, training=training)
            deep_output = self.deep_output(deep_hidden)
        
        # Combine all parts
        final_input = ssr_output
        if self.has_wide and wide_output is not None:
            final_input = tf.concat([final_input, wide_output], axis=1)
        if self.deep_layers and deep_output is not None:
            final_input = tf.concat([final_input, deep_output], axis=1)
        
        # Final prediction
        predictions = self.prediction_layer(final_input)
        
        return predictions
    
    def get_config(self):
        """Get model configuration."""
        config = super(SSRNetT, self).get_config()
        config.update({
            'feature_size': self.feature_size,
            'field_size': self.field_size,
            'embedding_size': self.embedding_size,
            'tokennum_list': self.tokennum_list,
            'hidden_unit_list': self.hidden_unit_list,
            'top_k_list': self.top_k_list,
            'out_unit_list': self.out_unit_list,
            'has_wide': self.has_wide,
            'deep_layers': self.deep_layers,
            'dropout_rates': self.dropout_rates,
            'l2_reg': self.l2_reg,
            'use_batch_norm': self.use_batch_norm,
            'block_version': self.block_version,
            'use_ssr_linear': self.use_ssr_linear,
            'use_block_mean_pooling': self.use_block_mean_pooling,
            'use_block_ln': self.use_block_ln
        })
        return config


class SSRNetTTrainer:

    def __init__(self,
                 model: SSRNetT,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 learning_rate_wide: float = 0.001,
                 patience: int = 3,
                 save_path: Optional[str] = None,
                 verbose: int = 1):
        """
        Initialize SSRNet-T trainer.
        
        Args:
            model: SSRNetT model instance
            optimizer: Optimizer type ('adam', 'sgd')
            learning_rate: Learning rate for main parameters
            learning_rate_wide: Learning rate for wide parameters (kept for API compatibility)
            patience: Early stopping patience
            save_path: Path to save model checkpoints
            verbose: Verbosity level
        """
        self.model = model
        self.patience = patience
        self.save_path = save_path
        self.verbose = verbose
        
        # Setup optimizer
        if optimizer.lower() == 'adam':
            self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy',
            metrics=['AUC']
        )
        
        # Early stopping variables
        self.best_val_auc = -1.0
        self.wait = 0
        self.best_weights = None
        
        # Create save directory if specified
        if self.save_path and not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Setup TensorBoard logging
        import datetime
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,      
            write_graph=True,       
            write_images=False,     
            update_freq=100,        
            profile_batch=0,        
            embeddings_freq=0       
        )
        
        monitoring_log_dir = os.path.join(log_dir, 'ssrnet_params')
        self.monitoring_callback = SSRNetMonitoringCallback(
            log_dir=monitoring_log_dir,
            log_freq='batch',  
            log_interval=100   
        )
        
        if self.verbose > 0:
            print(f"TensorBoard logs will be saved to: {log_dir}")
            print(f"TensorBoard will update metrics every 100 steps")
            print(f"SSRNet parameters (alphas, scales) will be logged every 100 steps")
    
    def _separate_features_labels(self, dataset):

        def extract_x_y(batch):
            if isinstance(batch, dict) and 'labels' in batch:
                # Separate features and labels
                features = {k: v for k, v in batch.items() if k != 'labels'}
                labels = batch['labels']
                return features, labels
            else:
                # If dataset is already in (x, y) format, return as-is
                return batch
        
        return dataset.map(extract_x_y)
    
    def fit(self, 
            train_dataset,
            validation_dataset=None,
            epochs: int = 1,
            steps_per_epoch: Optional[int] = None) -> bool:

        # Convert datasets to (x, y) format
        train_ds = self._separate_features_labels(train_dataset)
        val_ds = self._separate_features_labels(validation_dataset) if validation_dataset else None
        
        # Train using Keras fit
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[self.tensorboard_callback, self.monitoring_callback],
            verbose=self.verbose
        )
        
        # Get validation AUC
        if validation_dataset:
            val_auc = history.history['val_auc'][0] if epochs == 1 else history.history['val_auc'][-1]
            
            # Check for improvement
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.wait = 0
                self.best_weights = self.model.get_weights()
                
                if self.save_path:
                    self.model.save_weights(self.save_path)
                    
                if self.verbose > 0:
                    print(f"New best validation AUC: {val_auc:.4f}")
            else:
                self.wait += 1
                if self.verbose > 0:
                    print(f"No improvement. Patience: {self.wait}/{self.patience}")
            
            # Check for early stopping
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print(f"Early stopping triggered!")
                return False
        
        return True
    
    def evaluate(self, test_dataset):

        test_ds = self._separate_features_labels(test_dataset)
        results = self.model.evaluate(test_ds, verbose=self.verbose)
        
        # results is [loss, auc]
        test_loss = results[0]
        test_auc = results[1]
        
        return test_loss, test_auc
    
    def load_best_weights(self) -> bool:

        if self.save_path and os.path.exists(self.save_path + '.index'):
            try:
                self.model.load_weights(self.save_path)
                if self.verbose > 0:
                    print(f"Loaded best model weights from {self.save_path}")
                return True
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to load weights: {e}")
                return False
        elif self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            if self.verbose > 0:
                print("Loaded best model weights from memory")
            return True
        else:
            if self.verbose > 0:
                print("No saved weights found")
            return False

