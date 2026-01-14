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

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class StructuralSparseBlock(layers.Layer):

    def __init__(self, len_embeddings, b_matrices, d_mid_cols, out_units, use_block_mean_pooling, num_hidden_layers, has_residual, use_uniformly_indices, stage_name="structural_sparse_block_", **kwargs):
        super(StructuralSparseBlock, self).__init__(name=stage_name, **kwargs)
        self.len_embeddings = len_embeddings
        self.b_matrices = b_matrices
        self.d_mid_cols = d_mid_cols
        self.out_units = out_units
        self.layer_id = int(self.name.split("_")[-1])  
        self.num_hidden_layers = num_hidden_layers
        self.has_residual = has_residual
        self.use_uniformly_indices = use_uniformly_indices
        self.use_block_mean_pooling = use_block_mean_pooling

        
        if self.use_uniformly_indices:
            indices = generate_structured_indices_uniformly_(
                self.len_embeddings, self.b_matrices, self.d_mid_cols, 
                int(self.layer_id * 1000)
            )
        else:
            indices = generate_structured_indices_(
                self.len_embeddings, self.b_matrices, self.d_mid_cols, 
                int(self.layer_id * 1000)
            )
        
        self.fixed_indices_group = tf.constant(
            indices,
            dtype=tf.int32,
            name=f'fixed_indices_g{self.layer_id}'
        )

    def build(self, input_shape):

        if self.use_block_mean_pooling:
            self.hidden_layers = [
                [tf.keras.layers.Dense(self.out_units, activation="gelu") for _ in range(self.num_hidden_layers)]
                for _ in range(self.b_matrices)]
        else:
            self.hidden_layers = [
                [tf.keras.layers.Dense(self.d_mid_cols, activation="gelu") for _ in range(self.num_hidden_layers)]
                for _ in range(self.b_matrices)]

        self.layer_norms = [tf.keras.layers.LayerNormalization(axis=-1) for _ in range(self.b_matrices)]

    def call(self, inputs, training=None):

        indices_group = self.fixed_indices_group  # [b_matrices, d_mid_cols]
        out_tensor = []
        inputs_flat = tf.reshape(inputs, [-1, self.len_embeddings])
        
        for token_id in range(indices_group.shape[0]):
            with tf.name_scope(f"{self.name}_tokenizer_{token_id}"):
                indices = indices_group[token_id]  # [d_mid_cols]
                interaction_output = tf.gather(inputs_flat, indices, axis=1)
            
            with tf.name_scope(f"{self.name}_hidden_{token_id}"):
                hidden_state = interaction_output
                for hidden_layer in self.hidden_layers[token_id]:
                    hidden_state = hidden_layer(hidden_state, training=training)
                if self.has_residual:
                    hidden_state = self.layer_norms[token_id](hidden_state + interaction_output)
                else:
                    hidden_state = self.layer_norms[token_id](hidden_state)
            
            out_tensor.append(hidden_state)

        outputs = tf.stack(out_tensor, axis=1)  # [B, b_matrices, d_mid_cols]
        return outputs


class SSRNet(keras.Model):

    def __init__(self,
                 feature_size: int,
                 field_size: int,
                 embedding_size: int = 16,
                 b_matrices: List[int] = [16, 16],
                 d_mid_cols: List[int] = [256, 256],
                 num_hidden_layers: int = 2,
                 has_residual: bool = True,
                 use_uniformly_indices: bool = True,
                 has_wide: bool = False,
                 deep_layers: Optional[List[int]] = None,
                 dropout_rates: Optional[List[float]] = None,
                 l2_reg: float = 0.0,
                 use_batch_norm: bool = False,
                 input_emb_norm: str = "bn",
                 use_block_mean_pooling: bool = False,
                 out_units: List[int] = [128, 128],
                 **kwargs):        
        """
        Initialize SSRNet  model (optimized for n_ones=1).
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields
            embedding_size: Embedding dimension
            b_matrices: List of matrix counts for each SSR block
            d_mid_cols: List of middle layer dimensions for each SSR block
            num_hidden_layers: Number of hidden layers in each block
            has_residual: Whether to use residual connections
            use_uniformly_indices: Whether to use uniformly distributed indices
            has_wide: Whether to include wide (linear) part
            deep_layers: Hidden dimensions for deep network
            dropout_rates: Dropout rates [unused, embedding, deep]
            l2_reg: L2 regularization coefficient
            use_batch_norm: Whether to use batch normalization
        """
        super(SSRNet, self).__init__(**kwargs)
        
        # Model configuration
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.b_matrices = b_matrices
        self.d_mid_cols = d_mid_cols
        self.has_residual = has_residual
        self.use_uniformly_indices = use_uniformly_indices
        self.has_wide = has_wide
        self.deep_layers = deep_layers or []
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        self.num_hidden_layers = num_hidden_layers
        self.input_emb_norm = input_emb_norm
        self.use_block_mean_pooling = use_block_mean_pooling
        self.out_units = out_units
        
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
            embeddings_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name='feature_embeddings'
        )
        if self.input_emb_norm == "bn":
            print("==========Using Batch Normalization for embedding==========")
            self.embedding_norm = layers.BatchNormalization(axis=-2,name='embedding_bn')
        elif self.input_emb_norm == "ln":
            print("==========Using Layer Normalization for embedding==========")
            self.embedding_norm = layers.LayerNormalization(axis=-1,name='embedding_ln')
        else:
            self.embedding_norm = lambda x: x

        if self.has_wide:
            self.feature_bias = layers.Embedding(
                input_dim=self.feature_size,
                output_dim=1,
                name='feature_bias'
            )
        
        # Dropout layers
        self.embedding_dropout = layers.Dropout(self.dropout_rates[1])
        
        # SSR Blocks - using optimized  version
        self.ssr_blocks = []
        for i in range(len(self.b_matrices)):
            if i == 0:
                input_dim = self.field_size * self.embedding_size
            else:
                if self.use_block_mean_pooling:
                    input_dim = self.out_units[i - 1]
                else:
                    input_dim = self.b_matrices[i - 1] * self.out_units[i - 1]
                

            ssr_block = StructuralSparseBlock(
                input_dim,
                self.b_matrices[i],
                self.d_mid_cols[i],
                self.out_units[i],
                self.use_block_mean_pooling,
                self.num_hidden_layers,
                has_residual=self.has_residual,
                use_uniformly_indices=self.use_uniformly_indices,
                stage_name=f"ssr_block__{i+1}"
            )
            self.ssr_blocks.append(ssr_block)

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
            print("==========No deep network specified, using SSRNet  only==========")
        # Final prediction layer
        self.prediction_layer = keras.Sequential([
            layers.Dense(64, activation='relu', name='final_hidden'),
            layers.Dense(1, activation='sigmoid', name='prediction')
        ], name='prediction_layer')


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
        
        # SSR part
        ssr_hidden = self.embedding_norm(embeddings)
        for ssr_block in self.ssr_blocks:
            ssr_hidden = ssr_block(ssr_hidden, training=training)
            if self.use_block_mean_pooling:
                ssr_hidden = tf.reduce_mean(ssr_hidden, axis=1, keepdims=False) # [B, b_matrix, col] -> [B, col]  

        ## Squeeze ssr_output for final_input [batch_size, d_mid_cols]
        if not self.use_block_mean_pooling:
            ssr_output = tf.reduce_mean(ssr_hidden, axis=1, keepdims=False)
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
        config = super(SSRNet, self).get_config()
        config.update({
            'feature_size': self.feature_size,
            'field_size': self.field_size,
            'embedding_size': self.embedding_size,
            'b_matrices': self.b_matrices,
            'd_mid_cols': self.d_mid_cols,
            'has_residual': self.has_residual,
            'use_uniformly_indices': self.use_uniformly_indices,
            'has_wide': self.has_wide,
            'deep_layers': self.deep_layers,
            'dropout_rates': self.dropout_rates,
            'l2_reg': self.l2_reg,
            'use_batch_norm': self.use_batch_norm,
            'input_emb_norm': self.input_emb_norm
        })
        return config


class SSRNetTrainer:
    """
    Trainer class for SSRNet  model.
    """
    
    def __init__(self,
                 model: SSRNet,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 learning_rate_wide: float = 0.001,
                 patience: int = 5,
                 save_path: Optional[str] = None,
                 verbose: int = 1):

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
                    print(f"No improvement. Wait: {self.wait}/{self.patience}")
        
        return self.wait < self.patience
    
    def evaluate(self, dataset) -> Tuple[float, float]:

        # Convert dataset to (x, y) format
        test_ds = self._separate_features_labels(dataset)
        
        # Evaluate using Keras evaluate
        results = self.model.evaluate(test_ds, verbose=self.verbose)
        test_loss, test_auc = results[0], results[1]
        
        return test_loss, test_auc
    
    def load_best_weights(self) -> bool:

        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            return True
        elif self.save_path and os.path.exists(self.save_path + '.index'):
            self.model.load_weights(self.save_path)
            return True
        else:
            return False


# ==================== Optimized Index Generation Functions ====================

def generate_structured_indices_(rows, b_matrices, d_mid_cols, seed_offset):
    target = []
    for layer_id in range(b_matrices):
        indices = np.arange(rows)
        np.random.seed(seed_offset + layer_id)  
        shuffled_indices = np.random.permutation(indices)
        target.append(shuffled_indices[:d_mid_cols])
    return target



def generate_structured_indices_uniformly_(rows, b_matrices, d_mid_cols, seed_offset):

    all_indices = np.zeros((b_matrices, d_mid_cols), dtype=np.int32)
    
    for i in range(b_matrices):
        rng = np.random.RandomState(seed_offset + i)
        
        if d_mid_cols <= rows:
            permutation = rng.permutation(rows)
            all_indices[i] = permutation[:d_mid_cols]
        else:
            n_repeats = (d_mid_cols + rows - 1) // rows
            permuted_blocks = [rng.permutation(rows) for _ in range(n_repeats)]
            combined = np.concatenate(permuted_blocks)
            all_indices[i] = combined[:d_mid_cols]
    
    return all_indices
