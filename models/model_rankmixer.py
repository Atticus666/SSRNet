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


class TokenMixingLayer(layers.Layer):
    """
    Token Mixing Layer for RankMixer.
    Mixes information across tokens (axis=1).
    """
    
    def __init__(self, num_tokens: int, hidden_dim: int, stage_name: str = "token_mixing", **kwargs):
        super(TokenMixingLayer, self).__init__(name=stage_name, **kwargs)
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
    
    def build(self, input_shape):
        # LayerNormalization for residual connection
        self.layer_norm = layers.LayerNormalization(axis=-1, name=f"{self.name}_ln")
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: [B, num_tokens, hidden_dim]
        Returns:
            outputs: [B, num_tokens, hidden_dim]
        
        Implementation details:
        - Split along hidden_dim (axis=2) into num_tokens parts
        - Reshape each part to [B, hidden_dim]
        - Stack back along axis=1 to get [B, num_tokens, hidden_dim]
        - This creates a feature mixing effect different from simple transpose
        """
        # Split along hidden dimension (axis=2)
        # inputs: [B, num_tokens, hidden_dim] -> list of [B, num_tokens, hidden_dim//num_tokens]
        split_tensors = tf.split(inputs, num_or_size_splits=self.num_tokens, axis=2)
        
        # Reshape each split: [B, num_tokens, hidden_dim//num_tokens] -> [B, hidden_dim]
        reshaped_tensors = []
        for tensor in split_tensors:
            # Flatten and reshape to [B, hidden_dim]
            reshaped = tf.reshape(tensor, [-1, self.hidden_dim])
            reshaped_tensors.append(reshaped)
        
        # Stack along axis=1: list of [B, hidden_dim] -> [B, num_tokens, hidden_dim]
        new_tensor = tf.stack(reshaped_tensors, axis=1)
        
        # Residual connection + LayerNorm
        output = self.layer_norm(new_tensor + inputs)
        
        return output


class MLPMixerLayer(layers.Layer):
    """
    MLP Mixer Layer for RankMixer.
    Applies MLP to each token independently.
    """
    
    def __init__(self, num_tokens: int, hidden_dim: int, stage_name: str = "mlp_mixer", **kwargs):
        super(MLPMixerLayer, self).__init__(name=stage_name, **kwargs)
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
    
    def build(self, input_shape):
        
        self.mlp_layers = []
        for i in range(self.num_tokens):
            token_mlp = [
                layers.Dense(self.hidden_dim, activation='gelu', name=f"{self.name}_token{i}_fc1"),
                layers.Dense(self.hidden_dim, activation=None, name=f"{self.name}_token{i}_fc2")
            ]
            self.mlp_layers.append(token_mlp)
        
        # LayerNormalization
        self.layer_norm = layers.LayerNormalization(axis=-1, name=f"{self.name}_ln")
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: [B, num_tokens, hidden_dim]
        Returns:
            outputs: [B, num_tokens, hidden_dim]
        """
        # Split along token dimension
        token_list = tf.unstack(inputs, axis=1)  # List of [B, hidden_dim]
        
        # Apply MLP to each token
        output_tokens = []
        for i, token in enumerate(token_list):
            # Two-layer MLP
            hidden = self.mlp_layers[i][0](token)  # [B, hidden_dim]
            output = self.mlp_layers[i][1](hidden)  # [B, hidden_dim]
            output_tokens.append(output)
        
        # Stack back
        mlp_output = tf.stack(output_tokens, axis=1)  # [B, num_tokens, hidden_dim]
        
        # Residual connection + LayerNorm
        output = self.layer_norm(mlp_output + inputs)
        
        return output


class RankMixerBlock(layers.Layer):
    """
    RankMixer Block: Token Mixing + MLP Mixing
    """
    
    def __init__(self, num_tokens: int, hidden_dim: int, block_id: int, **kwargs):
        super(RankMixerBlock, self).__init__(name=f"rankmixer_block_{block_id}", **kwargs)
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.block_id = block_id
    
    def build(self, input_shape):
        self.token_mixing = TokenMixingLayer(
            self.num_tokens, 
            self.hidden_dim, 
            stage_name=f"token_mixing_{self.block_id}"
        )
        self.mlp_mixing = MLPMixerLayer(
            self.num_tokens, 
            self.hidden_dim, 
            stage_name=f"mlp_mixing_{self.block_id}"
        )
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        # Token mixing
        x = self.token_mixing(inputs, training=training)
        # MLP mixing
        x = self.mlp_mixing(x, training=training)
        return x


class RankMixer(keras.Model):
    """
    RankMixer model using TensorFlow 2.11 with Keras fit.
    """
    
    def __init__(self,
                 feature_size: int,
                 field_size: int,
                 embedding_size: int = 16,
                 num_tokens: int = 16,
                 hidden_dim: int = 256,
                 num_blocks: int = 2,
                 has_wide: bool = False,
                 deep_layers: Optional[List[int]] = None,
                 dropout_rates: Optional[List[float]] = None,
                 l2_reg: float = 0.0,
                 use_batch_norm: bool = False,
                 **kwargs):
        """
        Initialize RankMixer model.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields
            embedding_size: Embedding dimension
            num_tokens: Number of tokens to split into
            hidden_dim: Hidden dimension for each token
            num_blocks: Number of RankMixer blocks
            has_wide: Whether to include wide (linear) part
            deep_layers: Hidden dimensions for deep network
            dropout_rates: Dropout rates [unused, embedding, deep]
            l2_reg: L2 regularization coefficient
            use_batch_norm: Whether to use batch normalization
        """
        super(RankMixer, self).__init__(**kwargs)
        
        # Model configuration
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.has_wide = has_wide
        self.deep_layers = deep_layers or []
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        
        # Set default dropout rates
        if dropout_rates is None:
            self.dropout_rates = [0.0, 0.0, 0.1]  # [unused, embedding, deep]
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
        
        if self.has_wide:
            self.feature_bias = layers.Embedding(
                input_dim=self.feature_size,
                output_dim=1,
                name='feature_bias'
            )
        
        # Dropout layers
        self.embedding_dropout = layers.Dropout(self.dropout_rates[1])
        
        # Token mapping layers (map each token to hidden_dim)
        self.token_mappers = []
        for i in range(self.num_tokens):
            mapper = layers.Dense(
                self.hidden_dim,
                activation=None,
                name=f'token_mapper_{i}'
            )
            self.token_mappers.append(mapper)
        
        # RankMixer blocks
        self.rankmixer_blocks = []
        for i in range(self.num_blocks):
            block = RankMixerBlock(
                self.num_tokens,
                self.hidden_dim,
                block_id=i+1
            )
            self.rankmixer_blocks.append(block)
        
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
        
        # Final prediction layer
        # self.prediction_layer = layers.Dense(1, activation='sigmoid', name='prediction')
        self.prediction_layer = keras.Sequential([
            layers.Dense(64, activation='relu', name='final_hidden'),
            layers.Dense(1, activation='sigmoid', name='prediction')
        ], name='prediction_layer')
    
    def call(self, inputs, training=None):
        """
        Forward pass of the model.
        
        Args:
            inputs: Dictionary with 'feat_index' and 'feat_value'
            training: Training mode flag
            
        Returns:
            Predicted probabilities
        """
        feat_index = inputs['feat_index']  # [batch_size, field_size]
        feat_value = inputs['feat_value']  # [batch_size, field_size]
        
        # Embedding lookup
        embeddings = self.feature_embedding(feat_index)  # [batch_size, field_size, embedding_size]
        feat_value = tf.expand_dims(feat_value, axis=-1)  # [batch_size, field_size, 1]
        embeddings = embeddings * feat_value  # Element-wise multiplication
        embeddings = self.embedding_dropout(embeddings, training=training)
        
        # Flatten embeddings
        embeddings_flat = tf.reshape(embeddings, [-1, self.field_size * self.embedding_size])
        
        # Wide part (linear)
        wide_output = None
        if self.has_wide:
            wide_logits = self.feature_bias(feat_index)  # [batch_size, field_size, 1]
            wide_output = tf.reduce_sum(wide_logits * feat_value, axis=1)  # [batch_size, 1]
        
        # Split into tokens and map to hidden_dim
        
        token_size = (self.field_size * self.embedding_size) // self.num_tokens
        split_embeddings = tf.split(embeddings_flat, num_or_size_splits=self.num_tokens, axis=1)
        
        # Map each token to hidden_dim
        mapped_tokens = []
        for i, token in enumerate(split_embeddings):
            mapped = self.token_mappers[i](token)  # [B, hidden_dim]
            mapped_tokens.append(mapped)
        
        # Stack to [B, num_tokens, hidden_dim]
        rankmixer_input = tf.stack(mapped_tokens, axis=1)
        
        # RankMixer blocks
        rankmixer_output = rankmixer_input
        for block in self.rankmixer_blocks:
            rankmixer_output = block(rankmixer_output, training=training)
        
        # Global pooling
        rankmixer_pooled = tf.reduce_mean(rankmixer_output, axis=1)  # [B, hidden_dim]
        
        # Deep part (optional)
        deep_output = None
        if self.deep_layers:
            deep_hidden = embeddings_flat
            for layer in self.deep_network:
                deep_hidden = layer(deep_hidden, training=training)
            deep_output = self.deep_output(deep_hidden)
        
        # Combine all parts
        final_input = rankmixer_pooled
        if self.has_wide and wide_output is not None:
            final_input = tf.concat([final_input, wide_output], axis=1)
        if self.deep_layers and deep_output is not None:
            final_input = tf.concat([final_input, deep_output], axis=1)
        
        # Final prediction
        predictions = self.prediction_layer(final_input)
        
        return predictions
    
    def get_config(self):
        """Get model configuration."""
        config = super(RankMixer, self).get_config()
        config.update({
            'feature_size': self.feature_size,
            'field_size': self.field_size,
            'embedding_size': self.embedding_size,
            'num_tokens': self.num_tokens,
            'hidden_dim': self.hidden_dim,
            'num_blocks': self.num_blocks,
            'has_wide': self.has_wide,
            'deep_layers': self.deep_layers,
            'dropout_rates': self.dropout_rates,
            'l2_reg': self.l2_reg,
            'use_batch_norm': self.use_batch_norm,
        })
        return config


class RankMixerTrainer:
    """
    Trainer class for RankMixer model using Keras fit method.
    Compatible with SSRNet, DCN v2, and Wukong trainers.
    """
    
    def __init__(self,
                 model: RankMixer,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 learning_rate_wide: float = 0.001,
                 patience: int = 5,
                 save_path: Optional[str] = None,
                 verbose: int = 1):
        """
        Initialize RankMixer trainer.
        
        Args:
            model: RankMixer model instance
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            learning_rate: Learning rate for main parameters
            learning_rate_wide: Learning rate for wide parameters (not used in keras.fit, kept for API compatibility)
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
    
    def _separate_features_labels(self, dataset):
        """
        Convert dataset with labels to (x, y) format for Keras fit.
        
        Args:
            dataset: Dataset with dict containing 'feat_index', 'feat_value', 'labels'
            
        Returns:
            Dataset in (features, labels) format
        """
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
        """
        Train the model for one epoch using Keras fit method.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Number of epochs (typically 1 for compatibility)
            steps_per_epoch: Number of steps per epoch
            
        Returns:
            Whether to continue training (not early stopped)
        """
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
        """
        Evaluate model on dataset.
        
        Args:
            dataset: Test dataset
            
        Returns:
            Tuple of (test_loss, test_auc)
        """
        # Convert dataset to (x, y) format
        test_ds = self._separate_features_labels(dataset)
        
        # Evaluate using Keras evaluate
        results = self.model.evaluate(test_ds, verbose=self.verbose)
        test_loss, test_auc = results[0], results[1]
        
        return test_loss, test_auc
    
    def load_best_weights(self) -> bool:
        """
        Load the best model weights.
        
        Returns:
            Whether weights were successfully loaded
        """
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            return True
        elif self.save_path and os.path.exists(self.save_path + '.index'):
            self.model.load_weights(self.save_path)
            return True
        else:
            return False
