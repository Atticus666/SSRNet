import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, log_loss
from typing import Optional, List, Tuple, Dict, Any
import os
import sys
from time import time
from tqdm import tqdm

class LayerNormalization(layers.Layer):
    
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        """
        Initialize Layer Normalization.
        
        Args:
            epsilon: Small value to prevent division by zero
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
    
    def build(self, input_shape):
        """Build layer parameters."""
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )
        super(LayerNormalization, self).build(input_shape)
    
    def call(self, inputs):
        """Apply layer normalization."""
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

class MultiHeadAttention(layers.Layer):

    def __init__(self, 
                 num_units: Optional[int] = None,
                 num_heads: int = 1,
                 dropout_rate: float = 0.0,
                 has_residual: bool = True,
                 activation: str = 'relu',
                 **kwargs):

        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.has_residual = has_residual
        self.activation = keras.activations.get(activation)
        
    def build(self, input_shape):
        """Build layer parameters."""
        if self.num_units is None:
            self.num_units = input_shape[-1]
            
        # Linear projection layers
        self.query_dense = layers.Dense(self.num_units, activation=self.activation)
        self.key_dense = layers.Dense(self.num_units, activation=self.activation)
        self.value_dense = layers.Dense(self.num_units, activation=self.activation)
        
        if self.has_residual:
            self.residual_dense = layers.Dense(self.num_units, activation=self.activation)
            
        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm = LayerNormalization()
        
        super(MultiHeadAttention, self).build(input_shape)
    
    def call(self, inputs, training=None):

        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Linear projections
        queries = self.query_dense(inputs)  # [B, L, num_units]
        keys = self.key_dense(inputs)       # [B, L, num_units]
        values = self.value_dense(inputs)   # [B, L, num_units]
        
        if self.has_residual:
            residual = self.residual_dense(inputs)
        
        # Split into multiple heads
        queries = tf.reshape(queries, [batch_size * self.num_heads, seq_length, self.num_units // self.num_heads])
        keys = tf.reshape(keys, [batch_size * self.num_heads, seq_length, self.num_units // self.num_heads])
        values = tf.reshape(values, [batch_size * self.num_heads, seq_length, self.num_units // self.num_heads])
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(queries, keys, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.num_units // self.num_heads, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention to values
        attention_output = tf.matmul(attention_weights, values)
        
        # Concatenate heads
        attention_output = tf.reshape(attention_output, [batch_size, seq_length, self.num_units])
        
        # Residual connection
        if self.has_residual:
            attention_output += residual
            
        # Activation and normalization
        attention_output = self.activation(attention_output)
        attention_output = self.layer_norm(attention_output)
        
        return attention_output

class AutoInt(keras.Model):
    
    def __init__(self,
                 feature_size: int,
                 field_size: int,
                 embedding_size: int = 16,
                 attention_blocks: int = 3,
                 attention_heads: int = 2,
                 block_shape: Optional[List[int]] = None,
                 has_residual: bool = True,
                 has_wide: bool = False,
                 deep_layers: Optional[List[int]] = None,
                 dropout_rates: Optional[List[float]] = None,
                 l2_reg: float = 0.0,
                 use_batch_norm: bool = False,
                 **kwargs):
        """
        Initialize AutoInt model.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields
            embedding_size: Embedding dimension
            attention_blocks: Number of attention blocks
            attention_heads: Number of attention heads
            block_shape: Output dimensions for each attention block
            has_residual: Whether to use residual connections
            has_wide: Whether to include wide (linear) part
            deep_layers: Hidden dimensions for deep network
            dropout_rates: Dropout rates [attention, embedding, deep]
            l2_reg: L2 regularization coefficient
            use_batch_norm: Whether to use batch normalization
        """
        super(AutoInt, self).__init__(**kwargs)
        
        # Model configuration
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.attention_blocks = attention_blocks
        self.attention_heads = attention_heads
        self.has_residual = has_residual
        self.has_wide = has_wide
        self.deep_layers = deep_layers or []
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        
        # Set default block shapes
        if block_shape is None:
            self.block_shape = [embedding_size] * attention_blocks
        else:
            self.block_shape = block_shape
            
        # Set default dropout rates
        if dropout_rates is None:
            self.dropout_rates = [0.0, 0.0, 0.1]  # [attention, embedding, deep]
        else:
            self.dropout_rates = dropout_rates
            
        self.output_size = self.block_shape[-1]
        
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
        
        # Multi-head attention blocks
        self.attention_layers = []
        for i in range(self.attention_blocks):
            attention_layer = MultiHeadAttention(
                num_units=self.block_shape[i],
                num_heads=self.attention_heads,
                dropout_rate=self.dropout_rates[0],
                has_residual=self.has_residual,
                name=f'attention_block_{i}'
            )
            self.attention_layers.append(attention_layer)
        
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
        
        # AutoInt part (multi-head attention)
        attention_output = embeddings
        for attention_layer in self.attention_layers:
            attention_output = attention_layer(attention_output, training=training)
        
        # Flatten attention output
        attention_flat = tf.reshape(attention_output, [-1, self.output_size * self.field_size])
        
        # Deep part (optional)
        deep_output = None
        if self.deep_layers:
            deep_input = tf.reshape(embeddings, [-1, self.field_size * self.embedding_size])
            deep_hidden = deep_input
            for layer in self.deep_network:
                deep_hidden = layer(deep_hidden, training=training)
            deep_output = self.deep_output(deep_hidden)
        
        # Combine all parts
        final_input = attention_flat
        if self.has_wide and wide_output is not None:
            final_input = tf.concat([final_input, wide_output], axis=1)
        if self.deep_layers and deep_output is not None:
            final_input = tf.concat([final_input, deep_output], axis=1)
        
        # Final prediction
        predictions = self.prediction_layer(final_input)
        
        return predictions
    
    def get_config(self):
        """Get model configuration."""
        config = super(AutoInt, self).get_config()
        config.update({
            'feature_size': self.feature_size,
            'field_size': self.field_size,
            'embedding_size': self.embedding_size,
            'attention_blocks': self.attention_blocks,
            'attention_heads': self.attention_heads,
            'block_shape': self.block_shape,
            'has_residual': self.has_residual,
            'has_wide': self.has_wide,
            'deep_layers': self.deep_layers,
            'dropout_rates': self.dropout_rates,
            'l2_reg': self.l2_reg,
            'use_batch_norm': self.use_batch_norm,
        })
        return config

class AutoIntTrainer:
    
    def __init__(self,
                 model: AutoInt,
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
