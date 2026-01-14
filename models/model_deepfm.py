import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, log_loss
from typing import Optional, List, Tuple, Dict, Any
import os


class DeepFM(keras.Model):

    def __init__(self,
                 feature_size: int,
                 field_size: int,
                 embedding_size: int = 16,
                 deep_layers: Optional[List[int]] = None,
                 dropout_rates: Optional[List[float]] = None,
                 l2_reg: float = 0.0,
                 use_batch_norm: bool = False,
                 **kwargs):

        super(DeepFM, self).__init__(**kwargs)
        
        # Model configuration
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.deep_layers = deep_layers or [256, 128, 64]
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        
        # Set default dropout rates
        if dropout_rates is None:
            self.dropout_rates = [0.0, 0.1]  # [embedding, deep]
        else:
            self.dropout_rates = dropout_rates
        
        # Build model components
        self._build_model()
    
    def _build_model(self):
        # ==================== Embedding Layers ====================
        # Feature embeddings for FM and Deep parts
        self.feature_embedding = layers.Embedding(
            input_dim=self.feature_size,
            output_dim=self.embedding_size,
            embeddings_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name='feature_embeddings'
        )
        
        # Linear weights for FM first-order
        self.feature_bias = layers.Embedding(
            input_dim=self.feature_size,
            output_dim=1,
            embeddings_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name='feature_bias'
        )
        
        # ==================== Dropout Layers ====================
        self.embedding_dropout = layers.Dropout(self.dropout_rates[0])
        
        # ==================== Deep Part ====================
        self.deep_network = []
        input_size = self.field_size * self.embedding_size
        
        for i, hidden_size in enumerate(self.deep_layers):
            # Dense layer
            dense = layers.Dense(
                hidden_size,
                activation=None,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
                name=f'deep_dense_{i}'
            )
            self.deep_network.append(dense)
            
            # Batch normalization (optional)
            if self.use_batch_norm:
                bn = layers.BatchNormalization(name=f'deep_bn_{i}')
                self.deep_network.append(bn)
            
            # Activation
            activation = layers.Activation('relu', name=f'deep_relu_{i}')
            self.deep_network.append(activation)
            
            # Dropout
            if self.dropout_rates[1] > 0:
                dropout = layers.Dropout(self.dropout_rates[1], name=f'deep_dropout_{i}')
                self.deep_network.append(dropout)
        
        # Deep output layer
        self.deep_output = layers.Dense(
            1,
            activation=None,
            kernel_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name='deep_output'
        )
        
        # ==================== Final Output ====================
        # Global bias
        # self.global_bias = self.add_weight(
        #     name='global_bias',
        #     shape=[1],
        #     initializer='zeros',
        #     trainable=True
        # )
    
    def call(self, inputs, training=None):
        """
        Forward pass of the model.
        
        Args:
            inputs: Dictionary with 'feat_index' and 'feat_value'
            training: Training mode flag
            
        Returns:
            Predicted logits
        """
        feat_index = inputs['feat_index']  # [batch_size, field_size]
        feat_value = inputs['feat_value']  # [batch_size, field_size]
        
        # ==================== Embedding Lookup ====================
        # Get embeddings: [batch_size, field_size, embedding_size]
        embeddings = self.feature_embedding(feat_index)
        
        # Multiply by feature values: [batch_size, field_size, embedding_size]
        feat_value_expanded = tf.expand_dims(feat_value, axis=-1)
        embeddings = embeddings * feat_value_expanded
        
        # Apply dropout to embeddings
        embeddings = self.embedding_dropout(embeddings, training=training)
        
        # ==================== FM Part ====================
        # First-order (linear) part
        linear_terms = self.feature_bias(feat_index)  # [batch_size, field_size, 1]
        linear_terms = linear_terms * feat_value_expanded  # Weight by feature values
        y_first_order = tf.reduce_sum(linear_terms, axis=1)  # [batch_size, 1]
        
        # Second-order (interaction) part
        # Sum of squares: (sum of embeddings)^2
        sum_of_embeddings = tf.reduce_sum(embeddings, axis=1)  # [batch_size, embedding_size]
        sum_of_embeddings_square = tf.square(sum_of_embeddings)  # [batch_size, embedding_size]
        
        # Square of sums: sum of (embeddings^2)
        square_of_embeddings = tf.square(embeddings)  # [batch_size, field_size, embedding_size]
        square_of_embeddings_sum = tf.reduce_sum(square_of_embeddings, axis=1)  # [batch_size, embedding_size]
        
        # FM interaction: 0.5 * (sum_square - square_sum)
        y_second_order = 0.5 * tf.reduce_sum(
            sum_of_embeddings_square - square_of_embeddings_sum,
            axis=1,
            keepdims=True
        )  # [batch_size, 1]
        
        # ==================== Deep Part ====================
        # Flatten embeddings for deep network
        deep_input = tf.reshape(embeddings, [-1, self.field_size * self.embedding_size])
        
        # Pass through deep network
        deep_hidden = deep_input
        for layer in self.deep_network:
            deep_hidden = layer(deep_hidden, training=training)
        
        # Deep output
        y_deep = self.deep_output(deep_hidden)  # [batch_size, 1]
        
        # ==================== Combine All Parts ====================
        # Final prediction: global_bias + FM_first_order + FM_second_order + Deep
        logits = y_first_order + y_second_order + y_deep
        
        return logits
    
    def get_config(self):
        """Get model configuration."""
        config = super(DeepFM, self).get_config()
        config.update({
            'feature_size': self.feature_size,
            'field_size': self.field_size,
            'embedding_size': self.embedding_size,
            'deep_layers': self.deep_layers,
            'dropout_rates': self.dropout_rates,
            'l2_reg': self.l2_reg,
            'use_batch_norm': self.use_batch_norm,
        })
        return config


class DeepFMTrainer:
    """
    Trainer class for DeepFM model using Keras fit method.
    Compatible with AutoInt trainer pattern.
    """
    
    def __init__(self,
                 model: DeepFM,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 learning_rate_wide: float = 0.001,  # Not used, for compatibility
                 patience: int = 3,
                 save_path: Optional[str] = None,
                 verbose: int = 1):
        """
        Initialize DeepFM trainer.
        
        Args:
            model: DeepFM model instance
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            learning_rate: Learning rate
            learning_rate_wide: Learning rate for wide parameters (not used, for compatibility)
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
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.AUC(name='auc', from_logits=True)]
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
