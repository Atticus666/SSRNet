import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import roc_auc_score, log_loss
from typing import Optional, List, Tuple, Dict, Any
import os

class DNNLayer(layers.Layer):
    """
    Deep Neural Network layer with multiple hidden layers.
    """
    
    def __init__(self,
                 hidden_units: List[int],
                 activation: str = 'relu',
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = False,
                 l2_reg: float = 0.0,
                 **kwargs):
        """
        Initialize DNN layer.
        
        Args:
            hidden_units: List of hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            l2_reg: L2 regularization coefficient
        """
        super(DNNLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.l2_reg = l2_reg
        
        # Build DNN layers
        self.dnn_layers = []
        regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
        
        for i, units in enumerate(hidden_units):
            # Dense layer
            dense = layers.Dense(
                units,
                activation=None,
                kernel_regularizer=regularizer,
                kernel_initializer=keras.initializers.GlorotNormal(),
                name=f'dnn_layer_{i}'
            )
            self.dnn_layers.append(dense)
            
            # Batch normalization
            if use_batch_norm:
                bn = layers.BatchNormalization(name=f'bn_{i}')
                self.dnn_layers.append(bn)
            
            # Activation
            act = layers.Activation(activation, name=f'activation_{i}')
            self.dnn_layers.append(act)
            
            # Dropout
            if dropout_rate > 0:
                dropout = layers.Dropout(dropout_rate, name=f'dropout_{i}')
                self.dnn_layers.append(dropout)
    
    def call(self, inputs, training=None):
        """
        Apply DNN transformation.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            
        Returns:
            DNN output
        """
        x = inputs
        for layer in self.dnn_layers:
            if isinstance(layer, (layers.Dropout, layers.BatchNormalization)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


class FFN(keras.Model):
    """
    Feed Forward Network (FFN) model for CTR prediction using TensorFlow 2.11.
    
    This is a simple baseline model that only uses fully connected layers (DNN)
    for feature transformation and prediction.
    """
    
    def __init__(self,
                 feature_size: int,
                 field_size: int,
                 embedding_size: int = 16,
                 deep_layers: Optional[List[int]] = None,
                 l2_reg_embedding: float = 0.0,
                 l2_reg_deep: float = 0.0,
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = False,
                 **kwargs):
        """
        Initialize FFN model.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields  
            embedding_size: Embedding dimension
            deep_layers: Hidden dimensions for deep network
            l2_reg_embedding: L2 regularization for embeddings
            l2_reg_deep: L2 regularization for deep network
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(FFN, self).__init__(**kwargs)
        
        # Model configuration
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.deep_layers = deep_layers or [256, 256, 256]
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_deep = l2_reg_deep
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Calculate dimensions
        self.input_dim = field_size * embedding_size
        
        # Build model components
        self._build_model()
    
    def _build_model(self):
        """Build model components."""
        # Embedding layer
        self.feature_embedding = layers.Embedding(
            input_dim=self.feature_size,
            output_dim=self.embedding_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
            embeddings_regularizer=keras.regularizers.l2(self.l2_reg_embedding) if self.l2_reg_embedding > 0 else None,
            name='feature_embeddings'
        )
        
        # Embedding dropout
        self.embedding_dropout = layers.Dropout(self.dropout_rate)
        
        # Deep network
        self.deep_net = DNNLayer(
            hidden_units=self.deep_layers,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            l2_reg=self.l2_reg_deep
        )
        
        # Output layer
        self.output_layer = layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer=keras.initializers.GlorotNormal(),
            name='output'
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass through FFN.
        
        Args:
            inputs: Input dictionary or tensor
                - If dict: {'feat_index': tensor, 'feat_value': tensor, 'labels': tensor}
                - If tensor: [batch_size, field_size]
            training: Training mode flag
            
        Returns:
            Prediction probabilities [batch_size, 1]
        """
        feat_index = inputs['feat_index']  # [batch_size, field_size]
        feat_value = inputs['feat_value']  # [batch_size, field_size]
     
        # Embedding lookup
        embeddings = self.feature_embedding(feat_index)  # [batch_size, field_size, embedding_size]
        feat_value = tf.expand_dims(feat_value, axis=-1)  # [batch_size, field_size, 1]
        embeddings = embeddings * feat_value  # Element-wise multiplication
        embeddings = self.embedding_dropout(embeddings, training=training)
        
        # Flatten embeddings
        flat_embeddings = tf.reshape(embeddings, [-1, self.input_dim])  # [batch_size, field_size * embedding_size]
        
        # Deep network
        deep_output = self.deep_net(flat_embeddings, training=training)
        
        # Output
        output = self.output_layer(deep_output)
        
        return output
    
    def train_step(self, data):
        """
        Custom training step to handle dictionary input format.
        
        Args:
            data: Input data dictionary containing 'feat_index', 'feat_value', and 'labels'
            
        Returns:
            Dictionary of metrics
        """
        # Extract labels from input dictionary
        if isinstance(data, dict):
            y = data.pop('labels')
            x = data
        else:
            x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return metrics
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Custom test step to handle dictionary input format.
        
        Args:
            data: Input data dictionary containing 'feat_index', 'feat_value', and 'labels'
            
        Returns:
            Dictionary of metrics
        """
        # Extract labels from input dictionary
        if isinstance(data, dict):
            y = data.pop('labels')
            x = data
        else:
            x, y = data
        
        # Compute predictions
        y_pred = self(x, training=False)
        
        # Update compiled loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return metrics
        return {m.name: m.result() for m in self.metrics}
    
    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'feature_size': self.feature_size,
            'field_size': self.field_size,
            'embedding_size': self.embedding_size,
            'deep_layers': self.deep_layers,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
        })
        return config


class FFNTrainer:
    """
    Trainer class for FFN model.
    """
    
    def __init__(self,
                 model: FFN,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 patience: int = 5,
                 save_path: Optional[str] = None,
                 verbose: int = 1):
        """
        Initialize trainer.
        
        Args:
            model: FFN model instance
            optimizer: Optimizer name
            learning_rate: Learning rate
            patience: Early stopping patience
            save_path: Path to save best model
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
    
    def fit(self,
            train_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset,
            epochs: int = 1,
            steps_per_epoch: Optional[int] = None) -> bool:
        """
        Train the model for one or more epochs.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Number of epochs
            steps_per_epoch: Steps per epoch
            
        Returns:
            Whether to continue training (not early stopped)
        """
        # Train the model
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_dataset,
            verbose=self.verbose
        )
        
        # Get validation AUC
        val_auc = history.history['val_auc'][-1]
        
        # Early stopping logic
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.wait = 0
            self.best_weights = self.model.get_weights()
            
            # Save model if path provided
            if self.save_path:
                os.makedirs(self.save_path, exist_ok=True)
                self.model.save_weights(os.path.join(self.save_path, 'best_model.weights.h5'))
                if self.verbose > 0:
                    print(f"Saved best model with validation AUC: {val_auc:.4f}")
        else:
            self.wait += 1
            
        return self.wait < self.patience
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Tuple[float, float]:
        """
        Evaluate the model on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Tuple of (test_loss, test_auc)
        """
        results = self.model.evaluate(test_dataset, verbose=self.verbose)
        test_loss = results[0]
        test_auc = results[1]
        
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
        elif self.save_path and os.path.exists(os.path.join(self.save_path, 'best_model.weights.h5')):
            self.model.load_weights(os.path.join(self.save_path, 'best_model.weights.h5'))
            return True
        return False
