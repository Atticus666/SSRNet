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


class AFN(keras.Model):
    """
    AFN learns adaptive-order feature interactions through logarithmic transformation,
    which can capture both low-order and high-order feature interactions adaptively.
    """
    
    def __init__(self,
                 feature_size: int,
                 field_size: int,
                 embedding_size: int = 16,
                 logarithmic_neurons: int = 5,
                 afn_hidden_units: Optional[List[int]] = None,
                 afn_activation: str = 'relu',
                 afn_dropout: float = 0.0,
                 ensemble_dnn: bool = True,
                 dnn_hidden_units: Optional[List[int]] = None,
                 dnn_activation: str = 'relu',
                 dnn_dropout: float = 0.0,
                 l2_reg: float = 0.0,
                 use_batch_norm: bool = True,
                 **kwargs):
        """
        Initialize AFN model.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields
            embedding_size: Embedding dimension
            logarithmic_neurons: Number of logarithmic neurons (adaptive order)
            afn_hidden_units: Hidden layer sizes for AFN dense network
            afn_activation: Activation function for AFN network
            afn_dropout: Dropout rate for AFN network
            ensemble_dnn: Whether to use ensemble DNN
            dnn_hidden_units: Hidden layer sizes for DNN network
            dnn_activation: Activation function for DNN network
            dnn_dropout: Dropout rate for DNN network
            l2_reg: L2 regularization coefficient
            use_batch_norm: Whether to use batch normalization
        """
        super(AFN, self).__init__(**kwargs)
        
        # Model configuration
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.logarithmic_neurons = logarithmic_neurons
        self.ensemble_dnn = ensemble_dnn
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        
        # Set default hidden units
        self.afn_hidden_units = afn_hidden_units or [64, 64, 64]
        self.dnn_hidden_units = dnn_hidden_units or [64, 64, 64]
        
        # Build model components
        self._build_model(afn_activation, afn_dropout, dnn_activation, dnn_dropout)
    
    def _build_model(self, afn_activation, afn_dropout, dnn_activation, dnn_dropout):
        """Build model components."""
        # Embedding layer for AFN
        self.feature_embedding = layers.Embedding(
            input_dim=self.feature_size,
            output_dim=self.embedding_size,
            embeddings_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name='afn_feature_embeddings'
        )
        
        # Logarithmic transformation network components
        # Coefficient W: transforms from num_fields to logarithmic_neurons
        self.coefficient_W = layers.Dense(
            units=self.logarithmic_neurons,
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name='coefficient_W'
        )
        
        # Batch normalization layers for logarithmic network
        if self.use_batch_norm:
            self.log_batch_norm = layers.BatchNormalization(name='log_batch_norm')
            self.exp_batch_norm = layers.BatchNormalization(name='exp_batch_norm')
        
        # AFN dense network
        self.afn_dense_layers = []
        for i, units in enumerate(self.afn_hidden_units):
            self.afn_dense_layers.append(
                layers.Dense(
                    units=units,
                    activation=afn_activation,
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
                    name=f'afn_dense_{i}'
                )
            )
            if afn_dropout > 0:
                self.afn_dense_layers.append(layers.Dropout(afn_dropout, name=f'afn_dropout_{i}'))
            if self.use_batch_norm:
                self.afn_dense_layers.append(layers.BatchNormalization(name=f'afn_bn_{i}'))
        
        # AFN output layer
        self.afn_output = layers.Dense(
            units=1,
            activation=None,
            kernel_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name='afn_output'
        )
        
        # Ensemble DNN components (optional)
        if self.ensemble_dnn:
            # Separate embedding layer for DNN
            self.dnn_feature_embedding = layers.Embedding(
                input_dim=self.feature_size,
                output_dim=self.embedding_size,
                embeddings_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
                name='dnn_feature_embeddings'
            )
            
            # DNN layers
            self.dnn_layers = []
            for i, units in enumerate(self.dnn_hidden_units):
                self.dnn_layers.append(
                    layers.Dense(
                        units=units,
                        activation=dnn_activation,
                        kernel_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
                        name=f'dnn_dense_{i}'
                    )
                )
                if dnn_dropout > 0:
                    self.dnn_layers.append(layers.Dropout(dnn_dropout, name=f'dnn_dropout_{i}'))
                if self.use_batch_norm:
                    self.dnn_layers.append(layers.BatchNormalization(name=f'dnn_bn_{i}'))
            
            # DNN output layer
            self.dnn_output = layers.Dense(
                units=1,
                activation=None,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
                name='dnn_output'
            )
            
            # Final fusion layer
            self.fusion_layer = layers.Dense(
                units=1,
                activation=None,
                name='fusion_output'
            )
    
    def logarithmic_net(self, feature_emb, training=None):
        """
        Logarithmic Transformation Network.
        
        Args:
            feature_emb: Feature embeddings [batch_size, num_fields, embedding_dim]
            training: Training mode flag
            
        Returns:
            Transformed features [batch_size, logarithmic_neurons * embedding_dim]
        """
       
        feature_emb = tf.abs(feature_emb)
        feature_emb = tf.maximum(feature_emb, 1e-5)
        
        
        log_feature_emb = tf.math.log(feature_emb)  # [batch_size, num_fields, embedding_dim]
        

        if self.use_batch_norm:
            log_feature_emb = self.log_batch_norm(log_feature_emb, training=training)
        
        # Transpose to [batch_size, embedding_dim, num_fields] for transformation
        log_feature_emb_transposed = tf.transpose(log_feature_emb, perm=[0, 2, 1])
        
        # Apply coefficient_W: [batch_size, embedding_dim, num_fields] -> [batch_size, embedding_dim, logarithmic_neurons]
        logarithmic_out = self.coefficient_W(log_feature_emb_transposed)
        
        # Transpose back to [batch_size, logarithmic_neurons, embedding_dim]
        logarithmic_out = tf.transpose(logarithmic_out, perm=[0, 2, 1])
        
        cross_out = tf.exp(logarithmic_out)
    
        if self.use_batch_norm:
            cross_out = self.exp_batch_norm(cross_out, training=training)
        
        # [batch_size, logarithmic_neurons, embedding_dim] -> [batch_size, logarithmic_neurons * embedding_dim]
        concat_out = tf.reshape(cross_out, [-1, self.logarithmic_neurons * self.embedding_size])
        
        return concat_out
    
    def call(self, inputs, training=None):
        """
        Forward pass of AFN model.
        
        Args:
            inputs: Dictionary with 'feat_index' key
            training: Training mode flag
            
        Returns:
            Logits (before sigmoid activation)
        """
        feature_index = inputs['feat_index']  # [batch_size, field_size]
        feat_value = inputs['feat_value']
        
        # AFN path
        # Get embeddings
        feature_emb = self.feature_embedding(feature_index)  # [batch_size, field_size, embedding_size]
        
        feat_value_expanded = tf.expand_dims(feat_value, axis=-1)  # [batch_size, field_size, 1]
        feature_emb = feature_emb * feat_value_expanded

        # Apply logarithmic transformation network
        afn_input = self.logarithmic_net(feature_emb, training=training)
        
        # Pass through AFN dense layers
        afn_hidden = afn_input
        for layer in self.afn_dense_layers:
            afn_hidden = layer(afn_hidden, training=training)
        
        # Get AFN output
        afn_out = self.afn_output(afn_hidden)
        
        # Ensemble DNN path (optional)
        if self.ensemble_dnn:
            # Get separate embeddings for DNN
            dnn_feature_emb = self.dnn_feature_embedding(feature_index)
            
            feat_value_expanded = tf.expand_dims(feat_value, axis=-1)  # [batch_size, field_size, 1]
            dnn_feature_emb = dnn_feature_emb * feat_value_expanded  # [batch_size, field_size, embedding_size]
            
            # Flatten embeddings
            dnn_input = tf.reshape(dnn_feature_emb, [-1, self.field_size * self.embedding_size])
            
            # Pass through DNN layers
            dnn_hidden = dnn_input
            for layer in self.dnn_layers:
                dnn_hidden = layer(dnn_hidden, training=training)
            
            # Get DNN output
            dnn_out = self.dnn_output(dnn_hidden)
            
            # Fuse AFN and DNN outputs
            combined = tf.concat([afn_out, dnn_out], axis=-1)
            logits = self.fusion_layer(combined)
        else:
            logits = afn_out
        
        return logits


class AFNTrainer:
    """
    Trainer class for AFN model using Keras fit method.
    Compatible with AutoInt trainer pattern.
    """
    
    def __init__(self,
                 model: AFN,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 learning_rate_wide: Optional[float] = None,
                 patience: int = 3,
                 save_path: Optional[str] = None,
                 verbose: int = 1):
        """
        Initialize AFN trainer.
        
        Args:
            model: AFN model instance
            optimizer: Optimizer type ('adam', 'sgd', etc.)
            learning_rate: Learning rate
            learning_rate_wide: Learning rate for wide part (not used in AFN)
            patience: Early stopping patience
            save_path: Path to save model checkpoints
            verbose: Verbosity level
        """
        self.model = model
        self.patience = patience
        self.save_path = save_path
        self.verbose = verbose
        
        # Create optimizer
        if optimizer.lower() == 'adam':
            self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer.lower() == 'adagrad':
            self.optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Compile model
        self.model.compile(
            optimizer=self.optimizer,
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.AUC(from_logits=True, name='auc')]
        )
        
        # Early stopping and checkpointing
        self.best_val_auc = 0.0
        self.wait = 0
        self.best_weights = None
        
        # Create save directory if needed
        if self.save_path and not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
    
    def _separate_features_labels(self, dataset):
        """
        Convert dataset with labels to (x, y) format for Keras fit.
        
        Args:
            dataset: tf.data.Dataset with dict containing 'labels'
            
        Returns:
            Dataset in (x, y) format
        """
        def extract_x_y(batch):
            if isinstance(batch, dict) and 'labels' in batch:
                features = {k: v for k, v in batch.items() if k != 'labels'}
                labels = batch['labels']
                return features, labels
            return batch
        
        return dataset.map(extract_x_y)
    
    def fit(self,
            train_dataset,
            validation_dataset=None,
            epochs: int = 1,
            steps_per_epoch: Optional[int] = None) -> bool:
        """
        Train model for one epoch using Keras fit.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Number of epochs (usually 1 for incremental training)
            steps_per_epoch: Steps per epoch
            
        Returns:
            True if training should continue, False if early stopping triggered
        """
        # Separate features and labels
        train_data = self._separate_features_labels(train_dataset)
        val_data = self._separate_features_labels(validation_dataset) if validation_dataset else None
        
        # Train for one epoch
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=self.verbose
        )
        
        # Check early stopping
        if validation_dataset is not None:
            val_auc = history.history['val_auc'][-1]
            
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.wait = 0
                # Save best weights
                self.best_weights = self.model.get_weights()
                if self.save_path:
                    self.model.save_weights(os.path.join(self.save_path, 'best_model.weights.h5'))
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    if self.verbose > 0:
                        print(f"Early stopping triggered. Best val AUC: {self.best_val_auc:.4f}")
                    return False
        
        return True
    
    def evaluate(self, test_dataset) -> Tuple[float, float]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Tuple of (test_loss, test_auc)
        """
        test_data = self._separate_features_labels(test_dataset)
        results = self.model.evaluate(test_data, verbose=self.verbose)
        
        # results is [loss, auc]
        test_loss = results[0]
        test_auc = results[1]
        
        return test_loss, test_auc
    
    def load_best_weights(self) -> bool:
        """
        Load best model weights.
        
        Returns:
            True if weights loaded successfully
        """
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            return True
        elif self.save_path and os.path.exists(os.path.join(self.save_path, 'best_model.weights.h5')):
            self.model.load_weights(os.path.join(self.save_path, 'best_model.weights.h5'))
            return True
        return False
