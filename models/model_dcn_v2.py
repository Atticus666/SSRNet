import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import roc_auc_score, log_loss
from typing import Optional, List, Tuple, Dict, Any
import math

class CrossNetV2(layers.Layer):
    
    def __init__(self, 
                 input_dim: int,
                 num_layers: int = 2,
                 l2_reg: float = 0.0,
                 **kwargs):

        super(CrossNetV2, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.l2_reg = l2_reg
        
        # Cross transformation layers
        self.cross_layers = []
        for i in range(num_layers):
            layer = layers.Dense(
                input_dim,
                kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
                kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                bias_initializer='zeros',
                name=f'cross_layer_{i}'
            )
            self.cross_layers.append(layer)
    
    def call(self, inputs, training=None):

        x0 = inputs  # Original input
        xi = inputs  # Current layer input
        
        for i in range(self.num_layers):
            # Cross transformation: xi+1 = x0 ⊙ (W·xi + b) + xi
            xl = self.cross_layers[i](xi)  # Linear transformation
            xi = x0 * xl + xi  # Element-wise product and residual
            
        return xi

class CrossNetMix(layers.Layer):
    
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 2,
                 low_rank: int = 32,
                 num_experts: int = 4,
                 l2_reg: float = 0.0,
                 **kwargs):

        super(CrossNetMix, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.l2_reg = l2_reg
        
        # Initialize expert networks for each layer
        self.U_layers = []  # List of U matrices for each layer
        self.V_layers = []  # List of V matrices for each layer
        self.C_layers = []  # List of C matrices for each layer
        self.gating_layers = []  # Gating networks
        self.bias_layers = []  # Bias terms
        
        regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
        
        for i in range(num_layers):
            # U matrix: [num_experts, input_dim, low_rank]
            U = self.add_weight(
                name=f'U_{i}',
                shape=(num_experts, input_dim, low_rank),
                initializer='glorot_uniform',
                regularizer=regularizer,
                trainable=True
            )
            self.U_layers.append(U)
            
            # V matrix: [num_experts, low_rank, input_dim]  
            V = self.add_weight(
                name=f'V_{i}',
                shape=(num_experts, low_rank, input_dim),
                initializer='glorot_uniform',
                regularizer=regularizer,
                trainable=True
            )
            self.V_layers.append(V)
            
            # C matrix: [num_experts, low_rank, low_rank]
            C = self.add_weight(
                name=f'C_{i}',
                shape=(num_experts, low_rank, low_rank),
                initializer='glorot_uniform',
                regularizer=regularizer,
                trainable=True
            )
            self.C_layers.append(C)
            
            # Gating network
            gating = layers.Dense(
                num_experts,
                activation='softmax',
                kernel_initializer='glorot_uniform',
                name=f'gating_{i}'
            )
            self.gating_layers.append(gating)
            
            # Bias
            bias = self.add_weight(
                name=f'bias_{i}',
                shape=(input_dim,),
                initializer='zeros',
                trainable=True
            )
            self.bias_layers.append(bias)
    
    def call(self, inputs, training=None):

        x0 = inputs  # Original input
        xi = inputs  # Current layer input
        
        for i in range(self.num_layers):
            # Generate gating weights
            gating_weights = self.gating_layers[i](xi)  # [batch_size, num_experts]
            gating_weights = tf.expand_dims(gating_weights, axis=-1)  # [batch_size, num_experts, 1]
            
            # Apply expert transformations
            expert_outputs = []
            for e in range(self.num_experts):
                # Low-rank transformation: U * (C * (V * x))
                # V_layers[i][e] shape: [low_rank, input_dim], need transpose for matmul
                Vx = tf.matmul(tf.expand_dims(xi, axis=1), tf.expand_dims(self.V_layers[i][e], axis=0), transpose_b=True)  # [batch_size, 1, low_rank]
                CVx = tf.matmul(Vx, tf.expand_dims(self.C_layers[i][e], axis=0))  # [batch_size, 1, low_rank]  
                UCVx = tf.matmul(CVx, tf.expand_dims(self.U_layers[i][e], axis=0), transpose_b=True)  # [batch_size, 1, input_dim]
                expert_outputs.append(tf.squeeze(UCVx, axis=1))  # [batch_size, input_dim]
            
            # Stack expert outputs
            expert_outputs = tf.stack(expert_outputs, axis=1)  # [batch_size, num_experts, input_dim]
            
            # Weighted combination
            weighted_output = tf.reduce_sum(expert_outputs * gating_weights, axis=1)  # [batch_size, input_dim]
            
            # Add bias and residual connection
            xi = x0 * (weighted_output + self.bias_layers[i]) + xi
            
        return xi

class DNNLayer(layers.Layer):
    """
    Deep Neural Network component for DCN v2.
    """
    
    def __init__(self,
                 hidden_units: List[int],
                 activation: str = 'relu',
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = False,
                 l2_reg: float = 0.0,
                 **kwargs):

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

        x = inputs
        for layer in self.dnn_layers:
            if isinstance(layer, (layers.Dropout, layers.BatchNormalization)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

class DCNV2(keras.Model):
    
    def __init__(self,
                 feature_size: int,
                 field_size: int,
                 embedding_size: int = 16,
                 cross_layers: int = 2,
                 deep_layers: Optional[List[int]] = None,
                 is_stacked: bool = False,
                 use_low_rank_mixture: bool = True,
                 low_rank: int = 32,
                 num_experts: int = 4,
                 l2_reg_embedding: float = 0.0,
                 l2_reg_cross: float = 0.0,
                 l2_reg_deep: float = 0.0,
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = False,
                 **kwargs):
        """
        Initialize DCN v2 model.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields  
            embedding_size: Embedding dimension
            cross_layers: Number of cross layers
            deep_layers: Hidden dimensions for deep network
            is_stacked: Whether to use stacked (True) or parallel (False) architecture
            use_low_rank_mixture: Whether to use mixture of experts in cross network
            low_rank: Low-rank dimension for MOE
            num_experts: Number of experts for MOE
            l2_reg_embedding: L2 regularization for embeddings
            l2_reg_cross: L2 regularization for cross network
            l2_reg_deep: L2 regularization for deep network
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(DCNV2, self).__init__(**kwargs)
        
        # Model configuration
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.cross_layers = cross_layers
        self.deep_layers = deep_layers or [256, 256]
        self.is_stacked = is_stacked
        self.use_low_rank_mixture = use_low_rank_mixture
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_cross = l2_reg_cross
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
        
        # Cross network
        if self.use_low_rank_mixture:
            self.cross_net = CrossNetMix(
                input_dim=self.input_dim,
                num_layers=self.cross_layers,
                low_rank=self.low_rank,
                num_experts=self.num_experts,
                l2_reg=self.l2_reg_cross
            )
        else:
            self.cross_net = CrossNetV2(
                input_dim=self.input_dim,
                num_layers=self.cross_layers,
                l2_reg=self.l2_reg_cross
            )
        
        # Deep network
        self.deep_net = DNNLayer(
            hidden_units=self.deep_layers,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            l2_reg=self.l2_reg_deep
        )
        
        # Output layer
        if self.is_stacked:
            # Stacked architecture: Cross -> Deep -> Output
            self.output_layer = layers.Dense(
                1,
                activation='sigmoid',
                kernel_initializer=keras.initializers.GlorotNormal(),
                name='output'
            )
        else:
            # Parallel architecture: [Cross, Deep] -> Output
            self.output_layer = layers.Dense(
                1,
                activation='sigmoid',
                kernel_initializer=keras.initializers.GlorotNormal(),
                name='output'
            )
    
    def call(self, inputs, training=None):

        feat_index = inputs['feat_index']  # [batch_size, field_size]
        feat_value = inputs['feat_value']  # [batch_size, field_size]
     
        # Embedding lookup
        embeddings = self.feature_embedding(feat_index)  # [batch_size, field_size, embedding_size]
        feat_value = tf.expand_dims(feat_value, axis=-1)  # [batch_size, field_size, 1]
        embeddings = embeddings * feat_value  # Element-wise multiplication
        embeddings = self.embedding_dropout(embeddings, training=training)
        
        # Flatten embeddings
        flat_embeddings = tf.reshape(embeddings, [-1, self.input_dim])  # [batch_size, field_size * embedding_size]
        
        # Cross network
        cross_output = self.cross_net(flat_embeddings, training=training)
        
        if self.is_stacked:
            # Stacked architecture
            deep_output = self.deep_net(cross_output, training=training)
            output = self.output_layer(deep_output)
        else:
            # Parallel architecture
            deep_output = self.deep_net(flat_embeddings, training=training)
            combined_output = tf.concat([cross_output, deep_output], axis=-1)
            output = self.output_layer(combined_output)
        
        return output
    
    def train_step(self, data):

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
            'cross_layers': self.cross_layers,
            'deep_layers': self.deep_layers,
            'is_stacked': self.is_stacked,
            'use_low_rank_mixture': self.use_low_rank_mixture,
            'low_rank': self.low_rank,
            'num_experts': self.num_experts,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
        })
        return config

class DCNV2Trainer:
    """
    Trainer class for DCN v2 model.
    """
    
    def __init__(self,
                 model: DCNV2,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
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
    
    def fit(self,
            train_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset,
            epochs: int = 50,
            steps_per_epoch: Optional[int] = None) -> bool:

        # Train for one epoch
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=1,
            steps_per_epoch=steps_per_epoch,
            verbose=self.verbose
        )
        
        # Get validation AUC
        val_auc = history.history['val_auc'][0]
        
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
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Tuple[float, float]:

        results = self.model.evaluate(test_dataset, verbose=self.verbose)
        test_loss, test_auc = results[0], results[1]
        return test_loss, test_auc
    
    def load_best_weights(self) -> bool:

        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            return True
        elif self.save_path and tf.io.gfile.exists(self.save_path + '.index'):
            self.model.load_weights(self.save_path)
            return True
        else:
            return False