import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from typing import Optional, List, Tuple, Dict, Any
import os


class DNNLayer(layers.Layer):
    def __init__(self,
                 hidden_units: List[int],
                 activation: str = 'relu',
                 dropout_rate: float = 0.0,
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


class FactorizationMachineBlock(layers.Layer):

    def __init__(self,
                 input_features: int,
                 output_features: int,
                 embedding_dim: int,
                 rank_k: Optional[int] = 8,
                 mlp_hidden_units: Optional[List[int]] = None,
                 mlp_hidden_activations: str = "relu",
                 mlp_dropout: float = 0.0,
                 **kwargs):
        """
        Initialize FMB.
        
        Args:
            input_features: Number of input features
            output_features: Number of output features
            embedding_dim: Embedding dimension
            rank_k: Rank for optimized FM (None for vanilla FM)
            mlp_hidden_units: Hidden units for MLP
            mlp_hidden_activations: Activation for MLP
            mlp_dropout: Dropout rate for MLP
        """
        super(FactorizationMachineBlock, self).__init__(**kwargs)
        self.input_features = input_features
        self.output_features = output_features
        self.embedding_dim = embedding_dim
        self.rank_k = rank_k
        self.mlp_hidden_units = mlp_hidden_units or [16, 16]
        
        # Projection matrix for optimized FM
        if rank_k is not None:
            self.proj_Y = self.add_weight(
                name='proj_Y',
                shape=(self.input_features, rank_k),
                initializer=keras.initializers.RandomNormal(),
                trainable=True
            )
            self.fm_out_dim = input_features * rank_k
        else:
            self.fm_out_dim = input_features * input_features
        
        # Layer normalization
        self.layer_norm = layers.LayerNormalization(axis=-1, name='fmb_layer_norm')
        
        # MLP
        self.mlp = DNNLayer(
            hidden_units=self.mlp_hidden_units,
            activation=mlp_hidden_activations,
            dropout_rate=mlp_dropout
        )
        
        # Output layer
        self.output_layer = layers.Dense(
            output_features * embedding_dim,
            activation='relu',
            kernel_initializer=keras.initializers.GlorotNormal(),
            name='fmb_output'
        )
    
    def call(self, x, training=None):
        """
        Forward pass through FMB.
        
        Args:
            x: Input tensor [batch_size, input_features, embedding_dim]
            training: Training mode flag
            
        Returns:
            Output tensor [batch_size, output_features, embedding_dim]
        """
        # Optimized FM
        flatten_fm = self.optimized_fm(x)
        
        # Layer normalization
        mlp_in = self.layer_norm(flatten_fm, training=training)
        
        # MLP
        mlp_out = self.mlp(mlp_in, training=training)
        
        # Output
        output = self.output_layer(mlp_out)
        
        # Reshape to [batch_size, output_features, embedding_dim]
        output = tf.reshape(output, [-1, self.output_features, self.embedding_dim])
        
        return output
    
    def optimized_fm(self, x):
        """
        Optimized factorization machine operation.
        
        Args:
            x: Input tensor [batch_size, n, d]
            
        Returns:
            Flattened FM matrix [batch_size, n*k] or [batch_size, n*n]
        """
        if self.rank_k is not None:
            # Optimized FM with rank-k projection
            # x: [batch_size, n, d], proj_Y: [n, k]
            # x_transposed: [batch_size, d, n]
            x_transposed = tf.transpose(x, perm=[0, 2, 1])
            
            # projected: [batch_size, d, k]
            projected = tf.matmul(x_transposed, self.proj_Y)
            
            # fm_matrix: [batch_size, n, k]
            fm_matrix = tf.matmul(x, projected)
        else:
            # Vanilla FM
            # fm_matrix: [batch_size, n, n]
            x_transposed = tf.transpose(x, perm=[0, 2, 1])
            fm_matrix = tf.matmul(x, x_transposed)
        
        # Flatten - use static dimension
        return tf.reshape(fm_matrix, [-1, self.fm_out_dim])


class LinearCompressionBlock(layers.Layer):

    def __init__(self, input_features: int, output_features: int, **kwargs):
        """
        Initialize LCB.
        
        Args:
            input_features: Number of input features
            output_features: Number of output features
        """
        super(LinearCompressionBlock, self).__init__(**kwargs)
        self.input_features = input_features
        self.output_features = output_features
        
        self.linear = layers.Dense(
            output_features,
            use_bias=False,
            kernel_initializer=keras.initializers.GlorotNormal(),
            name='lcb_linear'
        )
    
    def call(self, x, training=None):
        """
        Forward pass through LCB.
        
        Args:
            x: Input tensor [batch_size, input_features, embedding_dim]
            training: Training mode flag
            
        Returns:
            Output tensor [batch_size, output_features, embedding_dim]
        """
        # Transpose to [batch_size, embedding_dim, input_features]
        x_transposed = tf.transpose(x, perm=[0, 2, 1])
        
        # Linear: [batch_size, embedding_dim, output_features]
        out = self.linear(x_transposed)
        
        # Transpose back to [batch_size, output_features, embedding_dim]
        out = tf.transpose(out, perm=[0, 2, 1])
        
        return out


class WuKongLayer(layers.Layer):

    def __init__(self,
                 input_features: int,
                 lcb_features: int = 8,
                 fmb_features: int = 8,
                 embedding_dim: int = 16,
                 fmp_rank_k: int = 4,
                 fmb_mlp_units: Optional[List[int]] = None,
                 fmb_mlp_activations: str = "relu",
                 fmb_dropout: float = 0.1,
                 layer_norm: bool = True,
                 **kwargs):
        """
        Initialize WuKong layer.
        
        Args:
            input_features: Number of input features
            lcb_features: Number of LCB output features
            fmb_features: Number of FMB output features
            embedding_dim: Embedding dimension
            fmp_rank_k: Rank for FM projection
            fmb_mlp_units: Hidden units for FMB MLP
            fmb_mlp_activations: Activation for FMB MLP
            fmb_dropout: Dropout rate for FMB
            layer_norm: Whether to use layer normalization
        """
        super(WuKongLayer, self).__init__(**kwargs)
        self.input_features = input_features
        self.lcb_features = lcb_features
        self.fmb_features = fmb_features
        self.embedding_dim = embedding_dim
        self.use_layer_norm = layer_norm
        
        # FMB
        self.fmb = FactorizationMachineBlock(
            input_features=input_features,
            output_features=fmb_features,
            embedding_dim=embedding_dim,
            rank_k=fmp_rank_k,
            mlp_hidden_units=fmb_mlp_units or [16, 16],
            mlp_hidden_activations=fmb_mlp_activations,
            mlp_dropout=fmb_dropout
        )
        
        # LCB
        self.lcb = LinearCompressionBlock(
            input_features=input_features,
            output_features=lcb_features
        )
        
        # Layer normalization
        if layer_norm:
            self.layer_norm = layers.LayerNormalization(name='wukong_layer_norm')
        
        # Residual projection if dimensions don't match
        if input_features != lcb_features + fmb_features:
            self.residual_proj = layers.Dense(
                lcb_features + fmb_features,
                use_bias=True,
                kernel_initializer=keras.initializers.GlorotNormal(),
                name='residual_proj'
            )
        else:
            self.residual_proj = None
    
    def call(self, x, training=None):
        """
        Forward pass through WuKong layer.
        
        Args:
            x: Input tensor [batch_size, input_features, embedding_dim]
            training: Training mode flag
            
        Returns:
            Output tensor [batch_size, lcb_features + fmb_features, embedding_dim]
        """
        # FMB output
        fmb_out = self.fmb(x, training=training)
        
        # LCB output
        lcb_out = self.lcb(x, training=training)
        
        # Concatenate
        concat_out = tf.concat([fmb_out, lcb_out], axis=1)
        
        # Residual connection
        out = self.residual(concat_out, x)
        
        # Layer normalization
        if self.use_layer_norm:
            out = self.layer_norm(out, training=training)
        
        return out
    
    def residual(self, out, x):
        """
        Apply residual connection.
        
        Args:
            out: Output tensor
            x: Input tensor
            
        Returns:
            Residual output
        """
        if self.residual_proj is not None:
            x_transposed = tf.transpose(x, perm=[0, 2, 1])
            res = self.residual_proj(x_transposed)
            res = tf.transpose(res, perm=[0, 2, 1])
        else:
            res = x
        
        return out + res


class Wukong(keras.Model):

    def __init__(self,
                 feature_size: int,
                 field_size: int,
                 embedding_size: int = 16,
                 num_wukong_layers: int = 3,
                 lcb_features: int = 40,
                 fmb_features: int = 40,
                 fmb_mlp_units: Optional[List[int]] = None,
                 fmb_mlp_activations: str = "relu",
                 fmp_rank_k: int = 8,
                 mlp_hidden_units: Optional[List[int]] = None,
                 mlp_hidden_activations: str = 'relu',
                 mlp_batch_norm: bool = True,
                 layer_norm: bool = True,
                 net_dropout: float = 0.0,
                 l2_reg_embedding: float = 0.0,
                 l2_reg_deep: float = 0.0,
                 **kwargs):
        """
        Initialize WuKong model.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields
            embedding_size: Embedding dimension
            num_wukong_layers: Number of WuKong layers
            lcb_features: Number of LCB features
            fmb_features: Number of FMB features
            fmb_mlp_units: Hidden units for FMB MLP
            fmb_mlp_activations: Activation for FMB MLP
            fmp_rank_k: Rank for FM projection
            mlp_hidden_units: Hidden units for output MLP
            mlp_hidden_activations: Activation for output MLP
            mlp_batch_norm: Whether to use batch norm in output MLP
            layer_norm: Whether to use layer normalization
            net_dropout: Dropout rate
            l2_reg_embedding: L2 regularization for embeddings
            l2_reg_deep: L2 regularization for deep network
        """
        super(Wukong, self).__init__(**kwargs)
        
        # Model configuration
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.num_wukong_layers = num_wukong_layers
        self.lcb_features = lcb_features
        self.fmb_features = fmb_features
        self.fmb_mlp_units = fmb_mlp_units or [32, 32]
        self.fmb_mlp_activations = fmb_mlp_activations
        self.fmp_rank_k = fmp_rank_k
        self.mlp_hidden_units = mlp_hidden_units or [32, 32]
        self.mlp_hidden_activations = mlp_hidden_activations
        self.mlp_batch_norm = mlp_batch_norm
        self.net_dropout = net_dropout
        self.layer_norm = layer_norm
        self.l2_reg_embedding = l2_reg_embedding
        self.l2_reg_deep = l2_reg_deep
        
        # Calculate dimensions
        self.output_features = lcb_features + fmb_features
        self.final_dim = self.output_features * embedding_size
        
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
        self.embedding_dropout = layers.Dropout(self.net_dropout)
        
        # Build WuKong layers stack
        self.wukong_layers = []
        for i in range(self.num_wukong_layers):
            input_features = self.field_size if i == 0 else self.output_features
            
            wukong_layer = WuKongLayer(
                input_features=input_features,
                lcb_features=self.lcb_features,
                fmb_features=self.fmb_features,
                embedding_dim=self.embedding_size,
                fmp_rank_k=self.fmp_rank_k,
                fmb_mlp_units=self.fmb_mlp_units,
                fmb_mlp_activations=self.fmb_mlp_activations,
                fmb_dropout=self.net_dropout,
                layer_norm=self.layer_norm,
                name=f'wukong_layer_{i}'
            )
            self.wukong_layers.append(wukong_layer)
        
        # Output MLP
        self.output_mlp = DNNLayer(
            hidden_units=self.mlp_hidden_units,
            activation=self.mlp_hidden_activations,
            dropout_rate=self.net_dropout,
            use_batch_norm=self.mlp_batch_norm,
            l2_reg=self.l2_reg_deep
        )
        
        # Final output layer
        self.output_layer = layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer=keras.initializers.GlorotNormal(),
            name='output'
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass through WuKong.
        
        Args:
            inputs: Input dictionary or tensor
                - If dict: {'feat_index': tensor, 'feat_value': tensor}
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
        
        # WuKong layers stack
        wukong_out = embeddings
        for wukong_layer in self.wukong_layers:
            wukong_out = wukong_layer(wukong_out, training=training)
        
        # Flatten for MLP
        flat_output = tf.reshape(wukong_out, [-1, self.final_dim])
        
        # Output MLP
        mlp_output = self.output_mlp(flat_output, training=training)
        
        # Final output
        output = self.output_layer(mlp_output)
        
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
            'num_wukong_layers': self.num_wukong_layers,
            'lcb_features': self.lcb_features,
            'fmb_features': self.fmb_features,
            'net_dropout': self.net_dropout,
        })
        return config


class WukongTrainer:
    """
    Trainer class for WuKong model.
    """
    
    def __init__(self,
                 model: Wukong,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 patience: int = 5,
                 save_path: Optional[str] = None,
                 verbose: int = 1):
        """
        Initialize trainer.
        
        Args:
            model: WuKong model instance
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
