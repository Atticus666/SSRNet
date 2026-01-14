import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, log_loss
from typing import Optional, List, Tuple, Dict, Any
import os
from itertools import combinations


def xavier_initializer(shape):
    limit = np.sqrt(6.0 / np.sum(shape))
    return keras.initializers.RandomUniform(minval=-limit, maxval=limit)


def generate_pairs(num_fields, mask=None, order=2):
    ranges = list(range(num_fields))
    res = [[] for _ in range(order)]
    
    for i, pair in enumerate(list(combinations(ranges, order))):
        if mask is None or mask[i] == 1:
            for j in range(order):
                res[j].append(pair[j])
    
    return res


class AutoFISLayer(layers.Layer):
    def __init__(self,
                 num_fields: int,
                 feature_size: int,
                 embedding_size: int = 16,
                 mlp_width: int = 256,
                 mlp_depth: int = 3,
                 num_pairs: Optional[int] = None,
                 use_bn: bool = True,
                 dropout_rate: float = 0.0,
                 l2_reg: float = 0.0,
                 trainable_mask: bool = True,  # 新增：控制 mask 是否可训练
                 fixed_mask: Optional[np.ndarray] = None,  # 新增：固定的 mask 值
                 **kwargs):
        super(AutoFISLayer, self).__init__(**kwargs)
        
        self.num_fields = num_fields
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.trainable_mask = trainable_mask
        self.fixed_mask = fixed_mask
        
        # Compute number of pairs
        if num_pairs is None:
            self.num_pairs = num_fields * (num_fields - 1) // 2
        else:
            self.num_pairs = num_pairs
    
    def build(self, input_shape):
        """Build layer weights."""
        regularizer = keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
        
        # 1. Embedding layers
        self.w_embedding = layers.Embedding(
            self.feature_size,
            1,
            embeddings_initializer=xavier_initializer([self.feature_size, 1]),
            embeddings_regularizer=regularizer,
            name='linear_embedding'
        )
        
        self.v_embedding = layers.Embedding(
            self.feature_size,
            self.embedding_size,
            embeddings_initializer=xavier_initializer([self.feature_size, self.embedding_size]),
            embeddings_regularizer=regularizer,
            name='fm_embedding'
        )
        
        # 2. MLP layers
        self.mlp_layers = []
        self.bn_layers = []
        self.dropout_layers = []
        if self.mlp_depth > 0:
            for i in range(self.mlp_depth + 1):
                if i == self.mlp_depth:
                    # Last layer
                    output_dim = 1
                else:
                    output_dim = self.mlp_width
                
                dense = layers.Dense(
                    output_dim,
                    activation='relu' if i < self.mlp_depth else None,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizer,
                    use_bias=True,
                    name=f'mlp_dense_{i}'
                )
                self.mlp_layers.append(dense)
                
                if i < self.mlp_depth:
                    if self.use_bn:
                        bn = layers.BatchNormalization(name=f'mlp_bn_{i}')
                    else:
                        bn = layers.Lambda(lambda x: x, name=f'mlp_identity_{i}')
                    self.bn_layers.append(bn)
                    
                    dropout = layers.Dropout(self.dropout_rate, name=f'mlp_dropout_{i}')
                    self.dropout_layers.append(dropout)
        
        # 3. Feature interaction selection mask
        if self.fixed_mask is not None:
            self.interaction_mask = self.add_weight(
                name='interaction_mask',
                shape=(1, self.num_pairs),
                initializer=keras.initializers.Constant(self.fixed_mask.reshape(1, -1)),
                trainable=False  # 不可训练
            )
        else:
            self.interaction_mask = self.add_weight(
                name='interaction_mask',
                shape=(1, self.num_pairs),
                initializer=keras.initializers.RandomUniform(0.599, 0.601),
                trainable=self.trainable_mask
            )
        
        # Batch normalization for FM interactions
        if self.use_bn:
            self.fm_bn = layers.BatchNormalization(name='fm_bn')
        else:
            self.fm_bn = layers.Lambda(lambda x: x, name='fm_identity')

        # Output layer
        self.output_layer = layers.Dense(
            1, 
            activation=None, 
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            name='output'
        )
    
        super(AutoFISLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """Forward pass."""
        feat_index = inputs['feat_index']
        feat_value = inputs['feat_value']
        
        # 1. Linear part
        xw = self.w_embedding(feat_index)
        xw = tf.squeeze(xw, axis=-1)
        xw = xw * feat_value
        linear_out = tf.reduce_sum(xw, axis=1)
        
        # 2. Embedding for FM
        xv = self.v_embedding(feat_index)
        xv = xv * tf.expand_dims(feat_value, axis=-1)
        
        # 3. MLP part
        if self.mlp_depth > 0:
            mlp_input = tf.reshape(xv, [-1, self.num_fields * self.embedding_size])
            deep_out = mlp_input
            for i in range(self.mlp_depth):
                deep_out = self.mlp_layers[i](deep_out)
                # deep_out = self.bn_layers[i](deep_out, training=training)
                deep_out = self.dropout_layers[i](deep_out, training=training)
            deep_out = self.mlp_layers[self.mlp_depth](deep_out)
            deep_out = tf.squeeze(deep_out, axis=-1)
        else:
            deep_out = tf.constant(0.0)

        # 4. FM part with interaction selection
        rows, cols = generate_pairs(self.num_fields)
        left_embeddings = tf.gather(xv, cols, axis=1)
        right_embeddings = tf.gather(xv, rows, axis=1)
        
        pairwise_interactions = tf.reduce_sum(
            left_embeddings * right_embeddings, 
            axis=-1
        )
        
        pairwise_interactions = self.fm_bn(pairwise_interactions, training=training)
        pairwise_interactions = pairwise_interactions * self.interaction_mask
        fm_out = tf.reduce_sum(pairwise_interactions, axis=-1)
        
        # 5. Combine all parts
        logits = linear_out + fm_out + deep_out
        logits = tf.expand_dims(logits, axis=-1)
        predictions = self.output_layer(logits)
        
        return predictions
    
    def get_interaction_mask(self):
        return self.interaction_mask.numpy()


class AutoFIS(keras.Model):
    
    def __init__(self,
                 feature_size: int,
                 field_size: int,
                 embedding_size: int = 16,
                 mlp_width: int = 256,
                 mlp_depth: int = 3,
                 use_bn: bool = True,
                 dropout_rate: float = 0.0,
                 l2_reg: float = 0.0,
                 trainable_mask: bool = True,
                 fixed_mask: Optional[np.ndarray] = None,
                 **kwargs):
        super(AutoFIS, self).__init__(**kwargs)
        
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        self.autofis_layer = AutoFISLayer(
            num_fields=field_size,
            feature_size=feature_size,
            embedding_size=embedding_size,
            mlp_width=mlp_width,
            mlp_depth=mlp_depth,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            trainable_mask=trainable_mask,
            fixed_mask=fixed_mask
        )
    
    def call(self, inputs, training=None):

        return self.autofis_layer(inputs, training=training)
    
    def get_interaction_mask(self):
        return self.autofis_layer.get_interaction_mask()


class AutoFISTwoStageTrainer:
    
    def __init__(self,
                 feature_size: int,
                 field_size: int,
                 embedding_size: int = 16,
                 mlp_width: int = 256,
                 mlp_depth: int = 3,
                 use_bn: bool = True,
                 dropout_rate: float = 0.0,
                 l2_reg: float = 0.0,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 keep_ratio: float = 0.5,
                 early_stop_patience: int = 3,
                 save_path: Optional[str] = None,
                 verbose: int = 1):
        """
        Initialize two-stage trainer.
        
        Args:
            feature_size: Total number of features
            field_size: Number of feature fields
            embedding_size: Embedding dimension
            mlp_width: MLP hidden layer width
            mlp_depth: Number of MLP hidden layers
            use_bn: Whether to use batch normalization
            dropout_rate: Dropout rate
            l2_reg: L2 regularization coefficient
            optimizer: Optimizer type
            learning_rate: Learning rate
            keep_ratio: Ratio of interactions to keep after pruning
            save_path: Path to save models
            verbose: Verbosity level
        """
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate
        self.keep_ratio = keep_ratio
        self.patience = early_stop_patience
        self.save_path = save_path
        self.verbose = verbose
        
        # Early stopping variables (consistent with AutoInt)
        self.best_val_auc = -1.0
        self.wait = 0
        self.best_weights = None
        
        # Stage 1 model: trainable mask
        self.model_stage1 = AutoFIS(
            feature_size=feature_size,
            field_size=field_size,
            embedding_size=embedding_size,
            mlp_width=mlp_width,
            mlp_depth=mlp_depth,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            trainable_mask=True,
            fixed_mask=None
        )
        
        # Compile stage 1 model
        self._compile_model(self.model_stage1)
        
        # Stage 2 model will be created after pruning
        self.model_stage2 = None
        self.selected_mask = None
        
    def _get_optimizer(self):
        """Create optimizer instance."""
        if self.optimizer_type.lower() == 'adam':
            return keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_type.lower() == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        elif self.optimizer_type.lower() == 'sgd':
            return keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
    
    def _compile_model(self, model):
        """Compile model with optimizer and loss."""
        model.compile(
            optimizer=self._get_optimizer(),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.AUC(name='auc', from_logits=True)]
        )
    
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
    
    def train_stage1(self,
                     train_dataset,
                     epochs: int = 1,
                     steps_per_epoch: Optional[int] = None):
        # Convert datasets to (x, y) format
        train_ds = self._separate_features_labels(train_dataset)
        
        history = self.model_stage1.fit(
            train_ds,
            validation_data=None,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=self.verbose
        )
        
        return history
    
    def _transfer_weights_from_stage1(self):

        if self.verbose > 0:
            print("Transferring weights from Stage 1 to Stage 2...")
        

        stage1_weights = self.model_stage1.get_weights()
        stage2_weights = self.model_stage2.get_weights()
        
        stage1_weight_names = [w.name for w in self.model_stage1.weights]
        stage2_weight_names = [w.name for w in self.model_stage2.weights]
        
        transferred_count = 0
        for i, (s1_name, s1_weight) in enumerate(zip(stage1_weight_names, stage1_weights)):
            if 'interaction_mask' in s1_name:
                continue
            
            if i < len(stage2_weights) and s1_name in stage2_weight_names:
                j = stage2_weight_names.index(s1_name)
                
                if s1_weight.shape == stage2_weights[j].shape:
                    stage2_weights[j] = s1_weight
                    transferred_count += 1
        
        
        self.model_stage2.set_weights(stage2_weights)
        
        if self.verbose > 0:
            print(f"Transferred {transferred_count} weight tensors from Stage 1 to Stage 2\n")
    
    def prune_interactions(self):
        if self.verbose > 0:
            print("\n" + "="*80)
            print("Pruning Feature Interactions")
            print("="*80 + "\n")
         
        mask_values = self.model_stage1.get_interaction_mask().flatten()
        num_total = len(mask_values)
        num_keep = int(num_total * self.keep_ratio)
        
        threshold = np.sort(mask_values)[-num_keep]
        self.selected_mask = (mask_values >= threshold).astype(np.float32)
        
        num_selected = int(self.selected_mask.sum())
        
        if self.verbose > 0:
            print(f"Pruning: Keeping {num_selected}/{num_total} interactions ({num_selected/num_total*100:.1f}%)")
            print(f"Threshold: {threshold:.4f}, Min: {mask_values.min():.4f}, Max: {mask_values.max():.4f}\n")
        
        
        self.model_stage2 = AutoFIS(
            feature_size=self.feature_size,
            field_size=self.field_size,
            embedding_size=self.embedding_size,
            mlp_width=self.mlp_width,
            mlp_depth=self.mlp_depth,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            trainable_mask=False,  # 
            fixed_mask=self.selected_mask
        )
        
        self._compile_model(self.model_stage2)
        
        # self._transfer_weights_from_stage1()
        # if self.verbose > 0:
        #     print("Stage 2 model created and weights transferred from Stage 1\n")
        
        return self.selected_mask
    
    def check_early_stopping(self, val_auc: float) -> bool:

        # Check for improvement
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.wait = 0
            self.best_weights = self.model_stage2.get_weights()
            
            # Save best model
            if self.save_path:
                best_path = os.path.join(self.save_path, 'stage2_best_weights.h5')
                self.model_stage2.save_weights(best_path)
                if self.verbose > 0:
                    print(f"  New best validation AUC: {val_auc:.4f}")
            
            return True  # Continue training
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f"  No improvement. Wait: {self.wait}/{self.patience}")
            
            return self.wait < self.patience  # Continue if wait < patience
    
    def train_stage2(self,
                     train_dataset,
                     validation_dataset,
                     epochs: int = 1,
                     steps_per_epoch: Optional[int] = None):

        if self.model_stage2 is None:
            raise ValueError("Must call prune_interactions() before train_stage2()")
        
        # Convert datasets to (x, y) format
        train_ds = self._separate_features_labels(train_dataset)
        val_ds = self._separate_features_labels(validation_dataset)
        
        # Train on this part
        history = self.model_stage2.fit(
            train_ds,
            validation_data=None,  # Don't validate during fit
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=self.verbose
        )
        
        # Evaluate on validation set after training this part
        results = self.model_stage2.evaluate(val_ds, verbose=0)
        val_loss, val_auc = results[0], results[1]
        
        if self.verbose > 0:
            print(f"  Validation - AUC: {val_auc:.4f}, Loss: {val_loss:.4f}")
        
        return history, val_auc, val_loss
    
    def load_best_weights(self) -> bool:

        if self.best_weights is not None:
            self.model_stage2.set_weights(self.best_weights)
            return True
        elif self.save_path:
            best_path = os.path.join(self.save_path, 'stage2_best_weights.h5')
            if os.path.exists(best_path):
                self.model_stage2.load_weights(best_path)
                return True
        return False
    
    def evaluate(self, dataset) -> Tuple[float, float]:

        if self.model_stage2 is None:
            raise ValueError("Must complete stage 2 training before evaluation")
        
        # Convert dataset to (x, y) format
        test_ds = self._separate_features_labels(dataset)
        
        results = self.model_stage2.evaluate(test_ds, verbose=self.verbose)
        test_loss, test_auc = results[0], results[1]
        
        return test_loss, test_auc
