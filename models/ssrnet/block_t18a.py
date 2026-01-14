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



class StructuralSparseBlockT18a(layers.Layer):
    """
    Structural Sparse Block with Top-k layer for SSRNet-T.
    Implements structured sparsity with Statistical Top-k operator and learnable weights.
    """
    def __init__(self, 
                 len_embeddings: int,
                 tokennum: int,
                 hidden_unit: int,
                 top_k: int,
                 out_unit: int,
                 alpha_init: float = 0.0,
                 scale_init: float = 1.0,
                 use_gate: bool = True,
                 use_block_dense: bool = False,
                 iterations: int = 5,
                 dropout_rates: List[float] = [0.0, 0.0, 0.1],
                 l2_reg: float = 0.0,
                 stage_name: str = "structural_sparse_block_t18a",
                 **kwargs):
        """
        Initialize StructuralSparseBlockT18a.
        
        Args:
            len_embeddings: Length of input embeddings
            tokennum: Number of tokens (similar to b_matrices)
            hidden_unit: Hidden layer dimension
            top_k: Top-k parameter for sparse operator
            stage_name: Name of this block
        """
        super(StructuralSparseBlockT18a, self).__init__(name=stage_name, **kwargs)
        self.len_embeddings = len_embeddings
        self.tokennum = tokennum
        self.hidden_unit = hidden_unit
        self.top_k = top_k
        self.out_unit = out_unit
        self.layer_id = int(self.name.split("_")[-1]) if "_" in self.name else 1,
        self.alpha_init=alpha_init,
        self.scale_init=scale_init
        self.use_gate = use_gate 
        self.iterations = iterations
        self.use_block_dense = use_block_dense
        self.dropout_rates = dropout_rates
        self.l2_reg = l2_reg
        
        print("=====================Using Block T18a=====================")
    
    def build(self, input_shape):
        """Build learnable weights and layers."""
        # Create weights for each token
        self.gate_weights = []
        self.value_weights = []
        self.batch_norms = []
        self.dense_layers = []
        self.layer_norms = []
        self.input_weights = []
        self.value_weights = []
        self.alphas = []
        self.scales = []
        for token_id in range(self.tokennum):
            input_weight = self.add_weight(
                name=f"input_weight_{token_id}",
                shape=(self.len_embeddings, self.hidden_unit),
                initializer=tf.keras.initializers.GlorotNormal(),
                trainable=True
            )
            self.input_weights.append(input_weight)
            if self.use_gate:
                value_weight = self.add_weight(
                    name=f"value_weight_{token_id}",
                    shape=(self.len_embeddings, self.hidden_unit),
                    initializer=tf.keras.initializers.GlorotNormal(),
                    trainable=True
                )
                self.value_weights.append(value_weight)
            
            # Dense layer (MLP mixer)
            if self.use_block_dense:
                dense_layer = keras.Sequential([
                    layers.Dense(
                        self.out_unit,
                        activation="gelu",
                        kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
                        name=f"dense_{token_id}"
                    ),
                    layers.Dropout(self.dropout_rates[2], name=f"dropout_{token_id}")
                ], name=f"dense_block_{token_id}")
                self.dense_layers.append(dense_layer)
            
            # Layer normalization
            layer_norm = layers.LayerNormalization(
                epsilon=1e-6,
                name=f"layer_norm_{token_id}"
            )
            self.layer_norms.append(layer_norm)

            self.alphas.append(self.add_weight(
                f"alpha_{token_id}",
                shape=[1],
                initializer=tf.constant_initializer(self.alpha_init),
                dtype=tf.float32,
            ))
            self.scales.append(self.add_weight(
                f"scale_{token_id}",
                shape=[1, self.len_embeddings],
                initializer=tf.constant_initializer(self.scale_init),
                dtype=tf.float32,
            ))

        
        super(StructuralSparseBlockT18a, self).build(input_shape)

    def iterative_competitive_inhibition(self,
            inputs,
            token_id: int,
            iterations: int,
            epsilon=1e-6,
            name="ici_sparsity",
    ):
    
        x0 = tf.nn.relu(inputs)

        def compute_inhibition(x):
            # mean over last dim (feature dim)
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            return mean

        def cond(i, x_t):
            return i < iterations

        def body(i, x_t):
            mean = compute_inhibition(x_t)
            x_next = tf.nn.relu(x_t - self.alphas[token_id] * mean)
            return i + 1, x_next

        _, x_res = tf.while_loop(
            cond,
            body,
            [tf.constant(0), x0],
            back_prop=True,
        )

        x_final = self.scales[token_id] * x_res
        return x_final

    def call(self, inputs, training=None):
        """
        Forward pass of StructuralSparseBlockT.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            
        Returns:
            Output tensor [B, tokennum, top_k]
        """
        # Flatten inputs if needed
        inputs_flat = tf.reshape(inputs, [-1, self.len_embeddings])
        
        out_tensor = []
        
        for token_id in range(self.tokennum):
            # Sparse matrix multiplication with statistical top-k
            with tf.name_scope(f"token_{token_id}"):
                hidden_x = tf.matmul(inputs_flat, self.input_weights[token_id])  # [B, hidden_unit]
                sparse_g, _ = tf.math.top_k(hidden_x, self.top_k)
                if self.use_block_dense:
                    sparse_x = self.dense_layers[token_id](sparse_g)

                # Layer normalization
                sparse_x = self.layer_norms[token_id](sparse_x, training=training)
                
                output = sparse_x
                out_tensor.append(output)
        
        # Stack outputs: [B, tokennum, hidden_unit]
        outputs = tf.stack(out_tensor, axis=1)
        
        return outputs


