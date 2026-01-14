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



class StructuralSparseBlockT21(layers.Layer):
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
                 stage_name: str = "structural_sparse_block_t21",
                 **kwargs):
        """
        Initialize StructuralSparseBlockT21.
        
        Args:
            len_embeddings: Length of input embeddings
            tokennum: Number of tokens (similar to b_matrices)
            hidden_unit: Hidden layer dimension
            top_k: Top-k parameter for sparse operator
            stage_name: Name of this block
        """
        super(StructuralSparseBlockT21, self).__init__(name=stage_name, **kwargs)
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
        
        print("=====================Using Block T21=====================")
    
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

            # Create separate alpha for each iteration as a single trainable tensor
            # Shape: [iterations, 1] - each iteration has its own learnable alpha
            alpha_tensor = self.add_weight(
                f"alpha_{token_id}",
                shape=[self.iterations, 1],
                initializer=tf.constant_initializer(self.alpha_init),
                # initializer=tf.keras.initializers.TruncatedNormal(
                #     mean=self.alpha_init,
                #     stddev=0.1
                # ),
                dtype=tf.float32,
                trainable=True
            )
            self.alphas.append(alpha_tensor)
            self.scales.append(self.add_weight(
                f"scale_{token_id}",
                shape=[1, self.hidden_unit],
                initializer=tf.constant_initializer(self.scale_init),
                dtype=tf.float32,
            ))

        
        super(StructuralSparseBlockT21, self).build(input_shape)

    def iterative_competitive_inhibition(self,
            inputs,
            token_id: int,
            iterations: int,
            epsilon=1e-6,
            name="ici_sparsity",
        ):
        x_t = tf.nn.relu(inputs)
        alphas_tensor = self.alphas[token_id]  # Shape: [iterations, 1]

        def compute_inhibition(x):
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            return mean

        for i in range(iterations):
            mean = compute_inhibition(x_t)
            current_alpha = tf.gather(alphas_tensor, i)  # Shape: [1]
            current_alpha = tf.reshape(current_alpha, [1, 1])
            x_t = tf.nn.relu(x_t - current_alpha * mean)

        x_final = self.scales[token_id] * x_t
        return x_final

    def call(self, inputs, training=None, return_sparse_g=False):
        """
        Forward pass of StructuralSparseBlockT.
        
        Args:
            inputs: Input tensor
            training: Training mode flag
            return_sparse_g: Whether to return sparse_g tensors for sparsity monitoring
            
        Returns:
            If return_sparse_g is False:
                Output tensor [B, tokennum, top_k]
            If return_sparse_g is True:
                (Output tensor [B, tokennum, top_k], List of sparse_g tensors)
        """
        # Flatten inputs if needed
        inputs_flat = tf.reshape(inputs, [-1, self.len_embeddings])
        
        out_tensor = []
        sparse_g_list = []
        
        for token_id in range(self.tokennum):
            # Sparse matrix multiplication with statistical top-k
            with tf.name_scope(f"token_{token_id}"):
                hidden_x = tf.matmul(inputs_flat, self.input_weights[token_id])  # [B, hidden_unit]
                # Apply statistical top-k sparsification
                sparse_g = self.iterative_competitive_inhibition(hidden_x, token_id, self.iterations)
                
                if return_sparse_g:
                    sparse_g_list.append(sparse_g)
                
                if self.use_gate:
                    value_x = tf.matmul(inputs_flat, self.value_weights[token_id])
                    sparse_x = sparse_g * value_x
                else:
                    sparse_x = sparse_g

                if self.use_block_dense:
                    sparse_x = sparse_x + self.dense_layers[token_id](sparse_x)

                # Layer normalization
                sparse_x = self.layer_norms[token_id](sparse_x, training=training)
                
                output = sparse_x
                out_tensor.append(output)
        
        # Stack outputs: [B, tokennum, hidden_unit]
        outputs = tf.stack(out_tensor, axis=1)
        
        if return_sparse_g:
            return outputs, sparse_g_list
        return outputs

