import tensorflow as tf
from tensorflow import keras
from tensorflow.python.trackable.data_structures import ListWrapper
import numpy as np

class SSRNetMonitoringCallback(keras.callbacks.Callback):

    def __init__(self, log_dir: str, log_freq: str = 'epoch', log_interval: int = 100):
        """
        Initialize the monitoring callback.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            log_freq: Logging frequency, 'epoch' or 'batch'
            log_interval: Interval for logging when log_freq='batch' (e.g., 100 means log every 100 batches)
        """
        super(SSRNetMonitoringCallback, self).__init__()
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.log_interval = log_interval
        self.file_writer = None
        self.batch_count = 0
        self.global_step = 0  
        
    def on_train_begin(self, logs=None):
        """Initialize file writer when training begins."""
        if self.file_writer is not None:
            self.file_writer.flush()
            self.file_writer.close()
        
        self.file_writer = tf.summary.create_file_writer(self.log_dir)
        
    def on_train_end(self, logs=None):
        """Flush file writer when training ends (but don't close for multi-part training)."""
        if self.file_writer is not None:
            self.file_writer.flush()
    
    def on_epoch_end(self, epoch, logs=None):
        """Log parameters at the end of each epoch."""
        if self.log_freq == 'epoch':
            self.global_step += 1  
            self._log_parameters(self.global_step)
    
    def on_batch_end(self, batch, logs=None):
        """Log parameters at the end of each batch (if configured)."""
        if self.log_freq == 'batch':
            self.global_step += 1 
            # Only log at specified intervals
            if self.global_step % self.log_interval == 0:
                self._log_parameters(self.global_step) 
    
    def _log_parameters(self, step):
        """
        Extract and log alphas and scales from SSR blocks.
        
        Args:
            step: Current training step (epoch or batch number)
        """
        if self.file_writer is None:
            print(f"[Monitoring] Warning: file_writer is None at step {step}")
            return
        
        try:
            with self.file_writer.as_default():
            # Iterate through all layers in the model
                ssr_blocks_found = 0
                for layer in self.model.layers:
                    # Check if this is an SSR block layer
                    if 'ssr_block' in layer.name.lower() or 'structural_sparse' in layer.name.lower():
                        ssr_blocks_found += 1
                        # Extract alphas and scales if they exist
                        if hasattr(layer, 'alphas') and hasattr(layer, 'scales'):
                            tokennum = len(layer.alphas)
                            
                            for token_id in range(tokennum):
                                # Log alpha values
                                if token_id < len(layer.alphas):
                                    alpha_param = layer.alphas[token_id]
                                
                                    if isinstance(alpha_param, (list, ListWrapper)):
                                        alpha_values = [head_alpha.numpy() for head_alpha in alpha_param]
                                        alpha_value = np.mean(alpha_values)
                                    else:
                                        
                                        alpha_value = alpha_param.numpy()
                                        
                                        if alpha_value.ndim > 0 and alpha_value.size > 1:
                                            
                                            for iter_id in range(alpha_value.shape[0]):
                                                tf.summary.scalar(
                                                    f'{layer.name}/alphas/token_{token_id}_iter_{iter_id}',
                                                    alpha_value[iter_id, 0],
                                                    step=step
                                                )
                                            
                                            tf.summary.scalar(
                                                f'{layer.name}/alphas/token_{token_id}_mean',
                                                np.mean(alpha_value),
                                                step=step
                                            )
                                            continue  
                                        else:
                                            alpha_value = alpha_value[0] if alpha_value.size > 0 else 0.0
                                    
                                    tf.summary.scalar(
                                        f'{layer.name}/alphas/token_{token_id}',
                                        alpha_value,
                                        step=step
                                    )
                                
                                # Log scale statistics
                                if token_id < len(layer.scales):
                                    scale_param = layer.scales[token_id]
                                    
                                    if isinstance(scale_param, (list, ListWrapper)):
                                        
                                        scale_values = np.concatenate([head_scale.numpy().flatten() for head_scale in scale_param])
                                    else:
                                        
                                        scale_values = scale_param.numpy()
                                    
                                    # Log histogram of scales
                                    tf.summary.histogram(
                                        f'{layer.name}/scales/token_{token_id}',
                                        scale_values,
                                        step=step
                                    )
                                    
                                    # Log scale mean
                                    tf.summary.scalar(
                                        f'{layer.name}/scales_mean/token_{token_id}',
                                        np.mean(scale_values),
                                        step=step
                                    )
                                    
                                    # Log scale std
                                    tf.summary.scalar(
                                        f'{layer.name}/scales_std/token_{token_id}',
                                        np.std(scale_values),
                                        step=step
                                    )
                                    
                                    # Log scale min/max
                                    tf.summary.scalar(
                                        f'{layer.name}/scales_min/token_{token_id}',
                                        np.min(scale_values),
                                        step=step
                                    )
                                    
                                    tf.summary.scalar(
                                        f'{layer.name}/scales_max/token_{token_id}',
                                        np.max(scale_values),
                                        step=step
                                    )
            
            # Flush the writer
            self.file_writer.flush()
            
            # if ssr_blocks_found == 0:
            #     print(f"[Monitoring] Warning: No SSR blocks found in model at step {step}")
            # else:
            #     print(f"[Monitoring] Logged parameters for {ssr_blocks_found} SSR blocks at step {step}")
                
        except Exception as e:
            print(f"[Monitoring] Error logging parameters at step {step}: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        """Close the file writer. Call this after all training is complete."""
        if self.file_writer is not None:
            self.file_writer.flush()
            self.file_writer.close()
            self.file_writer = None