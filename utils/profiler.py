"""
Model profiling utilities for calculating FLOPs and parameters.
"""

import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import numpy as np


def get_flops(model, batch_size=1, field_size=None, fwd_pass_only=True):
    """
    Calculate FLOPs for a TensorFlow/Keras model.
    
    Args:
        model: Keras model
        batch_size: Batch size for profiling
        field_size: Field size for models that require dictionary inputs (e.g., SSRNet)
        fwd_pass_only: If True, only count forward pass FLOPs
        
    Returns:
        Dictionary containing FLOPs and parameter information
    """
    if not isinstance(model, tf.keras.Model):
        raise ValueError("Model must be a Keras Model instance")
    
    # Build model if not built
    if not model.built:
        # Try to determine input format and build model
        if hasattr(model, 'field_size') and hasattr(model, 'feature_size'):
            # Dictionary input model (like SSRNet, AutoInt, etc.)
            field_size = field_size or getattr(model, 'field_size', 39)
            dummy_inputs = {
                'feat_index': tf.keras.Input(shape=(field_size,), dtype=tf.int32),
                'feat_value': tf.keras.Input(shape=(field_size,), dtype=tf.float32)
            }
            model(dummy_inputs, training=False)
        else:
            # Try to infer from input_shape
            try:
                if hasattr(model, 'input_shape'):
                    input_shape = model.input_shape
                    if isinstance(input_shape, tuple):
                        model.build(input_shape)
                    else:
                        raise ValueError("Cannot automatically determine input shape")
            except:
                raise ValueError("Cannot build model - please provide field_size for dictionary input models")
    
    # Create dummy input based on model type
    try:
        # Try dictionary input first (for SSRNet, AutoInt, DCN-v2, etc.)
        if hasattr(model, 'field_size'):
            field_size = field_size or getattr(model, 'field_size', 39)
            dummy_inputs = {
                'feat_index': tf.random.uniform(
                    (batch_size, field_size), 
                    minval=0, 
                    maxval=1000, 
                    dtype=tf.int32
                ),
                'feat_value': tf.random.uniform(
                    (batch_size, field_size), 
                    minval=0.0, 
                    maxval=1.0, 
                    dtype=tf.float32
                )
            }
        else:
            # Simple tensor input (for models like Wukong with direct tensor inputs)
            input_shape = model.input_shape
            if isinstance(input_shape, tuple):
                dummy_shape = (batch_size,) + input_shape[1:]
                dummy_inputs = tf.random.normal(dummy_shape)
            else:
                raise ValueError(f"Unsupported input_shape type: {type(input_shape)}")
        
        # Create concrete function for profiling
        forward_pass = tf.function(model.call)
        concrete_func = forward_pass.get_concrete_function(dummy_inputs, training=False)
        
        # Create profiler options
        options = ProfileOptionBuilder.float_operation()
        options['output'] = 'none'  # Don't print to stdout
        
        # Run profiler
        graph_info = profile(
            concrete_func.graph,
            options=options
        )
        
        # Extract FLOPs
        flops = graph_info.total_float_ops
        
        # Get trainable parameters
        trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables if 'embedding' not in var.name.lower()])
        total_params = sum([tf.size(var).numpy() for var in model.variables])
        
        return {
            'total_flops': flops,
            'flops_per_batch': flops,
            'trainable_params': int(trainable_params),
            'total_params': int(total_params)
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to profile model: {str(e)}")


def print_model_profile(model, batch_size=1, field_size=None, logger=None):
    """
    Print model profiling information including FLOPs and parameters.
    
    Args:
        model: Keras model to profile
        batch_size: Batch size for FLOPs calculation
        field_size: Field size for models that require dictionary inputs
        logger: Optional logger object. If None, prints to stdout
    """
    try:
        profile_info = get_flops(model, batch_size=batch_size, field_size=field_size)
        
        # Format numbers with commas
        flops = profile_info['total_flops']
        trainable_params = profile_info['trainable_params']
        total_params = profile_info['total_params']
        
        # Convert to more readable units
        if flops >= 1e12:
            flops_str = f"{flops / 1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            flops_str = f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            flops_str = f"{flops / 1e6:.2f} MFLOPs"
        else:
            flops_str = f"{flops:,} FLOPs"
        
        # Format parameters
        if total_params >= 1e9:
            params_str = f"{total_params / 1e9:.2f}B"
        else:
            params_str = f"{total_params / 1e6:.2f}M"

        # Format_trainable_params
        if trainable_params >= 1e9:
            trainable_params_str = f"{trainable_params / 1e9:.2f}B"
        else:
            trainable_params_str = f"{trainable_params / 1e6:.2f}M"
        
        msg = f"\n{'='*60}\n"
        msg += "Model Profile Summary\n"
        msg += f"{'='*60}\n"
        msg += f"Total FLOPs (per sample): {flops_str} ({flops:,})\n"
        msg += f"Trainable Parameters: {trainable_params:,} ({trainable_params_str})\n"
        msg += f"Total Parameters: {total_params:,} ({params_str})\n "
        msg += f"Batch Size: {batch_size}\n"
        msg += f"{'='*60}\n"
        
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
        return profile_info
        
    except Exception as e:
        error_msg = f"Error profiling model: {str(e)}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return None


def format_flops(flops):
    """Format FLOPs number to human-readable string."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    else:
        return f"{flops} FLOPs"