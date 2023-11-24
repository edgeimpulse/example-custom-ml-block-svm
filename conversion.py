import io, os, shutil
import tensorflow as tf
import numpy as np
import jax.numpy as jnp
from jax.experimental import jax2tf
from tensorflow.lite.tools import flatbuffer_utils

def convert_jax(input_shape, jax_fn, out_file):
    full_shape = tuple([ 1 ]) + tuple(input_shape)
    tf_predict = tf.function(
        jax2tf.convert(jax_fn, enable_xla=False),
        input_signature=[
            tf.TensorSpec(shape=full_shape, dtype=tf.float32, name='input')
        ],
        autograph=False)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_predict.get_concrete_function()], tf_predict)
    converter.allow_custom_ops = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    model = converter.convert()
    model_obj = flatbuffer_utils.convert_bytearray_to_object(model)
    flatbuffer_utils.write_model(model_obj, out_file)


# def convert_jax(input_shape, jax_fn, out_file):
#     full_shape = tuple([ 1 ]) + tuple(input_shape)
#     reference_input = jnp.zeros(full_shape)
#
#     converter = tf.lite.TFLiteConverter.experimental_from_jax(
#         [jax_fn], [[('input1', reference_input)]])
#
#     tflite_model = converter.convert()
#     with open(out_file, 'wb') as f:
#         f.write(tflite_model)
#