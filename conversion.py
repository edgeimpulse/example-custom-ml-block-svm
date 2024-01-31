import io, os, shutil
import tensorflow as tf
import numpy as np
import jax.numpy as jnp
from jax.experimental import jax2tf
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.lite.python import schema_py_generated as schema_fb

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
        #tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.target_spec.supported_types = [
        tf.dtypes.float32,
        tf.dtypes.int16,
        tf.dtypes.int8
    ]
    model = converter.convert()
    model_obj = flatbuffer_utils.convert_bytearray_to_object(model)
    flatbuffer_utils.write_model(model_obj, out_file)

    model = flatbuffer_utils.read_model(out_file)
    for o in model.operatorCodes:
        print(str(vars(o)))
        # if o.builtinCode == 123:
        #     o.builtinCode = 64
        #     o.deprecatedBuiltinCode = 64

    for o in model.subgraphs[0].operators:
        print(str(vars(o)))
        # if o.opcodeIndex == 4:
        #     print(str(vars(o.builtinOptions)))
        #     o.builtinOptions = schema_fb.SelectOptionsT()
        #     o.builtinOptionsType = 47

    print(str(vars(model.operatorCodes[0])))
    print(str(dir(model.operatorCodes)))

    model_data = flatbuffer_utils.convert_object_to_bytearray(model)
    model_file_out = open(out_file, 'wb')
    model_file_out.write(model_data)
    model_file_out.close()


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