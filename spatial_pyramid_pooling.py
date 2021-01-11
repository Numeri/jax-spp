import functools

from jax.experimental import stax
from jax.experimental.stax import FanInConcat, FanOut, MaxPool

def FlattenImage():
    def init_fun(rng, input_shape):
        output_shape = input_shape[0], input_shape[1]*input_shape[2], input_shape[3]
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        input_shape = inputs.shape
        return inputs.reshape((input_shape[0], input_shape[1]*input_shape[2], input_shape[3]))

    return init_fun, apply_fun

def pooling_layer_params(layer_window_dim, height, width):
    vertical_stride = height // layer_window_dim
    vertical_window = height - vertical_stride * (layer_window_dim - 1)
    horizontal_stride = width // layer_window_dim
    horizontal_window = width - horizontal_stride * (layer_window_dim - 1)

    if layer_window_dim > height or layer_window_dim > width:
        raise Exception(f'SpatialPooling passed input_shape {(height, width)} smaller than layer window dimension {layer_window_dim}')

    return {
            'window_shape': (vertical_window, horizontal_window),
            'strides': (vertical_stride, horizontal_stride)
           }

def SpatialPooling(layer_window_dims=(16, 8, 4, 2, 1), pooling_type=MaxPool):
    params_ = [(), ([(), ()],) * len(layer_window_dims), ()]

    def init_fun(rng, input_shape):
        fixed_length = functools.reduce(lambda x, y : y**2 + x, layer_window_dims, 0)
        output_shape = input_shape[0], fixed_length, input_shape[-1]
        return output_shape, ()
    
    def apply_fun(params, inputs, **kwargs):
        input_shape = inputs.shape
        height = input_shape[1]
        width  = input_shape[2]
        pools = [
                 stax.serial(
                    pooling_type(**pooling_layer_params(layer_window_dim, height, width)),
                    FlattenImage()
                 ) for layer_window_dim in layer_window_dims
                ]

        _, apply_fun_ = stax.serial(
            FanOut(len(layer_window_dims)),
            stax.parallel(*pools),
            FanInConcat(axis=1)
        )

        return apply_fun_(params_, inputs, **kwargs)
    return init_fun, apply_fun

