import numpy as np
import numpy
import parla.array as parray
import logging

import cupy

import im2col

from node import Node
from config import Config

from typing import Callable, Dict

def load_cpu(node: Node, alloc_map: Dict[str, np.ndarray], config):
    z_io = node.outputs["Z"]
    z = z_io.get_data(alloc_map)

    width = node.get_attr("width")
    batch_size = config.batch_size

    def fn():
        batch_id = node.get_attr("batch_id")
        start_idx = batch_id * batch_size + node.instance_id * width
        end_idx = start_idx + width
        parray.copy(z, config.user_load_fn(start_idx, end_idx))
        batch_id += 1
        node.set_attr("batch_id", batch_id)

    return fn


def store_cpu(node, alloc_map, config):
    x_io = node.inputs["X"]
    x = x_io.get_data(alloc_map)

    def fn():
        config.user_store_fn(x)

    return fn


def copy(node: Node, alloc_map, config: Config):
    x_io = node.inputs["X"]
    z_io = node.outputs["Z"]

    x = x_io.get_data(alloc_map)
    z = z_io.get_data(alloc_map)

    tz = type(z)

    def fn():
        if tz == numpy.ndarray:  # to cpu
            np.copyto(z, cupy.asnumpy(x))
        if tz == cupy.core.core.ndarray:  # to gpu
            with cupy.cuda.Device(node.device_id):
                cupy.copyto(z, cupy.asarray(x))

        logging.log(logging.INFO, f"done copy {z}, {tz}")

    return fn


def add_cpu(
    node: Node, alloc_map: Dict[str, np.ndarray], config: Config
) -> Callable[[], None]:

    """Add Kernel (CPU version)

    This function creates a kernel which adds two vectors on CPU

    Z = X + Y

    Args:
        node (node): A source node with operator `add`
        alloc_map (dict): The dictionary of names->allocations

    Returns:
        fn: A new kernel Z = X + Y
    """

    if node.get_operator() != "add":
        raise ValueError(
            "Node operator should be add, not {}".format(node.get_operator())
        )

    x_io = node.inputs["A"]
    y_io = node.inputs["B"]
    z_io = node.outputs["C"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)
    z = z_io.get_data(alloc_map)

    def fn():
        parray.copy(z, x + y)

    return fn


def add_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    """Add Kernel (GPU version)

    This function creates a kernel which adds two vectors on CPU

    Z = X + Y

    Args:
        node (node): A source node with operator `add`
        alloc_map (dict): The dictionary of names->allocations

    Returns:
        fn: A new kernel Z = X + Y
    """

    if node.get_operator() != "add":
        raise ValueError(
            "Node operator should be add, not {}".format(node.get_operator())
        )

    x_io = node.inputs["X"]
    y_io = node.inputs["Y"]
    z_io = node.outputs["Z"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)
    z = z_io.get_data(alloc_map)

    def fn():
        cupy.copyto(z, x + y)
        logging.log(logging.INFO, f"{z} = {x} + {y}")

    return fn

def conv_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    x_io = node.inputs["X"]
    w_io = node.inputs["W"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    w = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    stride = (node.get_attr("strides"))[0]	## Assuming same stride in all directions
    padding = (node.get_attr("pads"))[0]	## Assuming same padding in all directions

    def fn():
    	n_filters, c_filter, h_filter, w_filter = w.shape
    	n_x, c_x, h_x, w_x = x.shape
    	h_out = (h_x - h_filter + 2 * padding) / stride + 1
    	w_out = (w_x - w_filter + 2 * padding) / stride + 1

    	if not h_out.is_integer() or not w_out.is_integer():
        	raise Exception('Invalid output dimension!')

    	h_out, w_out = int(h_out), int(w_out)

    	x_col = im2col_indices(x, h_filter, w_filter, padding=padding, stride=stride)
    	w_col = w.reshape(n_filters, -1)

    	out = w_col @ x_col
    	out = out.reshape(n_filters, h_out, w_out, n_x)
    	out = out.transpose(3, 0, 1, 2)

    	parray.copy(y, out)

    	return fn

def relu_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    x_io = node.inputs["X"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    def fn():
    	parray.copy(y, np.maximum(x, 0))
    	return fn


def maxpool_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    def fn():
    	return fn

def batchnorm_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    def fn():
        return fn

def reshape_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    def fn():
        return fn
