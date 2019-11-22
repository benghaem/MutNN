import numpy as np
import numpy
import parla.array as parray
import logging

import cupy

from third_party import utils as utils

from node import Node
from config import Config

from typing import Callable, Dict


def load_cpu(node: Node, alloc_map, config):
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

    if node.get_operator() != "Add":
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
    w = w_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    stride = (node.get_attr("strides"))[0]  # Assuming same stride in all directions
    padding = (node.get_attr("pads"))[0]  # Assuming same padding in all directions

    def fn():
        n_filters, c_filter, h_filter, w_filter = w.shape
        n_x, c_x, h_x, w_x = x.shape
        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception("Invalid output dimension!")

        h_out, w_out = int(h_out), int(w_out)

        x_col = im2col.im2col_indices(
            x, h_filter, w_filter, padding=padding, stride=stride
        )
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

    x_io = node.inputs["X"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    stride = (node.get_attr("strides"))[0]  # Assuming same stride in all directions
    padding = (node.get_attr("pads"))[0]  # Assuming same padding in all directions
    kernel_shape = node.get_attr("kernel_shape")

    def fn():
        n_x, c_x, h_x, w_x = x.shape
        h_out = (h_x - kernel_shape[0] + 2 * padding) / stride + 1
        w_out = (w_x - kernel_shape[1] + 2 * padding) / stride + 1

        if not h_out.is_integer() or not w_out.is_integer():
            raise Exception("Invalid output dimension!")

        h_out, w_out = int(h_out), int(w_out)

        x_reshaped = x.reshape(n_x * c_x, 1, h_x, w_x)
        x_col = im2col.im2col_indices(
            x_reshaped, kernel_shape[0], kernel_shape[1], padding=padding, stride=stride
        )
        max_idx = np.argmax(x_col, axis=0)
        out = x_col[max_idx, range(max_idx.size)]
        out = out.reshape(h_out, w_out, n_x, c_x)
        out = out.transpose(2, 3, 0, 1)

        parray.copy(y, out)

    return fn


def batchnorm_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    x_io = node.inputs["X"]
    gamma_io = node.inputs["scale"]
    beta_io = node.inputs["B"]
    mean_io = node.inputs["mean"]
    var_io = node.inputs["var"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    gamma = gamma_io.get_data(alloc_map)
    beta = beta_io.get_data(alloc_map)
    mean = mean_io.get_data(alloc_map)
    var = var_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    epsilon = node.get_attr("epsilon")
    momentum = node.get_attr("momentum")
    spatial = node.get_attr("spatial")

    def fn():
        N, C, H, W = x.shape
        # mini-batch mean
        mean_batch = np.mean(x, axis=(0, 2, 3))
        mean_moving = (mean_batch * (1 - momentum)) + (mean * momentum)
        # mini-batch variance
        variance_batch = np.mean(
            (x - mean_batch.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3)
        )
        variance_moving = (variance_batch * (1 - momentum)) + (var * momentum)
        # normalize
        x_hat = (
            (x - mean_moving.reshape((1, C, 1, 1)))
            * 1.0
            / np.sqrt(variance_moving.reshape((1, C, 1, 1)) + epsilon)
        )
        # scale and shift
        out = gamma.reshape((1, C, 1, 1)) * x_hat + beta.reshape((1, C, 1, 1))
        parray.copy(y, out)

    return fn


def globalAveragePool_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    x_io = node.inputs["X"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    def fn():
        out = np.empty(x.shape[0], x.shape[1], 1, 1)
        for n in np.arange(0, x.shape[0]):
            for c in np.arange(0, x.shape[1]):
                out[n, c, 0, 0] = np.average(x[n, c, :, :])

        parray.copy(y, out)

    return fn


def flatten_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    x_io = node.inputs["input"]
    y_io = node.outputs["output"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    def fn():
        parray.copy(y, x.reshape(x.shape[0], -1))

    return fn


def gemm_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    x_io = node.inputs["A"]
    w_io = node.inputs["B"]
    b_io = node.inputs["C"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    w = w_io.get_data(alloc_map)
    b = b_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    alpha = node.get_attr("alpha")
    beta = node.get_attr("beta")
    transX = node.get_attr("transA")
    transW = node.get_attr("transB")

    def fn():
        if transX == 1:
            xt = np.transpose(x)
        else:
            xt = x
        if transW == 1:
            wt = np.transpose(w)
        else:
            wt = w

        parray.copy(y, alpha * (xt @ wt) + beta * b)

    return fn
