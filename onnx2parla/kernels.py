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

    """
        Function:
            Y = X CONV W (Using padding, stride and dilaton attribute
    """
    x_io = node.inputs["X"]
    w_io = node.inputs["W"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    w = w_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    # Assuming same stride in all directions
    stride = node.get_attr("strides", [1])[0]
    # Assuming same padding in all directions
    padding = node.get_attr("pads", [0])[0]
    dilations = node.get_attr("dilations", [1])[
        0
    ]  # Assuming same padding in all directions

    def fn():
        xt = x.transpose(0, 2, 3, 1)
        wt = w.transpose(2, 3, 1, 0)
        parray.copy(
            y,
            (
                utils.conv2D(
                    xt, wt, stride=stride, pad=padding, dilation=dilations
                ).transpose(0, 3, 1, 2)
            ),
        )

    return fn


def relu_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    """
        Function:
                Y = RELU(X)
            max (x, 0)
    """

    x_io = node.inputs["X"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    def fn():
        parray.copy(y, np.maximum(x, 0))

    return fn


def maxpool_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    """
        Function:
                Y = MAXPOOL(X) (Using padding, stride and pool kernel size)
            --> Propagate maximum value in the kernel window
    """

    x_io = node.inputs["X"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    # Assume same stride in all directions
    stride = (node.get_attr("strides", [1]))[0]
    # Assume same padding in all directions
    padding = (node.get_attr("pads", [0]))[0]
    kernel_shape = node.get_attr("kernel_shape")

    def fn():
        xt = x.transpose(0, 2, 3, 1)
        n_ex, in_rows, in_cols, nc_in = xt.shape
        (fr, fc), s, p = kernel_shape, stride, padding
        x_pad, (pr1, pr2, pc1, pc2) = utils.pad2D(xt, p, kernel_shape, s)

        out_rows = np.floor(1 + (in_rows + pr1 + pr2 - fr) / s).astype(int)
        out_cols = np.floor(1 + (in_cols + pc1 + pc2 - fc) / s).astype(int)
        Y = np.zeros((n_ex, out_rows, out_cols, nc_in))
        for m in range(n_ex):
            for i in range(out_rows):
                for j in range(out_cols):
                    for c in range(nc_in):
                        # calculate window boundaries, incorporating stride
                        i0, i1 = i * s, (i * s) + fr
                        j0, j1 = j * s, (j * s) + fc

                        xi = x_pad[m, i0:i1, j0:j1, c]
                        Y[m, i, j, c] = np.amax(xi)
        Y = Y.transpose(0, 3, 1, 2)
        parray.copy(y, Y)

    return fn


def batchnorm_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    """
        Function:
        Y = gamma * x_hat + beta
        where:
            x_hat = (x - r_mean)/sqrt(r_variance + epsilon)
        & r_mean and r_variance are running mean & variance

            r_mean = momentum * training_mean
                     + (1 - momentum) * calculated mean
            r_variance = momentum * training_variance
                         + (1 - momentum) * calculated variance
    """

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

    epsilon = node.get_attr("epsilon", 1e-05)
    momentum = node.get_attr("momentum", 0.9)
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

    """
        Function:
                Y = GLOBAL_AVERAGE_POOL(X)
            --> CONVE NCHW to NC11 (Average on HW dimensions)
    """

    x_io = node.inputs["X"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    def fn():
        out = np.empty([x.shape[0], x.shape[1], 1, 1])
        for n in np.arange(0, x.shape[0]):
            for c in np.arange(0, x.shape[1]):
                out[n, c, 0, 0] = np.average(x[n, c, :, :])

        parray.copy(y, out)

    return fn


def flatten_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    """
        Function:
                Y = FLATTEN(X)
            --> Convert 4D 'X' to 2D 'Y'
    """

    x_io = node.inputs["input"]
    y_io = node.outputs["output"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    def fn():
        parray.copy(y, x.reshape(x.shape[0], -1))

    return fn


def gemm_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    """
        Function:
                Y = alpha*(X @ W) + beta*b
    """

    x_io = node.inputs["A"]
    w_io = node.inputs["B"]
    b_io = node.inputs["C"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    w = w_io.get_data(alloc_map)
    b = b_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    alpha = node.get_attr("alpha", 1.0)
    beta = node.get_attr("beta", 1.0)
    transX = node.get_attr("transA", 0)
    transW = node.get_attr("transB", 0)

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
