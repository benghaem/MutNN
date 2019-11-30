import numpy as np
import numpy
import parla.array as parray
import logging
import datetime

import time
import cupy


import chainer
from chainer import functions as gputils

from onnx2parla.third_party import utils as utils
from onnx2parla.node import Node
from onnx2parla.config import Config

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
        time_st = datetime.datetime.now()

        if tz == numpy.ndarray:  # to cpu
            parray.copy(z,chainer.backends.cuda.to_cpu(x, stream=None))

        if tz == cupy.core.core.ndarray:  # to gpu
            with cupy.cuda.Device(node.device_id):
                tmp = cupy.asarray(x, dtype=cupy.float32)
                cupy.copyto(z,tmp)
                #cupy.copyto(z,chainer.backends.cuda.to_gpu(x,
                #    device=node.device_id, stream=None))


        # to gpu:

        #og_shape = x.shape


        #if tz == numpy.ndarray:  # to cpu
        #    with cupy.cuda.Device(device=node.device_id):
        #        arr_flat = x.reshape((-1))
        #        z_flat = np.ndarray(arr_flat.shape)

        #        for i, v in enumerate(arr_flat):
        #            z_flat[i] = v

        #        z_flat = z_flat.reshape(og_shape)

        #        parray.copy(z,z_flat)


        #if tz == cupy.core.core.ndarray:

        #    arr_flat = x.reshape((-1))

        #    with cupy.cuda.Device(device=node.device_id):
        #        z_flat = cupy.ndarray(arr_flat.shape)

        #        for i, v in enumerate(arr_flat):
        #            z_flat[i] = v

        #        z_flat = z_flat.reshape(og_shape)

        #        cupy.copyto(z,z_flat)

        time_end = datetime.datetime.now()
        #logging.log(logging.INFO, f"done copy {z}, {tz}")
        logging.log(logging.INFO, f"TIMER: <{node.operator},{node.node_id} {time_st} -> {time_end}")

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

    This function creates a kernel which adds two vectors on GPU

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

    return fn


def conv_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    """
        Function:
            Y = X CONV W (Using padding, stride and dilaton attribute
    """
    x_io = node.inputs["X"]
    w_io = node.inputs["W"]
    b_io = node.get_input("B")
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    w = w_io.get_data(alloc_map)
    b = b_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    # Assuming same stride in all directions
    stride = node.get_attr("strides", [1])[0]
    # Assuming same padding in all directions
    padding = node.get_attr("pads", [0])[0]
    dilations = node.get_attr("dilations", [1])[
        0
    ]  # Assuming same padding in all directions

    def fn():
        parray.copy(
            y,
            (
                chainer.functions.convolution_2d(
                    x, w, b=b, stride=stride, pad=padding, dilate=dilations,
                )
            ).array,
        )

    return fn


def conv_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    """
        GPU Function:
                Y = X CONV W (Using padding, stride and dilaton attribute
    """
    x_io = node.inputs["X"]
    w_io = node.inputs["W"]
    b_io = node.get_input("B")
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    w = w_io.get_data(alloc_map)
    b = b_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    stride = node.get_attr("strides")[0]  # Assuming same stride in all directions
    padding = node.get_attr("pads")[0]  # Assuming same padding in all directions
    dilations = node.get_attr("dilations")[0]  # Assuming same padding in all directions

    def fn():
        time_st = datetime.datetime.now()
        #logging.log(logging.INFO, f"CONVOP got -->  {x} CONVOP")
        with cupy.cuda.Device(node.device_id):
            cupy.copyto(
                y,
                (
                    chainer.functions.convolution_2d(
                        x, w, b, stride=stride, pad=padding, dilate=dilations,
                    )
                ).array,
            )

            cupy.cuda.Device(node.device_id).synchronize()
        time_end = datetime.datetime.now()
        logging.log(logging.INFO, f"TIMER: <{node.operator},{node.node_id}> {time_st} -> {time_end}")
        #logging.log(logging.INFO, f"CONVOP -->  {y} CONVOP")

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


def relu_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

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
        cupy.copyto(y, cupy.maximum(x, 0))

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
        time_st = datetime.datetime.now()
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
        time_end = datetime.datetime.now()
        #logging.log(logging.INFO, f"MAXPOOL sent -->  {y} MAXPOOL")
        logging.log(logging.INFO, f"TIMER: <{node.operator},{node.node_id}> {time_st} -> {time_end}")

    return fn


def maxpool_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    """
        Function:
                Y = MAXPOOL(X) (Using padding, stride and pool kernel size)
                --> Propagate maximum value in the kernel window
    """

    x_io = node.inputs["X"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    stride = (node.get_attr("strides"))[0]  # Assuming same stride in all directions
    padding = (node.get_attr("pads"))[0]  # Assuming same padding in all directions
    kernel_shape = node.get_attr("kernel_shape")

    def fn():
        cupy.copyto(
            y,
            (
                chainer.functions.max_pooling_2d(
                    x, kernel_shape, stride=stride, pad=0, return_indices=False
                )
            ).array,
        )

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

        #logging.log(logging.INFO, f"BATCHNORM got -->  {x} BATCHNORM")
        parray.copy(
            y,
            chainer.functions.fixed_batch_normalization(
                x,
                gamma,
                beta,
                mean,
                var,
                eps=epsilon
            ).array,
        )

    return fn


def batchnorm_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    """
        Function:
                Y = gamma * x_hat + beta
                        where:
                                x_hat = (x - r_mean)/sqrt(r_variance + epsilon)
                        & r_mean and r_variance are running mean & variance

                                r_mean = momentum * training_mean + (1 - momentum) * calculated mean
                                r_variance = momentum * training_variance + (1 - momentum) * calculated variance
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

    epsilon = node.get_attr("epsilon")
    momentum = node.get_attr("momentum")
    spatial = node.get_attr("spatial")

    def fn():

        cupy.copyto(
            y,
            chainer.functions.fixed_batch_normalization(
                x,
                gamma,
                beta,
                mean,
                var,
                eps=epsilon
            ).array,
        )

    return fn


def pad_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    x_io = node.inputs["data"]
    y_io = node.outputs["output"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    logging.log(logging.WARN, "Pad is currently a NOP")

    def fn():
        parray.copy(y, x)

    return fn


def average_pool_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    """
        Function:
                Y = AVERAGE_POOL(X)
            --> CONVE NCHW to NC11 (Average on HW dimensions)
    """

    x_io = node.inputs["X"]
    y_io = node.outputs["Y"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)

    kernel_size = node.get_attr("kernel_shape")
    padding = node.get_attr("pads", [0])[0]
    stride = node.get_attr("strides", [0])[0]

    def fn():
        out = chainer.functions.average_pooling_2d(
            x, kernel_size, stride=stride, pad=padding
        ).array
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
        out = chainer.functions.average_pooling_2d(x, (x.shape[2], x.shape[3])).array
        parray.copy(y, out)

    return fn


def globalAveragePool_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

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
        out = chainer.functions.average_pooling_2d(x, (x.shape[2], x.shape[3])).array
        cupy.copyto(y, out)

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


def flatten_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

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
        cupy.copyto(y, cupy.reshape(x, (x.shape[0], -1)))

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

        parray.copy(y, (alpha * (xt @ wt)) + (beta * b))

    return fn


def gemm_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

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
            xt = cupy.transpose(x)
        else:
            xt = x
        if transW == 1:
            wt = cupy.transpose(w)
        else:
            wt = w

        cupy.copyto(y, (alpha * (xt @ wt)) + (beta * b))

    return fn
