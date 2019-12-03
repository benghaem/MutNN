import numpy as np
import numpy
import logging
import datetime

import time
import cupy

import chainer
from chainer import functions as gputils

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
        np.copyto(z, config.user_load_fn(start_idx, end_idx))
        batch_id += 1
        node.set_attr("batch_id", batch_id)

    return fn


def store_cpu(node, alloc_map, config):
    x_io = node.inputs["X"]
    x = x_io.get_data(alloc_map)

    def fn():
        config.user_store_fn(x)

    return fn

def stream_callback_fn(stream, error, user):
    logging.log(logging.INFO, f"{stream} -> {error} ({user})")

def copy(node: Node, alloc_map, config: Config):
    x_io = node.inputs["X"]
    z_io = node.outputs["Z"]

    x = x_io.get_data(alloc_map)
    z = z_io.get_data(alloc_map)

    tz = type(z)

    def fn():
        time_st = datetime.datetime.now()
        logging.log(logging.INFO, cupy.cuda.get_current_stream())

        if tz == numpy.ndarray:  # to cpu
            np.copyto(z,cupy.asnumpy(x))
            #assert cupy.testing.assert_array_equal(z,x)

        if tz == cupy.core.core.ndarray:  # to gpu
            #with cupy.cuda.Device(node.device_id):
            cupy.copyto(z,cupy.asarray(x))
            #assert cupy.testing.assert_array_equal(z,x)

            #assert z.shape == x.shape
            #cupy.cuda.get_current_stream().synchronize()
            #tmp = cupy.asarray(x)
            #cupy.cuda.get_current_stream().synchronize()

            #neq = cupy.count_nonzero(cupy.logical_not(z==tmp))
            #print(neq)
            #assert cupy.testing.assert_array_equal(z,tmp)
                # to gpu:

        #og_shape = x.shape


        #if tz == numpy.ndarray:  # to cpu
        #    with cupy.cuda.Device(device=node.device_id):
        #        arr_flat = x.reshape((-1))
        #        z_flat = np.ndarray(arr_flat.shape)

        #        for i, v in enumerate(arr_flat):
        #            z_flat[i] = v

        #        z_flat = z_flat.reshape(og_shape)

        #        np.copyto(z,z_flat)


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
        #logging.log(logging.INFO, f"TIMER: <{node.operator},{node.node_id} {time_st} -> {time_end}")

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
        np.copyto(z, x + y)

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

    x_io = node.inputs["A"]
    y_io = node.inputs["B"]
    z_io = node.outputs["C"]

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
        np.copyto(
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
        logging.log(logging.INFO, f"CONVOP got -->  {x[-1]} CONVOP")
        #with cupy.cuda.Device(node.device_id):
        cupy.copyto(
            y,
            (
                chainer.functions.convolution_2d(
                    x, w, b, stride=stride, pad=padding, dilate=dilations,
                )
            ).array,
        )

        time_end = datetime.datetime.now()
        logging.log(logging.INFO, f"TIMER: <{node.operator},{node.node_id}> {time_st} -> {time_end}")
        logging.log(logging.INFO, f"CONV sent -->  {y[-1]} CONV")

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
        np.copyto(y, chainer.functions.relu(x).array)

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
        cupy.copyto(y, chainer.functions.relu(x).array)

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
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant', constant_values=0)
        batches, c, h, w = x.shape
        out_h = np.floor(((h - kernel_shape[0] + 2*padding)/stride) + 1).astype(int)
        out_w = np.floor(((w - kernel_shape[1] + 2*padding)/stride) + 1).astype(int)
        out = np.zeros((batches,c,out_h,out_w))
        for i in range(batches):
            for j in range(c):
                for p in range(out_h):
                    for q in range(out_w):
                        p0, p1 = p * stride, (p * stride) + kernel_shape[0]
                        q0, q1 = q * stride, (q * stride) + kernel_shape[1]
                        out[i, j, p, q] = np.max(x_pad[i, j, p0:p1, q0:q1])
        np.copyto(y, out)
        time_end = datetime.datetime.now()
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
        out = cupy.zeros_like(y)

        chainer.backends.cuda.cudnn.pooling_forward(
            x, out,
            (kernel_shape[0], kernel_shape[1]), (stride, stride), (padding, padding),
            chainer.backends.cuda.cuda.cudnn.CUDNN_POOLING_MAX)

        cupy.copyto(y, out)

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
    if (epsilon < 1e-05):
        epsilon = 1e-05
    def fn():

        logging.log(logging.INFO, f"BATCHNORM got -->  {x[-1]} BATCHNORM")
        np.copyto(
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
    if (epsilon < 1e-05):
        epsilon = 1e-05
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
        np.copyto(y, x)

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
        np.copyto(y, out)

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
        np.copyto(y, out)

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
        np.copyto(y, x.reshape(x.shape[0], -1))

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

        np.copyto(y, (alpha * (xt @ wt)) + (beta * b))

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
