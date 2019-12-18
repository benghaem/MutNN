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

# user_width: per gpu batch size
# batch size: the total width of the batch

# batch size = sum (width_device)


def load_cpu(node: Node, alloc_map, config):
    z_io = node.outputs["Z"]
    z = z_io.get_data(alloc_map)

    width = z_io.shape[0]
    batch_size = config.computed_batch_size

    def fn():
        batch_id = node.get_attr("batch_id")
        # indexing is setup for the static case
        # this indexing works for batch size > gpu_supported
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
        store_id = node.get_attr("store_id")
        logging.log(logging.INFO, f"Storing {store_id}")
        config.user_store_fn(x)
        node.set_attr("store_id", store_id + 1)

    return fn


def stream_callback_fn(stream, error, user):
    logging.log(logging.INFO, f"{stream} -> {error} ({user})")


def copy(node: Node, alloc_map, config: Config):
    x_io = node.inputs["X"]
    z_io = node.outputs["Z"]

    x = x_io.get_data(alloc_map)
    z = z_io.get_data(alloc_map)

    source_device_id = node.get_attr("source_device")[1]
    target_device_id = node.get_attr("target_device")[1]

    tz = type(z)
    tx = type(x)

    def fn():
        # time_st = datetime.datetime.now()

        if tz == numpy.ndarray:  # to cpu
            np.copyto(z, cupy.asnumpy(x))
            # assert cupy.testing.assert_array_equal(z,x)

        if tz == cupy.core.core.ndarray and tx != cupy.core.core.ndarray:  # to gpu
            with cupy.cuda.Device(node.device_id):
                cupy.copyto(z, cupy.asarray(x))

        if tz == cupy.core.core.ndarray and tx == cupy.core.core.ndarray:  # to gpu
            tmp = None
            with cupy.cuda.Device(source_device_id):
                tmp = cupy.asnumpy(x)
            with cupy.cuda.Device(target_device_id):
                cupy.copyto(z, cupy.asarray(tmp))

            # assert cupy.testing.assert_array_equal(z,x)

            # assert z.shape == x.shape
            # cupy.cuda.get_current_stream().synchronize()
            # tmp = cupy.asarray(x)
            # cupy.cuda.get_current_stream().synchronize()

            # neq = cupy.count_nonzero(cupy.logical_not(z==tmp))
            # print(neq)
            # assert cupy.testing.assert_array_equal(z,tmp)
            # to gpu:

        # og_shape = x.shape

        # if tz == numpy.ndarray:  # to cpu
        #    with cupy.cuda.Device(device=node.device_id):
        #        arr_flat = x.reshape((-1))
        #        z_flat = np.ndarray(arr_flat.shape)

        #        for i, v in enumerate(arr_flat):
        #            z_flat[i] = v

        #        z_flat = z_flat.reshape(og_shape)

        #        np.copyto(z,z_flat)

        # if tz == cupy.core.core.ndarray:

        #    arr_flat = x.reshape((-1))

        #    with cupy.cuda.Device(device=node.device_id):
        #        z_flat = cupy.ndarray(arr_flat.shape)

        #        for i, v in enumerate(arr_flat):
        #            z_flat[i] = v

        #        z_flat = z_flat.reshape(og_shape)

        #        cupy.copyto(z,z_flat)

        # time_end = datetime.datetime.now()
        # logging.log(logging.INFO, f"done copy {z}, {tz}")
        # logging.log(logging.INFO, f"TIMER: <{node.operator},{node.node_id} {time_st} -> {time_end}")

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
        with cupy.cuda.Device(node.device_id):
            cupy.add(x,y,out=z)

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
    groups = node.get_attr("group", 1)

    def fn():
        np.copyto(
            y,
            (
                chainer.functions.convolution_2d(
                    x,
                    w,
                    b=b,
                    stride=stride,
                    pad=padding,
                    dilate=dilations,
                    groups=groups,
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
    groups = node.get_attr("group", 1)

    stride = (stride,stride) 
    padding = (padding,padding)
    dilations = (dilations,dilations)

    def fn():
        # time_st = datetime.datetime.now()
        # logging.log(logging.INFO, f"CONVOP got -->  {x[-1]} CONVOP")

        with cupy.cuda.Device(node.device_id):

            cupy.cudnn.convolution_forward(
                x,
                w,
                b,
                y,
                padding,
                stride,
                dilations,
                groups,
                auto_tune = False,
                tensor_core = 'auto'
            )

        # time_end = datetime.datetime.now()
        # logging.log(logging.INFO, f"TIMER: <{node.operator},{node.node_id}> {time_st} -> {time_end}")
        # logging.log(logging.INFO, f"CONV sent -->  {y[-1]} CONV")

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
        with cupy.cuda.Device(node.device_id):
            #hack to get the copy off the default stream
            cupy.add(cupy.cudnn.activation_forward(x, cupy.cuda.cudnn.CUDNN_ACTIVATION_RELU),0,out=y)

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
        # time_st = datetime.datetime.now()
        x_pad = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )
        batches, c, h, w = x.shape
        out_h = np.floor(((h - kernel_shape[0] + 2 * padding) / stride) + 1).astype(int)
        out_w = np.floor(((w - kernel_shape[1] + 2 * padding) / stride) + 1).astype(int)
        out = np.zeros((batches, c, out_h, out_w))
        for i in range(batches):
            for j in range(c):
                for p in range(out_h):
                    for q in range(out_w):
                        p0, p1 = p * stride, (p * stride) + kernel_shape[0]
                        q0, q1 = q * stride, (q * stride) + kernel_shape[1]
                        out[i, j, p, q] = np.max(x_pad[i, j, p0:p1, q0:q1])
        np.copyto(y, out)
        # time_end = datetime.datetime.now()
        # logging.log(logging.INFO, f"TIMER: <{node.operator},{node.node_id}> {time_st} -> {time_end}")

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
        with cupy.cuda.Device(node.device_id):
            cupy.cudnn.pooling_forward(
                x,
                y,
                (kernel_shape[0], kernel_shape[1]),
                (stride, stride),
                (padding, padding),
                cupy.cuda.cudnn.CUDNN_POOLING_MAX,
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
    if epsilon < 1e-05:
        epsilon = 1e-05

    def fn():

        # logging.log(logging.INFO, f"BATCHNORM got -->  {x[-1]} BATCHNORM")
        np.copyto(
            y,
            chainer.functions.fixed_batch_normalization(
                x, gamma, beta, mean, var, eps=epsilon
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
    if epsilon < 1e-05:
        epsilon = 1e-05

    # modeled off of the chainer support


    def fn():
        with cupy.cuda.Device(node.device_id):

            #This is a hack to avoid the device to device copy stuck on the default stream
            cupy.add(cupy.cudnn.batch_normalization_forward_inference(
                x, gamma, beta, mean, var, epsilon, True,
                cupy.cuda.cudnn.CUDNN_BATCHNORM_SPATIAL
            ),0,out=y)

            #cupy.copyto(
            #    y,
            #    chainer.functions.fixed_batch_normalization(
            #        x, gamma, beta, mean, var, eps=epsilon
            #    ).array,
            #)

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
    stride = node.get_attr("strides", [1])[0]

    def fn():
        out = chainer.functions.average_pooling_2d(
            x, kernel_size, stride=stride, pad=padding
        ).array
        np.copyto(y, out)

    return fn


def average_pool_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

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
    stride = node.get_attr("strides", [1])[0]

    def fn():
        with cupy.cuda.Device(node.device_id):
            out = chainer.functions.average_pooling_2d(
                x, kernel_size, stride=stride, pad=padding
            ).array
            cupy.copyto(y, out)

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
        with cupy.cuda.Device(node.device_id):
            out = chainer.functions.average_pooling_2d(
                x, (x.shape[2], x.shape[3])
            ).array
            #hack to force copy off of default stream
            cupy.add(out, 0, out=y)
            # logging.log(logging.INFO, f"On GPU {node.device_id}")

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
        with cupy.cuda.Device(node.device_id):
            # hack to get copy off the default stream
            cupy.add(cupy.reshape(x, (x.shape[0], -1)),0, out=y)

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
            xt = chainer.functions.transpose(x)
        else:
            xt = x
        if transW == 1:
            wt = w
        else:
            wt = chainer.functions.transpose(w)

        np.copyto(y, chainer.functions.linear(alpha * xt, wt, b=(beta * b)).array)

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
        with cupy.cuda.Device(node.device_id):
            if transX == 1:
                #xt = chainer.functions.transpose(x)
                xt = cupy.transpose(x)
            else:
                xt = x
            if transW == 1:
                #wt = chainer.functions.transpose(w)
                wt = cupy.transpose(w)
            else:
                wt = w

            z = cupy.dot(alpha * xt, wt)
            cupy.add(z, beta * b, out = y)

    return fn


def dropout_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    data_io = node.inputs["data"]
    output_io = node.outputs["output"]
    opt_mask_io = node.get_output("mask")

    data = data_io.get_data(alloc_map)
    output = output_io.get_data(alloc_map)
    opt_mask = opt_mask_io.get_data(alloc_map)

    ratio = node.get_attr("ratio", 0.5)

    def fn():
        if opt_mask:
            o, m = chainer.functions.dropout(data, ratio=ratio, return_mask=True)
            np.copyto(output, o.array)
            np.copyto(opt_mask, m.array)
        else:
            np.copyto(output, chainer.functions.dropout(data, ratio=ratio).array)

    return fn


def dropout_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    data_io = node.inputs["data"]
    output_io = node.outputs["output"]
    opt_mask_io = node.get_output("mask")

    data = data_io.get_data(alloc_map)
    output = output_io.get_data(alloc_map)
    opt_mask = opt_mask_io.get_data(alloc_map)

    ratio = node.get_attr("ratio", 0.5)

    def fn():
        with cupy.cuda.Device(node.device_id):
            if opt_mask:
                o, m = chainer.functions.dropout(data, ratio=ratio, return_mask=True)
                cupy.copyto(output, o.array)
                cupy.copyto(opt_mask, m.array)
            else:
                cupy.copyto(output, chainer.functions.dropout(data, ratio=ratio).array)

    return fn


def reshape_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    data_io = node.inputs["data"]
    shape_io = node.inputs["shape"]
    reshaped_io = node.outputs["reshaped"]

    data = data_io.get_data(alloc_map)
    shape = shape_io.get_data(alloc_map)
    reshaped = reshaped_io.get_data(alloc_map)

    def fn():
        with cupy.cuda.Device(node.device_id):

            # Unfortunately we need to convert this cupy array to a tuple
            # this will force a copy back to host :(
            # TODO: support types on the frontend so we don't have to convert
            # here
            shape_tuple = list(cupy.asnumpy(shape).astype(np.int64))
            # correct shape tuple to match batch size
            shape_tuple[0] = data.shape[0]
            shape_tuple = tuple(shape_tuple)

            cupy.copyto(reshaped, chainer.functions.reshape(data, shape_tuple).array)

    return fn


def reshape_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    data_io = node.inputs["data"]
    shape_io = node.inputs["shape"]
    reshaped_io = node.outputs["reshaped"]

    data = data_io.get_data(alloc_map)
    shape = shape_io.get_data(alloc_map)
    reshaped = reshaped_io.get_data(alloc_map)

    def fn():
        shape_tuple = list(shape.astype(np.int64))
        shape_tuple[0] = data.shape[0]
        shape_tuple = tuple(shape_tuple)
        np.copyto(reshaped, chainer.functions.reshape(data, shape_tuple).array)

    return fn


def clip_v11_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    input_io = node.inputs["input"]
    min_io = node.get_input("min")
    max_io = node.get_input("max")

    output_io = node.outputs["output"]

    inp = input_io.get_data(alloc_map)
    min_data = min_io.get_data(alloc_map)
    if min_data is None:
        min_data = [-np.inf]
    max_data = max_io.get_data(alloc_map)
    if max_data is None:
        max_data = [np.inf]

    output = output_io.get_data(alloc_map)

    def fn():
        np.copyto(output, chainer.functions.clip(inp, min_data[0], max_data[0]).array)

    return fn


def clip_v6_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    input_io = node.inputs["input"]
    min_v = node.get_attr("min", -3.402823e38)
    max_v = node.get_attr("max", 3.402823e38)

    output_io = node.outputs["output"]

    inp = input_io.get_data(alloc_map)
    output = output_io.get_data(alloc_map)

    def fn():
        with cupy.cuda.Device(node.device_id):
            cupy.clip(inp, a_min=min_v, a_max=max_v, out=output)

    return fn


def clip_v6_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    input_io = node.inputs["input"]
    min_v = node.get_attr("min", -3.402823e38)
    max_v = node.get_attr("max", 3.402823e38)

    output_io = node.outputs["output"]

    inp = input_io.get_data(alloc_map)
    output = output_io.get_data(alloc_map)

    def fn():
        np.copyto(output, chainer.functions.clip(inp, min_v, max_v).array)

    return fn


def clip_v11_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    input_io = node.inputs["input"]
    min_io = node.get_input("min")
    max_io = node.get_input("max")

    output_io = node.outputs["output"]

    inp = input_io.get_data(alloc_map)
    min_data = min_io.get_data(alloc_map)
    if min_data is None:
        min_data = cupy.array([float("-inf")])

    max_data = max_io.get_data(alloc_map)
    if max_data is None:
        max_data = cupy.array([float("inf")])

    output = output_io.get_data(alloc_map)

    def fn():
        with cupy.cuda.Device(node.device_id):
            cupy.copyto(
                output, chainer.functions.clip(inp, min_data[0], max_data[0]).array
            )

    return fn


def reduce_mean_cpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    data_io = node.inputs["data"]
    reduced_io = node.outputs["reduced"]

    data = data_io.get_data(alloc_map)
    reduced = reduced_io.get_data(alloc_map)

    axes = node.get_attr("axes")
    keep_dims = node.get_attr("keepdims", 1) == 1

    def fn():
        np.mean(data, axis=axes, out=reduced, keepdims=keep_dims)

    return fn


def reduce_mean_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

    data_io = node.inputs["data"]
    reduced_io = node.outputs["reduced"]

    data = data_io.get_data(alloc_map)
    reduced = reduced_io.get_data(alloc_map)

    axes = node.get_attr("axes")
    keep_dims = node.get_attr("keepdims", 1) == 1

    def fn():
        with cupy.cuda.Device(node.device_id):
            cupy.mean(data, axis=axes, out=reduced, keepdims=keep_dims)

    return fn

