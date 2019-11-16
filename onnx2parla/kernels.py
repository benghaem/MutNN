import numpy as np
import numpy
import parla.array as parray
import logging

import cupy

from node import Node
from config import Config

from typing import Callable, Dict

def load_cpu(node: Node, alloc_map: Dict[str, np.ndarray], config):
    z_io = node.outputs["Z"]
    z = z_io.get_data(alloc_map)

    width = node.get_attr("width")
    batch_width = config.batch_width

    def fn():
        batch_id = node.get_attr("batch_id")
        start_idx = batch_id * batch_width + node.instance_id * width
        end_idx = start_idx + width
        parray.copy(z, config.user_load_fn(start_idx, end_idx))
        batch_id +=1
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

    tx = type(x)
    tz = type(z)

    def fn():
        if tz == numpy.ndarray: #to cpu
            np.copyto(z,cupy.asnumpy(x))
        if tz == cupy.core.core.ndarray: #to gpu
            with cupy.cuda.Device(node.device_id):
                cupy.copyto(z,cupy.asarray(x))

        logging.log(logging.INFO, f"done copy {z}, {tz}")


    return fn


def add_cpu(node: Node, alloc_map: Dict[str, np.ndarray], config: Config) -> Callable[[], None]:

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

    x_io = node.inputs["X"]
    y_io = node.inputs["Y"]
    z_io = node.outputs["Z"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)
    z = z_io.get_data(alloc_map)

    def fn():
        parray.copy(z, x + y)

    return fn

def add_gpu(node: Node, alloc_map, config: Config) -> Callable[[], None]:

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

    x_io = node.inputs["X"]
    y_io = node.inputs["Y"]
    z_io = node.outputs["Z"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)
    z = z_io.get_data(alloc_map)

    def fn():
        cupy.copyto(z, x+y)
        logging.log(logging.INFO, f"{z} = {x} + {y}")

    return fn
