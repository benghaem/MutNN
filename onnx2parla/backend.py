import numpy as np
import networkx as nx
import math

import logging
from typing import Dict, Callable

from parla import cpu as pcpu
from parla import cuda as pcuda
from parla import tasks as ptasks

import cupy

from node import Node
import kernels
from config import Config
import operators as ops


def place_n_opt(graph: nx.DiGraph, alloc_map: Dict[str, np.ndarray], config: Config) -> None:
    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]
        # if (node.operator == ops.ADD or node.operator == ops.O2P_COPY):
        #     node.device_type = "gpu"
        #     node.device_id = 0
        # else:
        #     node.device_type = "cpu"
        #     node.device_id = 0
        node.device_type = "cpu"
        node.device_id = 0
    return


def allocate(graph: nx.DiGraph, alloc_map: Dict[str, np.ndarray], config: Config) -> None:
    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]

        # iterate through all outputs and allocate storage
        # based upon shape
        if node.operator == "copy":
            io_z = node.outputs["Z"]

            # invariant -- all successors to a copy will
            # be on the same node
            scc_gnode = graph.successors(gnode).__next__()
            scc_node = graph.nodes[scc_gnode]["node"]
            # check successor
            if scc_node.device_type == "gpu":
                with cupy.cuda.Device(scc_node.device_id):
                    alloc_map[io_z.name] = cupy.ndarray(io_z.shape)
            else:
                alloc_map[io_z.name] = np.ndarray(io_z.shape)
        else:
            for io in node.outputs.values():
                if io.kind == "pointer":
                    if node.device_type == "gpu":
                        with cupy.cuda.Device(node.device_id):
                            alloc_map[io.name] = cupy.ndarray(io.shape)
                    else:
                        alloc_map[io.name] = np.ndarray(io.shape)

        for io in node.inputs.values():
            if io.kind == "static" and node.device_type == "gpu":
                print(node.operator, io.data, type(io.data))
                with cupy.cuda.Device(node.device_id):
                    io.data = cupy.array(io.data)
                print(node.operator, io.data, type(io.data))
    return


def build_graph(graph: nx.DiGraph, alloc_map: Dict[str, np.ndarray], config: Config) -> None:
    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]

        node.fn = build_kernel(node, alloc_map, config)

    return


def build_kernel(node: Node, alloc_map: Dict[str, np.ndarray], config: Config) -> Callable[[], None]:

    oper = node.get_operator()
    if oper == ops.ADD:
        if node.device_type == "cpu":
            return kernels.add_cpu(node, alloc_map, config)
        else:
            return kernels.add_gpu(node, alloc_map, config)
    if oper == ops.O2P_LOAD:
        return kernels.load_cpu(node, alloc_map, config)
    if oper == ops.O2P_STORE:
        return kernels.store_cpu(node, alloc_map, config)
    if oper == ops.O2P_COPY:
        return kernels.copy(node, alloc_map, config)

    if oper == ops.CONV:
    	return kernels.conv_cpu(node, alloc_map, config)
    if oper == ops.RELU:
    	return kernels.relu_cpu(node, alloc_map, config)

    raise ValueError(f"Operator {oper} not supported")


def build_execute(graph: nx.DiGraph, config: Config) -> Callable[[], None]:
    async def fn():
        async with ptasks.finish():

            # TODO: manage dependencies
            # We could do this by storing the task object onto the node
            # subsequent runs can update and replace this (or append)

            # ex: deps = [parent.task[batch_id] for parent in parents] +
            #            [node.task[batch_id-1]
            # node.task.append(spawn(dependencies = deps))

            batches = math.ceil(config.dataset_len / config.batch_width)

            for batch_id in range(batches):
                for gnode in nx.bfs_tree(graph, 0).nodes():
                    logging.log(logging.INFO, "execute node: {}".format(gnode))

                    node = graph.nodes[gnode]["node"]

                    deps = []
                    # get parents and get children
                    for gparent in graph.predecessors(gnode):
                        lto = graph.nodes[gparent]["node"].last_task_obj
                        if lto:
                            deps.append(lto)

                    for gchild in graph.successors(gnode):
                        lto = graph.nodes[gchild]["node"].last_task_obj
                        if lto:
                            deps.append(lto)

                    loc = None
                    if node.device_type == "cpu":
                        loc = pcpu.cpu(node.device_id)
                    else:
                        loc = pcuda.gpu(node.device_id)

                    node.last_task_obj = ptasks.spawn(placement=loc, dependencies=deps)(node.fn)

                    logging.log(logging.INFO, "---{}---".format(node.operator))
                    logging.log(logging.INFO, "launched {}".format(node.last_task_obj))
                    logging.log(logging.INFO, "deps {}".format(deps))

    return fn
