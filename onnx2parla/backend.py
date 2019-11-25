import numpy as np
import networkx as nx
import math

import logging
from typing import Dict, Callable

from parla import cpu as pcpu
from parla import cuda as pcuda
from parla import tasks as ptasks

import cupy

from node import Node, InOut
import kernels
from config import Config
import operators as ops


def build_copy_node(in_io, out_io, node_id):
    inputs = {"X": in_io}
    outputs = {"Z": out_io}
    attrs = {}
    new_node = Node(node_id, ops.O2P_COPY, inputs, outputs, attrs, 0)

    return new_node


def copy_insertion(graph: nx.DiGraph, alloc_map, config: Config) -> None:
    """
    If an output of a node is on a different device
     1) we need to insert a copy from cpu to gpu
     2) we need to rename the node that is used on the gpu to ensure that all
        names are globally unique
    eg

    a b               a b
    | |               | |
    add @ cpu  -----> add @ cpu
     |                 |
     c                 c
     |                 |
    relu @ gpu        copy @ gpu
                       |
                       c_gpu
                       |
                      relu @ gpu

    We also want to optimize the case where one output on device 1 may be used
    as multiple inputs on device 2. In this case we only insert one copy and
    allocate only 1 buffer on each device instead of duplicating the buffer
    """

    # get the starting id for new nodes
    node_id = graph.number_of_nodes()
    total_nodes = node_id

    # manually iterate over all the known node id's before we did modifications
    # to the graph. This way we can modify the graph as we traverse it
    for gnode in range(total_nodes):
        node = graph.nodes[gnode]["node"]

        # here we store a list of nodes for each unique set of
        # buffer-device pairs in a heirarchially arranged map
        # Each of these unique pairs will have a copy node
        # generated for it

        output_groups = {}  # {buffer: {device: [node ids]}

        for gchild in graph.successors(gnode):
            child = graph.nodes[gchild]["node"]
            buffer = graph.edges[gnode, gchild]["buffer"]
            device = child.get_device()

            if buffer not in output_groups:
                output_groups[buffer] = {}

            if device not in output_groups[buffer]:
                output_groups[buffer][device] = []

            output_groups[buffer][device].append(gchild)

        # now that we know all of the nodes in each of the buffer-device
        # pairs we iterate though the output buffers of the parent node
        # and insert a new copy node for each buffer-device pair
        for output_io in node.outputs.values():
            active_buffer = output_io.name

            for device, gchildren in output_groups[buffer].items():
                if device != node.get_device():
                    buffer_on_device = "{}_{}_{}".format(
                        active_buffer, device[0], device[1]
                    )
                    copy_in_io = InOut(active_buffer, "pointer", None, output_io.shape)

                    copy_out_io = InOut(
                        buffer_on_device, "pointer", None, output_io.shape
                    )

                    new_copy_node = build_copy_node(copy_in_io, copy_out_io, node_id)
                    new_copy_node.device_type = device[0]
                    new_copy_node.device_id = device[1]

                    graph.add_node(node_id)
                    graph.nodes[node_id]["node"] = new_copy_node
                    graph.add_edge(gnode, node_id, buffer=active_buffer)

                    for gchild in gchildren:
                        child = graph.nodes[gchild]["node"]
                        child.replace_io_for_input_buffer(active_buffer, copy_out_io)

                        graph.add_edge(node_id, gchild, buffer=buffer_on_device)
                        graph.remove_edge(gnode, gchild)

                    node_id += 1


def place_n_opt(
    graph: nx.DiGraph, alloc_map: Dict[str, np.ndarray], config: Config
) -> None:
    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]
        if node.operator == ops.MAXPOOL:
            node.device_type = "cpu"
            node.device_id = 0
        else:
            node.device_type = "cpu"
            node.device_id = 0
    return


def allocate(
    graph: nx.DiGraph, alloc_map: Dict[str, np.ndarray], config: Config
) -> None:
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
                with cupy.cuda.Device(node.device_id):
                    io.data = cupy.array(io.data)
    return


def build_graph(
    graph: nx.DiGraph, alloc_map: Dict[str, np.ndarray], config: Config
) -> None:
    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]

        node.fn = build_kernel(node, alloc_map, config)
        logging.log(logging.INFO, f"SAA -> {node.node_id} {node.operator}")

    return


def build_kernel(
    node: Node, alloc_map: Dict[str, np.ndarray], config: Config
) -> Callable[[], None]:

    """
    For each node in graph build a function for execution on the correct device
    """

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
        if node.device_type == "cpu":
            return kernels.conv_cpu(node, alloc_map, config)
        else:
            return kernels.conv_gpu(node, alloc_map, config)
    if oper == ops.BATCH_NORM:
        if node.device_type == "cpu":
            return kernels.batchnorm_cpu(node, alloc_map, config)
        else:
            return kernels.batchnorm_gpu(node, alloc_map, config)
    if oper == ops.RELU:
        if node.device_type == "cpu":
            return kernels.relu_cpu(node, alloc_map, config)
        else:
            return kernels.relu_gpu(node, alloc_map, config)
    if oper == ops.MAXPOOL:
        if node.device_type == "cpu":
            return kernels.maxpool_cpu(node, alloc_map, config)
        else:
            return kernels.maxpool_gpu(node, alloc_map, config)
    if oper == ops.GLOBALAVERAGEPOOL:
        if node.device_type == "cpu":
            return kernels.globalAveragePool_cpu(node, alloc_map, config)
        else:
            return kernels.globalAveragePool_gpu(node, alloc_map, config)
    if oper == ops.FLATTEN:
        if node.device_type == "cpu":
            return kernels.flatten_cpu(node, alloc_map, config)
        else:
            return kernels.flatten_gpu(node, alloc_map, config)
    if oper == ops.GEMM:
        if node.device_type == "cpu":
            return kernels.gemm_cpu(node, alloc_map, config)
        else:
            return kernels.gemm_gpu(node, alloc_map, config)

    raise ValueError(f"Operator {oper} not supported")


def build_execute(graph: nx.DiGraph, config: Config) -> Callable[[], None]:
    async def fn():
        async with ptasks.finish():

            batches = math.ceil(config.dataset_len / config.batch_size)

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

                    node.last_task_obj = ptasks.spawn(placement=loc, dependencies=deps)(
                        node.fn
                    )

                    logging.log(logging.INFO, "---{}---".format(node.operator))
                    logging.log(logging.INFO, "launched {}".format(node.last_task_obj))
                    logging.log(logging.INFO, "deps {}".format(deps))

    return fn
