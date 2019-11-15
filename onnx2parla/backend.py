import numpy as np
import networkx as nx
import math

import logging
from typing import Dict, Callable

import parla
from parla import cpu as pcpu
from parla import tasks as ptasks

from node import Node
import kernels
from config import Config

def allocate(graph: nx.DiGraph, alloc_map: Dict[str, np.ndarray], config: Config) -> None:
    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]

        # iterate through all outputs and allocate storage
        # based upon shape
        for io in node.outputs.values():
            if io.kind == "pointer":
                alloc_map[io.name] = np.ndarray(io.shape)

        # iterate through all inputs and allocate storage
        # based upon shape if they are of kind dynamic
        #for io in node.inputs.values():
        #    if io.kind == "dynamic":
        #        alloc_map[io.name] = np.ndarray(io.shape)
    return


def place_n_opt(graph: nx.DiGraph, alloc_map: Dict[str, np.ndarray], config: Config) -> None:
    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]
        node.device_type = "cpu"
        node.device_id = "0"

    return


def build_graph(graph: nx.DiGraph, alloc_map: Dict[str, np.ndarray], config: Config) -> None:
    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]

        node.fn = build_kernel(node, alloc_map, config)

    return


def build_kernel(node: Node, alloc_map: Dict[str, np.ndarray], config: Config) -> Callable[[], None]:

    oper = node.get_operator()
    if oper == "add":
        if node.device_type == "cpu":
            return kernels.add_cpu(node, alloc_map, config)
    if oper == "load":
        return kernels.load_cpu(node, alloc_map, config)
    if oper == "store":
        return kernels.store_cpu(node, alloc_map, config)

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
                    ptasks.spawn(placement=pcpu.cpu(0))(node.fn)


    return fn
