import numpy as np
import kernels
import networkx as nx
import parla
import logging
from parla import cpu as pcpu
from parla import tasks as ptasks

# 
# allocate
# @parameters
# raw_graph: the input graph without any annotations
# alloc_map: the empty input map
# @returns
# nothing
def allocate(graph, alloc_map):
    for gnode in graph.nodes:
        node = graph.nodes[gnode]['node']

        #iterate through all outputs and allocate storage
        #based upon shape
        for io in node.outputs.values():
            if (io.kind == "pointer"):
                alloc_map[io.name] = np.ndarray(io.shape)

        #iterate through all inputs and allocate storage
        #based upon shape if they are of kind dynamic
        for io in node.inputs.values():
            if (io.kind == "dynamic"):
                alloc_map[io.name] = np.ndarray(io.shape)
    return

def place_n_opt(graph, alloc_map):
    for gnode in graph.nodes:
        node = graph.nodes[gnode]['node']
        node.device_type = "cpu"
        node.device_id = "0"

    return

def build_graph(graph, alloc_map):
    for gnode in graph.nodes:
        node = graph.nodes[gnode]['node']

        node.fn = build_kernel(node, alloc_map)

    return

def build_kernel(node, alloc_map):
    if node.operator == "add":
        if (node.device_type == "cpu"):
            return kernels.add_cpu(node, alloc_map)

def build_execute(graph):
    async def fn():
        async with ptasks.finish():

            #TODO: manage dependencies
            for gnode in nx.bfs_tree(graph, 0).nodes():
                logging.log(logging.INFO, "execute node: {}".format(gnode))
                node = graph.nodes[gnode]['node']
                ptasks.spawn(placement=pcpu.cpu(0))(node.fn)


    return fn





