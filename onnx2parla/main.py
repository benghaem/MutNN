import sys
import numpy as np
import logging
import networkx as nx

import backend
import onnx_frontend as frontend
from config import Config
from node import node_stringizer

from parla import cpu as pcpu
from parla import tasks as ptasks

logging.basicConfig(filename="full.log", level=logging.DEBUG)

# User functions


def random_data(start_idx, end_idx):
    batches = end_idx - start_idx
    data = np.random.random((batches, 10))

    return data


def echo_store(arr):
    logging.log(logging.INFO, f"stored: {arr}")


def debug_print_graph(graph):
    for node_id in graph.nodes:
        node = graph.nodes[node_id]["node"]
        node.pretty_print()


config = Config(echo_store, random_data, 4, 4)
graph = frontend.from_onnx(sys.argv[1], config)

amap = {}
debug_print_graph(graph)

passes = [
    backend.place_n_opt,
    backend.copy_insertion,
    backend.allocate,
    backend.build_graph,
]

for i, opass in enumerate(passes):
    opass(graph, amap, config)

    print("---pass: {}---".format(opass.__name__))
    debug_print_graph(graph)
    nx.write_gml(graph, opass.__name__ + ".gml", node_stringizer)

# run everything!
ptasks.spawn(placement=pcpu.cpu(0))(backend.build_execute(graph, config))
