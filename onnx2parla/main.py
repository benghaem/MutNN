import sys
import numpy as np
import logging

import backend
import onnx_frontend as frontend
from config import Config

from parla import cpu as pcpu
from parla import tasks as ptasks

logging.basicConfig(filename="backend.log", level=logging.INFO)

# User functions


def echo_idx(start_idx, end_idx):
    data = [start_idx + i for i in range(end_idx - start_idx)]
    return np.array(data)


def echo_store(arr):
    logging.log(logging.INFO, f"stored: {arr}")


def debug_print_graph(graph):
    for node_id in graph.nodes:
        node = graph.nodes[node_id]["node"]
        node.pretty_print()


config = Config(echo_store, echo_idx, 4, 4 * 10000)
graph = frontend.from_onnx(sys.argv[1], config)

amap = {}
print(amap)
debug_print_graph(graph)

passes = [backend.place_n_opt, backend.allocate, backend.build_graph]

for i, opass in enumerate(passes):
    opass(graph, amap, config)

    print("---pass: {}---".format(opass.__name__))
    print(amap)
    debug_print_graph(graph)

# run everything!
ptasks.spawn(placement=pcpu.cpu(0))(backend.build_execute(graph, config))
