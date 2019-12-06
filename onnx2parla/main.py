import sys
import numpy as np
import logging
import networkx as nx

import backend
import onnx_frontend as frontend
from config import Config
from node import node_stringizer
import datetime

import resnet_data

# from parla import cpu as pcpu
from parla import cpucores as pcpu_cores
from parla import tasks as ptasks

# disable logging for benchmarking
logging.disable()

# User functions


def random_data(start_idx, end_idx):
    batches = end_idx - start_idx
    data = np.ones((batches, 10))

    return data


def echo_store(arr):
    sm = sorted(zip(arr[0], range(len(arr[0]))), reverse=True)[0:5]
    logging.log(logging.INFO, f"stored: {sm}")


def debug_print_graph(graph):
    for node_id in graph.nodes:
        node = graph.nodes[node_id]["node"]
        node.pretty_print()


config = Config(resnet_data.nop_store, resnet_data.get_random, int(sys.argv[2]), 128 * 12 * 4)

graph = frontend.from_onnx(sys.argv[1], config)

amap = {}
debug_print_graph(graph)

passes = [
    backend.place,
    backend.copy_insertion,
    backend.opt_graph_split,
    backend.allocate,
    backend.build_graph,
]

for i, opass in enumerate(passes):
    opass(graph, amap, config)

    print("---pass: {}---".format(opass.__name__))
    debug_print_graph(graph)
    nx.write_gml(graph, opass.__name__ + ".gml", node_stringizer)


start_time = datetime.datetime.now()
ptasks.spawn(placement=pcpu_cores.cpu(0))(backend.build_execute(graph, config))
end_time = datetime.datetime.now()


with open(sys.argv[3],"a+") as f:
    f.write("{},{}: {}\n".format(sys.argv[1], sys.argv[2], (end_time - start_time).total_seconds()))
