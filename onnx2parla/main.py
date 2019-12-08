import sys
import numpy as np
import logging
import networkx as nx

import onnx2parla.backend as backend
import onnx2parla.onnx_frontend as frontend
from onnx2parla.config import Config
from onnx2parla.node import node_stringizer
import datetime

import onnx2parla.vision_dataloaders as vision_dataloaders

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

def build(onnx_path, config):
    graph = frontend.from_onnx(onnx_path, config)

    amap = {}
    if config.debug_passes:
        debug_print_graph(graph)

    setup_passes = [backend.shape_inference,
                    backend.place]

    final_passes = [backend.allocate,
                    backend.build_graph]

    if config.use_data_para:
        opt_passes = [
            backend.copy_insertion,
            backend.opt_graph_split,
        ]

    if config.use_simple_model_para:
        opt_passes = [
                backend.opt_simple_model_para,
                backend.copy_insertion,
                ]

    passes = setup_passes + opt_passes + final_passes

    for opass in passes:
        opass(graph, amap, config)

        if config.debug_passes:
            print("---pass: {}---".format(opass.__name__))
            debug_print_graph(graph)
            nx.write_gml(graph, opass.__name__ + ".gml", node_stringizer)

    return Model(graph, config)

class Model:
    def __init__(self, graph, config):
        self.graph = graph
        self.config = config

    def run(self):
        ptasks.spawn(placement=pcpu_cores.cpu(0))(backend.build_execute(self.graph, self.config))

if __name__ == "__main__":
    config = Config(
        vision_dataloaders.echo_top5,
        vision_dataloaders.get_test,
        int(sys.argv[2]),
        int(sys.argv[3]),
    )
    config.debug_passes = True
    config.use_simple_model_para = True
    config.use_data_para = False

    o2p_model = build(sys.argv[1], config)

    st = datetime.datetime.now()
    o2p_model.run()
    end = datetime.datetime.now()

    time = end-st

    with open(sys.argv[4], "a+") as f:
        f.write("{},{}: {}\n".format(sys.argv[1], sys.argv[2], time.total_seconds()))
