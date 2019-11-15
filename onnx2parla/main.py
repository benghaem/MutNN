import networkx as nx
import numpy as np
import logging

import backend
from node import Node, InOut
from config import Config

import parla
from parla import cpu as pcpu
from parla import tasks as ptasks

logging.basicConfig(filename="backend.log", level=logging.INFO)

graph = nx.DiGraph()

## add node adds x,y and places in z

# "static" --> array is defined statically here
# "pointer" --> look up location in map
# "dynamic" --> look up, but can be accessed from outside

node_load = Node(
            "load",
            {},
            {"Z": InOut("data", "pointer", None, (4))},
            {"batch_id": 0,
             "width": 4},
            0
            )

node1 = Node(
    "add",
    {
        "X": InOut("data", "pointer", None, (4)),
        "Y": InOut("op0_y", "static", np.array([1, 2, 3, 4]), (4)),
    },
    {"Z": InOut("op0_z", "pointer", None, (4))},
    {},
    0
)

node2 = Node(
    "add",
    {
        "X": InOut("op0_z", "pointer", None, (4)),
        "Y": InOut("op1_y", "static", np.array([1, 2, 3, 4]), (4)),
    },
    {"Z": InOut("op1_z", "pointer", None, (4))},
    {},
    0
)

node_store = Node(
            "store",
            {"X": InOut("op1_z", "pointer", None, (4))},
            {},
            {},
            0
            )

def debug_print_graph(graph):
    for node_id in graph.nodes:
        node = graph.nodes[node_id]["node"]
        print(node)


graph.add_node(0)
graph.nodes[0]["node"] = node_load

graph.add_node(1)
graph.nodes[1]["node"] = node1

graph.add_edge(0, 1) #load to 1

graph.add_node(2)
graph.nodes[2]["node"] = node2

graph.add_edge(1, 2) #1 -> 2

graph.add_node(3)
graph.nodes[3]["node"] = node_store

graph.add_edge(2, 3) #2 -> store

amap = {}

def echo_idx(start_idx, end_idx):
    data = [start_idx + i for i in range(end_idx - start_idx)]
    return np.array(data)

def echo_store(arr):
    logging.log(logging.INFO, f"stored: {arr}")

config = Config(echo_store, echo_idx, 4, 4*1000)

print(amap)
debug_print_graph(graph)

passes = [backend.place_n_opt,
          backend.allocate,
          backend.build_graph]

for i, opass in enumerate(passes):
    opass(graph, amap, config)

    print("---pass: {}---".format(opass.__name__))
    print(amap)
    debug_print_graph(graph)

# run everything!
ptasks.spawn(placement=pcpu.cpu(0))(backend.build_execute(graph, config))

