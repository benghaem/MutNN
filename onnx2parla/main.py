import networkx as nx
import frontend
import backend
import numpy as np
import logging
from node import Node, InOut

import parla
from parla import cpu as pcpu
from parla import tasks as ptasks

logging.basicConfig(filename="backend.log",level=logging.DEBUG)

graph = nx.DiGraph()

## add node adds x,y and places in z

# "static" --> array is defined statically here
# "pointer" --> look up location in map
# "dynamic" --> look up, but can be accessed from outside


node0 = Node("add",
            {"X": InOut("op0_x",
                        "dynamic",
                        None,
                        (4)),
            "Y": InOut("op0_y",
                        "static",
                        np.array([1,2,3,4]),
                        (4))
            },
            {"Z": InOut("op0_z",
                        "pointer",
                        None,
                        (4))
            },
            {})

node1 = Node("add",
            {"X": InOut("op0_z",
                        "pointer",
                        None,
                        (4)),
            "Y": InOut("op1_y",
                        "static",
                        np.array([1,2,3,4]),
                        (4))
            },
            {"Z": InOut("op1_z",
                        "pointer",
                        None,
                        (4))
            },
            {})

def debug_print_graph(graph):
    for node_id in graph.nodes:
        node = graph.nodes[node_id]['node']
        print(node)

graph.add_node(0)
graph.nodes[0]['node'] = node0
graph.add_node(1)
graph.nodes[1]['node'] = node1
graph.add_edge(0,1)


amap = {}

print(amap)
debug_print_graph(graph)

passes = [backend.allocate,
          backend.place_n_opt,
          backend.build_graph]

for i, opass in enumerate(passes):
    opass(graph, amap)

    print("---pass: {}---".format(opass.__name__))
    print(amap)
    debug_print_graph(graph)


# push in input

for i in range(10):
    inp = np.random.random(4)

    np.copyto(amap['op0_x'],inp)

    ptasks.spawn(placement=pcpu.cpu(0))(backend.build_execute(graph))
    # print output
    print("------------")
    print("INPUT:",inp)
    print("OUTPUT:",amap['op1_z'])


