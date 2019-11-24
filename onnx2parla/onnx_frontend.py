import onnx
from onnx import numpy_helper
import onnx.utils

from node import InOut, Node
import operators as ops
from config import Config

import onnx_convert

import networkx as nx
import numpy as np

import logging


def onnx_type_to_shape(type_obj, batch_size):

    shape_dims = []
    for dim in type_obj.tensor_type.shape.dim:
        shape_dims.append(dim.dim_value)

    shape_dims[0] = batch_size
    return tuple(shape_dims)


def from_onnx(fname: str, config: Config) -> nx.DiGraph:

    # load onnx graph into memory
    model = onnx.load_model(fname)

    # check optimize and infer shapes
    polished_model = onnx.utils.polish_model(model)

    initializers = {}
    value_info = {}
    io_map = {}

    # this will capture all of the graph intializerexport_params (bool, default True) – if specified, all parameters will be exported. Set this to False if you want to export an untrained model. In this case, the exported model will first take all of its parameters as arguments, the ordering as specified by model.state_dict().values()s
    for init in polished_model.graph.initializer:
        initializers[init.name] = init
        logging.log(logging.DEBUG, f"Registered initializer: {init.name}")

    # this captures all internal values, but not the graph output for some
    # reason (onnx spec is strange)
    for vi in polished_model.graph.value_info:
        value_info[vi.name] = vi
        logging.log(logging.DEBUG, f"Registered value info: {vi.name}")

    # this captures the graph output
    for vi in polished_model.graph.output:
        value_info[vi.name] = vi
        logging.log(logging.DEBUG, f"Registered value info: {vi.name} (output)")

    # this captures all model inputs
    for inp in polished_model.graph.input:
        new_io = InOut(inp.name, None, None, None)

        # directly convert onnx initializers to static IOs in the graph
        if inp.name in initializers:
            new_io.kind = "static"
            new_io.data = numpy_helper.to_array(initializers[inp.name])
            new_io.shape = np.shape(new_io.data)

        # pointers will be allocated later by the allocate pass
        else:
            new_io.kind = "pointer"
            new_io.data = None
            new_io.shape = onnx_type_to_shape(inp.type,
                                              config.batch_size)

        io_map[inp.name] = new_io
        logging.log(logging.DEBUG, f"Built IO: {new_io}")

    # Create IOs for all node outputs
    for node in polished_model.graph.node:
        for out in node.output:
            new_io = InOut(out, None, None, None)
            new_io.kind = "pointer"
            new_io.data = None
            new_io.shape = onnx_type_to_shape(value_info[out].type,
                                              config.batch_size)
            io_map[out] = new_io
            logging.log(logging.DEBUG, f"Built IO: {new_io}")

    # at this point all inputs and outputs are availiable
    graph = nx.DiGraph()

    # usage map holds the uses of all of the _pointer_ IOs in the graph
    # eg IO : {use = [node2, node3], def = [node1]}
    # pointer IOs represent graph edges

    usage_map = {}
    for io_name, io_v in io_map.items():
        if io_v.kind == "pointer":
            usage_map[io_name] = {"use": [], "def": []}

    # start numbering nodes at zero
    node_id = 0

    # attach a load node for each of the dynamic inputs
    for dyninp_vi in polished_model.graph.input:
        if dyninp_vi.name not in initializers:
            built_node = build_load_node(dyninp_vi.name, io_map, usage_map,
                                         node_id, config.batch_size)
            graph.add_node(node_id)
            graph.nodes[node_id]["node"] = built_node

            logging.log(logging.DEBUG, f"Built node: {built_node}")

            node_id += 1

    # build normal nodes here
    for onnx_node in polished_model.graph.node:
        built_node = build_node(onnx_node, io_map, usage_map, node_id)
        graph.add_node(node_id)

        graph.nodes[node_id]["node"] = built_node
        logging.log(logging.DEBUG, f"Built node: {built_node}")

        node_id += 1

    # attach a store node for each of the model outputs
    for out_vi in polished_model.graph.output:
        built_node = build_store_node(out_vi.name, io_map, usage_map, node_id)
        graph.add_node(node_id)
        graph.nodes[node_id]["node"] = built_node

        logging.log(logging.DEBUG, f"Built node: {built_node}")

        node_id += 1

    # we don't know the iteration order so we build edges here
    # by checking the usage map
    for name, info in usage_map.items():

        defs = info["def"]
        if len(defs) > 1:
            logging.log(logging.ERROR, f"Multiple defn of {name} at {defs}")
        if len(defs) == 0:
            logging.log(logging.ERROR, f"{name} never defined")

        source = info["def"][0]
        for use in info["use"]:
            graph.add_edge(source, use, buffer=name)
            logging.log(logging.DEBUG, f"Added edge {source} -> {use} via {name}")

    # we sanity check that there are no nodes that have not been connected to
    # the graph
    num_wcc = nx.number_weakly_connected_components(graph)
    if num_wcc > 1:
        wcc = nx.weakly_connected_components(graph)
        logging.log(logging.WARN, "Multiple components in ouput graph")
        for i, wc in enumerate(wcc):
            logging.log(logging.WARN, f"\t<{i}> {wc}")

    return graph


def build_node(onnx_node, io_map, usage_map, node_id):

    input_names = onnx_convert.get_op_input_info(onnx_node.op_type)
    output_names = onnx_convert.get_op_output_info(onnx_node.op_type)

    inputs = {}
    for i, inp in enumerate(onnx_node.input):
        inputs[input_names[i]] = io_map[inp]

        # don't add static allocations to the usage map
        if io_map[inp].kind == "pointer":
            usage_map[inp]["use"].append(node_id)

    outputs = {}
    for i, out in enumerate(onnx_node.output):
        outputs[output_names[i]] = io_map[out]
        usage_map[out]["def"].append(node_id)

    attrs = {}
    for attr in onnx_node.attribute:
        attrs[attr.name] = onnx_convert.convert_attr(attr)

    new_node = Node(node_id, onnx_node.op_type, inputs, outputs, attrs, 0)

    return new_node


def build_store_node(target, io_map, usage_map, node_id):
    inputs = {"X": io_map[target]}
    outputs = {}
    attrs = {}
    new_node = Node(node_id, ops.O2P_STORE, inputs, outputs, attrs, 0)
    usage_map[target]["use"].append(node_id)

    return new_node


def build_load_node(target, io_map, usage_map, node_id, batch_size):
    inputs = {}
    outputs = {"Z": io_map[target]}
    attrs = {"width": batch_size,
             "batch_id": 0}
    new_node = Node(node_id, ops.O2P_LOAD, inputs, outputs, attrs, 0)
    usage_map[target]["def"].append(node_id)

    return new_node


if __name__ == "__main__":
    import sys

    logging.basicConfig(filename="onnx_frontend.log", level=logging.DEBUG)

    config = Config(None, None, 4, 4)

    g = from_onnx("../example.onnx", config)

    nx.write_gml(g, "frontend.gml", str)

    print("--------")
    for gnode in g.nodes:
        g.nodes[gnode]["node"].pretty_print()
