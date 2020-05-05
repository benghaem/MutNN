import numpy as np
import networkx as nx
import math

import logging
from typing import Dict, Callable

# from parla import cpu as pcpu
from parla import cpucores as pcpu_cores
from parla import cuda as pcuda
from parla import tasks as ptasks

import cupy

from onnx2parla.node import Node, InOut
import onnx2parla.kernels as kernels
from onnx2parla.config import Config
import onnx2parla.operators as ops
from onnx2parla.onnx_shape_inference import infer_shape
from collections import deque
from onnx2parla.memory import size_in_bytes


PNO_GRAPH_HEAD_ID = -1


def get_valid_cuda_devices():
    valid_ids = []
    for d_id in range(32):
        try:
            cupy.cuda.Device(d_id).compute_capability
        except cupy.cuda.runtime.CUDARuntimeError:
            continue

        valid_ids.append(d_id)

    return valid_ids


def debug_print_graph(graph):
    for node_id in graph.nodes:
        node = graph.nodes[node_id]["node"]
        node.pretty_print()


def build_copy_node(in_io, out_io, node_id):
    inputs = {"X": in_io}
    outputs = {"Z": out_io}
    attrs = {}
    new_node = Node(node_id, ops.O2P_COPY, inputs, outputs, attrs, 0)

    return new_node


def build_groups(graph: nx.DiGraph, alloc_map, config: Config) -> None:

    roots = graph.successors(PNO_GRAPH_HEAD_ID)
    q = deque(roots)

    # (from,to,group)
    copy_to_insert = []

    while len(q) > 0:
        gnode = q.popleft()
        node = graph.nodes[gnode]["node"]

        gparents = graph.predecessors(gnode)
        gchildren = graph.successors(gnode)

        # generate the correct group for node
        # based on parent groups
        parents_not_ready = False
        parent_group = set()
        even_parents = []
        odd_parents = []
        for gparent in gparents:
            parent = graph.nodes[gparent]["node"]
            if gparent == PNO_GRAPH_HEAD_ID:
                parent_group.add(-1)
                odd_parents.append(PNO_GRAPH_HEAD_ID)
            else:
                parent_group.add(parent.group)
                if parent.group == None:
                    parents_not_ready = True
                    break
                if parent.group % 2 == 0:
                    even_parents.append(gparent)
                else:
                    odd_parents.append(gparent)

        if parents_not_ready:
            q.append(gnode)
            continue

        # assign new group
        node.group = max(parent_group) + 1

        # check if we need to do copy insertion
        num_even_parents = len(even_parents)
        num_odd_parents = len(odd_parents)
        if num_even_parents > 0 and num_odd_parents > 0:
            # pick the one with fewer parents of that type
            target_parents = None
            # TODO: Equal case can be special
            if num_even_parents < num_odd_parents:
                target_parents = even_parents
            else:
                target_parents = odd_parents

            for gparent in target_parents:
                parent = graph.nodes[gparent]["node"]
                copy_to_insert.append((gparent, gnode, parent.group + 1))

        for gchild in gchildren:
            if gchild not in q:
                q.append(gchild)

    # do the copy insertion

    node_id = graph.number_of_nodes()

    for copy_info in copy_to_insert:
        gparent, gnode, copy_group = copy_info
        node = graph.nodes[gparent]["node"]

        for output_io in node.outputs.values():
            source_buffer = output_io.name
            target_buffer = source_buffer + "_groupcpy"
            copy_in_io = InOut(source_buffer, "pointer", None, output_io.shape)
            copy_out_io = InOut(target_buffer, "pointer", None, output_io.shape)

            new_copy_node = build_copy_node(copy_in_io, copy_out_io, node_id)

            new_copy_node.device_type = node.device_type
            new_copy_node.device_id = node.device_id
            new_copy_node.set_attr("target_device", node.get_device())
            new_copy_node.set_attr("source_device", node.get_device())
            new_copy_node.group = copy_group

            graph.add_node(node_id)
            graph.nodes[node_id]["node"] = new_copy_node
            graph.add_edge(gparent, node_id, buffer=source_buffer)

            child = graph.nodes[gnode]["node"]
            child.replace_io_for_input_buffer(source_buffer, copy_out_io)

            graph.add_edge(node_id, gnode, buffer=target_buffer)
            graph.remove_edge(gparent, gnode)

            node_id += 1

    return


def register_statics(graph: nx.DiGraph, alloc_map, config: Config) -> None:

    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]

        for io in node.inputs.values():
            if io.kind == "static":
                if node.device_type == "gpu":
                    alloc_map.register_gpu_static(
                        config.model_id, node.device_id, io.name, io.shape, io.dtype
                    )
                else:
                    alloc_map.register_cpu(config.model_id, io.name, io.shape, io.dtype)


def populate_statics(graph: nx.DiGraph, alloc_map, config: Config) -> None:
    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]

        for io in node.inputs.values():
            if io.kind == "static":
                if node.device_type == "gpu":
                    device_id = node.device_id
                    target = alloc_map.get_gpu_static_array(
                        config.model_id, device_id, io.name
                    )

                    with cupy.cuda.Device(device_id):
                        #need to cast to cupy array here
                        cupy.copyto(target, cupy.asarray(io.data))
                else:
                    target = alloc_map.get_cpu_array(config.model_id, io.name)
                    np.copyto(target, io.data)


def register_groups(graph: nx.DiGraph, alloc_map, config: Config) -> None:

    # collect the groups
    group_info = {}
    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]
        device = node.get_device()

        # don't register the graph head
        if gnode == PNO_GRAPH_HEAD_ID:
            continue

        # copy operators have an annoying property where they may actually
        # output to another device. We catch this here and reassign them
        # so that we correct output space requirement info
        if node.operator == "copy":
            device = node.get_attr("target_device")

        if device not in group_info:
            group_info[device] = {}
        if node.group not in group_info[device]:
            group_info[device][node.group] = []
        group_info[device][node.group].append(gnode)

    # find the requirement for the even and odd groups
    even_group_req = {}
    odd_group_req = {}

    for device in group_info.keys():
        if device not in even_group_req:
            even_group_req[device] = 0
        if device not in odd_group_req:
            odd_group_req[device] = 0

        for group, gnodes in group_info[device].items():
            group_sum = 0
            for gnode in gnodes:
                node = graph.nodes[gnode]["node"]
                for output_io in node.outputs.values():
                    group_sum += size_in_bytes(output_io.shape, output_io.dtype)
            if (group % 2) == 0:
                even_group_req[device] = max(even_group_req[device], group_sum)
            else:
                odd_group_req[device] = max(odd_group_req[device], group_sum)

    # allocate a workspace for even and odd gpu groups
    for device in group_info.keys():
        device_type, device_id = device

        if device_type == "gpu":
            alloc_map.register_gpu_workspace(
                config.model_id, device_id, "A", even_group_req[device]
            )
            alloc_map.register_gpu_workspace(
                config.model_id, device_id, "B", odd_group_req[device]
            )

    # device, workspace, alloc_name, shape, dtype
    for device in group_info.keys():
        device_type, device_id = device
        for group, gnodes in group_info[device].items():
            ios_in_group = []
            for gnode in gnodes:
                node = graph.nodes[gnode]["node"]
                for output_io in node.outputs.values():
                    ios_in_group.append(output_io)

            if device_type == "gpu":
                ws_name = ""
                if (group % 2) == 0:
                    ws_name = "A"
                else:
                    ws_name = "B"

                alloc_map.register_gpu_ws_group(
                    config.model_id, device_id, ws_name, ios_in_group
                )
            else:
                alloc_map.register_cpu_group(config.model_id, ios_in_group)

    return


def shape_inference(graph: nx.DiGraph, alloc_map, config: Config) -> None:

    # ensure that input shapes are available
    fixed_topo = list(nx.topological_sort(graph))

    for gnode in fixed_topo:
        node = graph.nodes[gnode]["node"]

        infer_shape(node)

    return


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

    # copy iterator
    fixed_node_set = list(graph.nodes())

    # don't apply to PNO HEAD
    if PNO_GRAPH_HEAD_ID in fixed_node_set:
        fixed_node_set.remove(PNO_GRAPH_HEAD_ID)

    for gnode in fixed_node_set:
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
                    buffer_on_device = "{}_cpy2{}".format(active_buffer, device[0])
                    copy_in_io = InOut(active_buffer, "pointer", None, output_io.shape)

                    copy_out_io = InOut(
                        buffer_on_device, "pointer", None, output_io.shape
                    )

                    new_copy_node = build_copy_node(copy_in_io, copy_out_io, node_id)
                    new_copy_node.device_type = device[0]
                    new_copy_node.device_id = device[1]
                    new_copy_node.set_attr("target_device", device)
                    new_copy_node.set_attr("source_device", node.get_device())

                    graph.add_node(node_id)
                    graph.nodes[node_id]["node"] = new_copy_node
                    graph.add_edge(gnode, node_id, buffer=active_buffer)

                    for gchild in gchildren:
                        child = graph.nodes[gchild]["node"]
                        child.replace_io_for_input_buffer(active_buffer, copy_out_io)

                        graph.add_edge(node_id, gchild, buffer=buffer_on_device)
                        graph.remove_edge(gnode, gchild)

                    node_id += 1


def place(graph: nx.DiGraph, alloc_map: Dict[str, np.ndarray], config: Config) -> None:

    gpu_supported = [
        ops.CONV,
        ops.MAXPOOL,
        ops.ADD,
        ops.RELU,
        ops.BATCH_NORM,
        ops.FLATTEN,
        ops.RESHAPE,  # ONNX reshape has implied copies
        ops.GLOBALAVERAGEPOOL,
        ops.AVERAGE_POOL,
        ops.PAD,
        ops.GEMM,
        ops.DROPOUT,
        ops.CLIP,
        ops.REDUCE_MEAN,
    ]

    cuda_devices = get_valid_cuda_devices()
    num_cuda = len(cuda_devices)

    # add the generic head node to the graph
    # connect it to the initial root node generated by frontend
    graph.add_node(PNO_GRAPH_HEAD_ID)
    graph.add_edge(PNO_GRAPH_HEAD_ID, 0)

    graph.nodes[PNO_GRAPH_HEAD_ID]["node"] = Node(-1, ops.O2P_GRAPH_HEAD, {}, {}, {}, 0)

    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]
        if num_cuda > 0 and node.operator in gpu_supported:
            node.device_type = "gpu"
            node.device_id = 0
        else:
            node.device_type = "cpu"
            node.device_id = 0

    return


def opt_simple_model_para(graph: nx.DiGraph, alloc_map, config: Config) -> None:

    graph.add_node(PNO_GRAPH_HEAD_ID)
    graph.add_edge(PNO_GRAPH_HEAD_ID, 0)

    graph.nodes[PNO_GRAPH_HEAD_ID]["node"] = Node(-1, ops.O2P_GRAPH_HEAD, {}, {}, {}, 0)

    cuda_devices = get_valid_cuda_devices()
    num_cuda = len(cuda_devices)

    total_nodes = graph.number_of_nodes()

    # HACK
    split_len = total_nodes // 5

    for gnode in graph.nodes:
        node = graph.nodes[gnode]["node"]
        if node.device_type == "gpu":
            node.device_id = node.node_id // split_len
            # HACKY SAFETY
            if node.device_id > num_cuda - 1:
                node.device_id = num_cuda - 1

    return


def build_replicated_io(inp_io, suffix):
    new_io = InOut(inp_io.name + suffix, inp_io.kind, None, inp_io.shape)

    # copy over static allocation if needed
    if new_io.kind == "static":
        new_io.data = np.ndarray(new_io.shape)
        np.copyto(new_io.data, inp_io.data)

    return new_io


def build_replicated_node(other, node_id, instance_id, suffix):

    inputs = {}
    for inp_name, inp_io in other.inputs.items():
        inputs[inp_name] = build_replicated_io(inp_io, suffix)

    outputs = {}
    for outp_name, outp_io in other.outputs.items():
        outputs[outp_name] = build_replicated_io(outp_io, suffix)

    attrs = {}
    for attr_name, attr_v in other.attrs.items():
        attrs[attr_name] = attr_v

    new_node = Node(node_id, other.operator, inputs, outputs, attrs, instance_id)

    return new_node


def opt_graph_split(
    graph: nx.DiGraph, alloc_map: Dict[str, np.ndarray], config: Config
) -> None:


    # need to rename and assign to the correct device
    cuda_devices = get_valid_cuda_devices()
    num_cuda = len(cuda_devices)

    config.computed_batch_size = num_cuda * config.user_width
    # there is now +1 node in the graph because of the -1 head
    new_gnode = graph.number_of_nodes() - 1

    if num_cuda > 0:

        # compute the correct split

        # gpu_name_maps = [{}] * num_cuda
        gpu_name_maps = [{} for i in range(num_cuda)]

        # source_gnode -> local_gnode

        # add a mapping from og graph head to graph head for all devices
        for i in range(num_cuda):
            gpu_name_maps[i][PNO_GRAPH_HEAD_ID] = PNO_GRAPH_HEAD_ID

        # start at the initial node of the non-replicated graph
        fixed_list = list(nx.topological_sort(graph))

        # skip the HEAD node
        for source_gnode in fixed_list[1:]:
            source_node = graph.nodes[source_gnode]["node"]

            gparents = list(graph.predecessors(source_gnode))

            for gpu_idx, device_id in enumerate(cuda_devices):

                device_node = build_replicated_node(
                    source_node, new_gnode, gpu_idx, f"_g{device_id}"
                )

                # configure device settings for the new node
                if source_node.device_type == "gpu":
                    device_node.device_type = "gpu"
                    device_node.device_id = device_id
                else:
                    device_node.device_type = "cpu"
                    device_node.device_id = 0

                graph.add_node(new_gnode)
                graph.nodes[new_gnode]["node"] = device_node

                # look up source node parent in gpu_name_maps
                for gparent in gparents:
                    edge_source = gpu_name_maps[gpu_idx][gparent]
                    graph.add_edge(edge_source, new_gnode)

                # add ourself to the gpu name map
                gpu_name_maps[gpu_idx][source_gnode] = new_gnode

                new_gnode += 1

        # remove the og graph
        for gnode in fixed_list[1:]:
            graph.remove_node(gnode)

    return


def opt_shape(**args):

    # support hetrogenous gpus?

    # we need a function to fix the IO widths. This is an issue if the
    # user_width is different than the optimal / maximum width for our GPUs

    # ensure that config.computed_batch_size = sum(all widths)
    # computed batch size is invalid until this pass completes

    # two cases

    # maximum = 64
    # user_width = 80
    # user_width > maximum
    #   WARN user?
    # user_width <= maximum
    #   do nothing

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
                    alloc_map[io_z.name] = cupy.ndarray(io_z.shape, dtype=cupy.float32)
            else:
                alloc_map[io_z.name] = np.ndarray(io_z.shape, dtype=np.float32)
        else:
            for io in node.outputs.values():
                if io.kind == "pointer":
                    if node.device_type == "gpu":
                        with cupy.cuda.Device(node.device_id):
                            alloc_map[io.name] = cupy.ndarray(
                                io.shape, dtype=cupy.float32
                            )
                    else:
                        alloc_map[io.name] = np.ndarray(io.shape, dtype=np.float32)

        for io in node.inputs.values():
            if io.kind == "static" and node.device_type == "gpu":
                with cupy.cuda.Device(node.device_id):
                    io.data = cupy.asarray(io.data, dtype=cupy.float32)
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

    if oper == ops.AVERAGE_POOL:
        if node.device_type == "cpu":
            return kernels.average_pool_cpu(node, alloc_map, config)
        else:
            return kernels.average_pool_gpu(node, alloc_map, config)

    if oper == ops.PAD:
        if node.device_type == "cpu":
            return kernels.pad_cpu(node, alloc_map, config)
        else:
            raise NotImplementedError()

    if oper == ops.FLATTEN:
        if node.device_type == "cpu":
            return kernels.flatten_cpu(node, alloc_map, config)
        else:
            return kernels.flatten_gpu(node, alloc_map, config)

    if oper == ops.RESHAPE:
        if node.device_type == "cpu":
            return kernels.reshape_cpu(node, alloc_map, config)
        else:
            return kernels.reshape_gpu(node, alloc_map, config)

    if oper == ops.GEMM:
        if node.device_type == "cpu":
            return kernels.gemm_cpu(node, alloc_map, config)
        else:
            return kernels.gemm_gpu(node, alloc_map, config)

    if oper == ops.DROPOUT:
        if node.device_type == "cpu":
            return kernels.dropout_cpu(node, alloc_map, config)
        else:
            return kernels.dropout_gpu(node, alloc_map, config)

    if oper == ops.CLIP:
        if node.device_type == "cpu":
            return kernels.clip_v6_cpu(node, alloc_map, config)
        else:
            return kernels.clip_v6_gpu(node, alloc_map, config)

    if oper == ops.REDUCE_MEAN:
        if node.device_type == "cpu":
            return kernels.reduce_mean_cpu(node, alloc_map, config)
        else:
            return kernels.reduce_mean_gpu(node, alloc_map, config)

    if oper == ops.O2P_GRAPH_HEAD:
        return None

    raise ValueError(f"Operator {oper} not supported")


def build_execute(graph: nx.DiGraph, config: Config) -> Callable[[], None]:

    num_gpus = len(get_valid_cuda_devices())

    async def fn():
        async with ptasks.finish():

            task_obj_map = {}

            batches = math.ceil(config.dataset_len // config.computed_batch_size)
            roots = list(graph.successors(PNO_GRAPH_HEAD_ID))

            print("all roots {}".format(roots))
            print(batches)

            num_streams = 4
            streams = []

            for i in range(num_streams):
                streams.append(cupy.cuda.Stream(non_blocking=False))

            for batch_id in range(batches):

                # initialize with roots of tree
                q = deque(roots)

                while len(q) > 0:
                    gnode = q.popleft()

                    node = graph.nodes[gnode]["node"]

                    deps = []
                    parents_have_launched = True

                    # get parents and get children
                    for gparent in graph.predecessors(gnode):

                        # the graph root is not a real dependency
                        if gparent == PNO_GRAPH_HEAD_ID:
                            continue

                        parent = graph.nodes[gparent]["node"]

                        if parent.last_launch_batch_id != batch_id:
                            # need to wait until all deps are launched
                            # break out early
                            parents_have_launched = False
                            break

                        lto = parent.last_task_obj
                        deps.append(lto)
                        assert lto

                    if not parents_have_launched:
                        q.append(gnode)
                        continue

                    for gchild in graph.successors(gnode):
                        # depend on children from previous batch, ignore none
                        # for first batch
                        if batch_id > 0:
                            lto = graph.nodes[gchild]["node"].last_task_obj
                            deps.append(lto)
                            assert lto

                        # continue to progress down the tree
                        if gchild not in q:
                            q.append(gchild)

                    queue = (batch_id % (num_gpus)) + 1
                    loc = pcpu_cores.cpu(queue)

                    node.streams = streams
                    node.build_wrapper()

                    node.last_task_obj = ptasks.spawn(placement=loc, dependencies=deps)(
                        node.wrapper_fn
                    )

                    node.last_launch_batch_id += 1

                    logging.log(
                        logging.INFO, "---<{}>{}---".format(node.node_id, node.operator)
                    )
                    logging.log(logging.INFO, "launched {}".format(node.last_task_obj))
                    task_obj_map[
                        node.last_task_obj
                    ] = f"<{node.node_id}>{node.operator} batch={batch_id}"
                    dep_string = ""
                    for dep in deps:
                        dep_string = dep_string + " " + task_obj_map[dep]
                    logging.log(logging.INFO, "deps {}".format(dep_string))

                # if batch_id % 2 == 0:
                #    await node.last_task_obj

    return fn
