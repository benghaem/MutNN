import multiprocessing as mp
from onnx2parla import onnx_frontend as frontend
from onnx2parla.config import Config
from onnx2parla.vision_dataloaders import get_random, nop_store
import numpy as np
import datetime
from onnx2parla.memory import AllocMap
import onnx2parla.backend as backend
from onnx2parla.backend import debug_print_graph
from onnx2parla.node import node_stringizer
import networkx as nx
import time
import pprint

def frontend_proc(cmd_q, stats_q, **kwargs):

    run_proc = True
    stats_dict = {}
    stats_update_time = datetime.datetime.now()

    while(run_proc):
        inp = input("> ").lower().strip().split(" ")
        if (inp[0] in ["exit", "quit"]):
            cmd_q.put(("quit", None))
            run_proc = False

        elif (inp[0] in ["add"]):
            if (len(inp) < 3):
                inp = ["add", "test_models/one_stage_resnet.onnx", 2]
            # new_graph my_onnx_graph.onnx 2,3,4,5
            fname = inp[1]
            batch_size = int(inp[2])

            # potentially infinite dataset. We will run until told to stop
            cfg = Config(nop_store, get_random, batch_size, -1)
            cfg.use_data_para = True
            cfg.debug_passes = True

            graph = frontend.from_onnx(fname, cfg)
            print(graph.nodes[4]["node"])

            cmd_q.put(("add", (graph, cfg)))

            print(str(cmd_q.qsize()) + " cmds in queue")

        elif (inp[0] in ["stats"]):
            if (not stats_q.empty()):
                stats_dict = stats_q.get_nowait()
                stats_update_time = datetime.datetime.now()
            print("STATS @ " + str(stats_update_time))
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(stats_dict)


def build(graph, alloc_map, config):

    alloc_map.register_new_model(config.model_id)

    pre_populate = [backend.shape_inference,
                     backend.place,
                     backend.copy_insertion,
                     backend.register_statics,
                     backend.build_groups,
                     backend.register_groups]

    post_populate = [backend.populate_statics,
                     backend.build_graph]

    for gpass in pre_populate:
        gpass(graph, alloc_map, config)

        if config.debug_passes:
            print("---pass: {}---".format(gpass.__name__))
            debug_print_graph(graph)
            nx.write_gml(graph, gpass.__name__ + ".gml", node_stringizer)

    #manage memory across all graphs
    alloc_map.populate()

    for gpass in post_populate:
        gpass(graph, alloc_map, config)

        if config.debug_passes:
            print("---pass: {}---".format(gpass.__name__))
            debug_print_graph(graph)
            nx.write_gml(graph, gpass.__name__ + ".gml", node_stringizer)

    return graph



def backend_proc(cmd_q, stats_q, **kwargs):

    #IMPORTS TO MAKE MULTI-PROCESS CUPY WORK
    active_devices = backend.get_valid_cuda_devices()

    print("CUDA DEVICES: ", active_devices)

    #control
    run_proc = True

    #stats
    stats = {"num_graphs":0}

    #state
    graphs = []

    alloc_map = AllocMap(active_devices,
                         1024 * (10**6),
                         AllocMap.greedy_pack)

    while (run_proc):
        if (not cmd_q.empty()):
            (cmd, args) = cmd_q.get()
            if (cmd == "add"):
                graph, cfg = args
                stats["num_graphs"] += 1

                # run the passes on the new model
                cfg.model_id = stats["num_graphs"]
                graph = build(graph, alloc_map, cfg)

                stats["deps"] = alloc_map.workspace_deps
                stats["ws_size"] = alloc_map.get_workspace_size()
                stats["ws_ptrs"] = alloc_map.get_workspace_ptrs()
                stats["ws_stack"] = alloc_map.get_pack_strat().workspace_stack
                #stats["static_size"] = alloc_map.get_static_size()
                stats["free"] = alloc_map.get_free()
                stats["used"] = alloc_map.get_used()
                stats_q.put(stats)
            if (cmd == "quit"):
                run_proc = False


        else:
            time.sleep(1)



if __name__ == "__main__":
    mp.set_start_method('spawn')

    cmd_q = mp.Queue()
    stats_q = mp.Queue()

    backend_p = mp.Process(target=backend_proc, args=(cmd_q, stats_q))
    backend_p.start()

    frontend_proc(cmd_q, stats_q)

    backend_p.join()
    cmd_q.close()
    stats_q.close()

"""
first_process:

    while True:
        if there is new user input:
            graph = frontend.process(filename, config)
            queue.push(graph)

        new_stats = stats_queue.pull_non_blocking()
        integrate_stats(new_stats)


second process:


    tails = {
                model_id:[tail_batch_0,tail_batch_1]
            }

    graph_topos = {
                model_id:of node tuples (node, dependencies)
            }

    stream_to_model_map = {}

    while True:

        if (timer % something):
            #check for new graph
            new_graph = queue.pull_non_blocking()

            if (new_graph):
                build(new_graph) #does allocations (decides what space to use)
                update_dependency_tracker()
                add dummy tails
                generate graph topo


                create a task space
                for each node in the topo

                    #setup the node
                    build_wrapper 
                    #(to setup streams and make sure that return
                    #values are set correctly when tasks complete)
                    # each model has own stream

                    deps = []

                    # we must depend on:
                        # own model predecessors
                        # other models based on memory dep tracker

                    # append to the dependency
                    # a tuple representing the model and the node offset
                    # if it depends on the current or previous batch
                    deps.append((model,offset,prev_batch))


                graph_topos[??] = ....


        for each graph in priority order:
            active = graph.batch_is_ready()
            remove all completed tails

            if (active and len(tails[model_id] < MAX)):
                for node,deps in graph_topos[graph]:

                    compute actual dependency with the following

                    comp_deps = []
                    for model_id,offset,prev_batch in deps:
                        if prev_batch:
                            base = SUMLEN*(model_map[model_id] - 1)
                        else:
                            base = SUMLEN*(model_map[model_id])

                        comp_deps.append(base+offset)

                            

                    pick which queue
                    pick parla cpu core location

                    ptasks.spawn()

"""




