from multiprocessing import Process, Queue
from onnx2parla import onnx_frontend as frontend
from onnx2parla.config import Config
from onnx2parla.vision_dataloaders import get_random, nop_store
import numpy as np
import datetime

def frontend_proc(cmd_q, stats_q, **kwargs):

    run_proc = True
    stats_dict = {}
    stats_update_time = datetime.datetime.now()

    while(run_proc):
        inp = input("> ").lower().strip().split(" ")
        if (inp[0] in ["exit", "quit"]):
            cmd_q.put(("quit", None))
            run_proc = False
        elif (inp[0] in ["new_graph"]):
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
            print(stats_dict)


def pre_alloc_build():

    pre_mp_passes = [backend.shape_inference,
                     backend.place,
                     backend.copy_insertion]

    # do all pre_mp_passes 

    graph_req_size = backend.get_required_size()

    return graph, info


def alloc_rebuild(all_graphs):

    alloc_priority = determine_alloc_priority(graph_req_sizes)

    for graph in all_graphs:
        backend.alloc_from_pool()??
        backend.build_graph()




def backend_proc(cmd_q, stats_q, **kwargs):

    #control
    run_proc = True

    #stats
    stats = {"num_graphs":0}

    #state
    # graph_id : [(node, [deps ....]) ....]
    dep_graphs = {}
    # graph_id : [dependent on graph_ids]
    dep_info = {}

    graphs = []

    #MAKE BIG ALLOCATION


    while (run_proc):
        if (not cmq_q.empty()):
            (cmd, args) = cmd_q.get()
            if (cmd == "add"):
                graph, cfg = args
                stats["num_graphs"] += 1
                stats_q.put(stats)

                # run the passes on the new model
                multi_model.build(graph, cfg)

                multi_model.update_memory


            if (cmd == "quit"):
                run_proc = False

        else:



if __name__ == "__main__":
    cmd_q = Queue()
    stats_q = Queue()

    backend_p = Process(target=backend_proc, args=(cmd_q, stats_q))
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




