import cupy
import numpy as np
from cupy.cuda import MemoryPointer, Memory
from operator import itemgetter
from functools import reduce

# memory pointer
# MemoryPointer(Memory, index)
# we grab a new buffer
# then we can produce new pointers by combining buffer and index


def size_in_bytes(shape, dtype):
    dtype_bytes = cupy.dtype(dtype).itemsize
    items = 1
    for ax in shape:
        items *= ax

    return items * dtype_bytes


class CupyMemPool:

    """
        workspace_pointers grow down from size
        static pointers grow up from zero

        workspace pointers always point to used space. Free space starts one
        index lower

        static pointers always point to free space. Used space starts one index
        lower.
    """

    def __init__(self, device_ids, size):
        self.size = size
        self.mem = {}
        self.static_pointers = {}
        self.workspace_pointers = {}

        for device_id in device_ids:
            with cupy.cuda.Device(device_id):
                self.mem[device_id] = Memory(self.size)
                self.static_pointers[device_id] = 0
                self.workspace_pointers[device_id] = self.size

    def memory_size(self, device_id):
        return self.size

    def memory_available(self, device_id):
        return self.workspace_pointers[device_id] - self.static_pointers[device_id]

    # alloc statics from 0
    def alloc_static(self, device_id, num_bytes):
        base_mem = self.mem[device_id]
        static_ptr = self.static_pointers[device_id]
        workspace_ptr = self.workspace_pointers[device_id]

        if static_ptr + num_bytes <= workspace_ptr:
            self.static_pointers[device_id] = static_ptr + num_bytes
            return MemoryPointer(base_mem, static_ptr)
        else:
            return None

    # alloc workspaces from size-1
    def alloc_workspace(self, device_id, num_bytes):
        base_mem = self.mem[device_id]
        workspace_ptr = self.workspace_pointers[device_id]
        static_ptr = self.static_pointers[device_id]

        if workspace_ptr - num_bytes >= static_ptr:
            self.workspace_pointers[device_id] = workspace_ptr - num_bytes
            return MemoryPointer(base_mem, workspace_ptr - num_bytes)
        else:
            return None

    def free_workspace(self, device_id):
        self.workspace_pointers[device_id] = self.size


class AllocInterval:

    # these intervals are [s,e)

    def __init__(self, model_id, name, start, end):
        self.model_id = model_id
        self.name = name
        self.start = start
        self.end = end

    def overlaps(self, other):
        return self.start <= (other.end - 1) and (self.end - 1) >= other.start

    def __str__(self):
        return f"[{self.start}, {self.end}):{self.model_id}:{self.name}"

    def __repr__(self):
        return str(self)


class AllocStrat:
    def __init__(self, ws_bytes, static_stack, workspace_stack):
        self.ws_bytes = ws_bytes
        self.static_stack = static_stack
        self.workspace_stack = workspace_stack


class AllocMap:
    def __init__(self, device_ids, size, pack_fn):
        self.models_ = []

        self.statics_ = {}
        self.workspaces_ = {}
        self.workspace_views_ = {}

        self.cpu_arrays_ = {}

        self.workspace_deps = {}

        self.last_pack_strat_ = None

        self.pack_fn = pack_fn
        self.pool_ = CupyMemPool(device_ids, size)
        self.device_ids_ = device_ids

        for dev_id in self.device_ids_:
            self.workspace_deps[dev_id] = {}

    """
    requested_workspaces = [("name", size), .....]
    request_statics = [("name",size), ......]

    sum up the sizes
    assert statics < total space
    #TODO: FIX THIS CASE IF WE HAVE TIME, CAN BE PROBLEMATIC HEY EMBEDDINGS :D

    run packing heuristic --> returns a set of allocation instructions

    execute these instructions
    """

    def get_workspace_ptrs(self):

        ptrs = {}
        for device_id in self.device_ids_:
            ptrs[device_id] = {}
            for model_id in self.models_:
                if device_id in self.workspaces_[model_id]:
                    ptrs[device_id][model_id] = {}
                    for ws_name, ws_info in self.workspaces_[model_id][
                        device_id
                    ].items():
                        ptrs[device_id][model_id][ws_name] = ws_info["ptr"].ptr

        return ptrs


    def get_workspace_size(self):

        sizes = {}
        for device_id in self.device_ids_:
            sizes[device_id] = {}
            for model_id in self.models_:
                if device_id in self.workspaces_[model_id]:
                    sizes[device_id][model_id] = {}
                    for ws_name, ws_info in self.workspaces_[model_id][
                        device_id
                    ].items():
                        sizes[device_id][model_id][ws_name] = ws_info["size"]

        return sizes

    def get_static_size(self):

        sizes = {}
        for device_id in self.device_ids_:
            sizes[device_id] = {}
            for model_id in self.models_:
                if device_id in self.statics_[model_id]:
                    sizes[device_id][model_id] = {}
                    for static_name, static_info in self.statics_[model_id][
                        device_id
                    ].items():
                        sizes[device_id][model_id][static_name] = static_info["size"]

        return sizes

    def get_free(self):
        free = {}
        for device_id in self.device_ids_:
            free[device_id] = self.pool_.memory_available(device_id)

        return free


    def get_used(self):
        used = {}
        for device_id in self.device_ids_:
            used[device_id] = self.pool_.memory_size(device_id) - self.pool_.memory_available(device_id)

        return used

    def get_pack_strat(self):
        return self.last_pack_strat_


    def register_new_model(self, model_id):
        if model_id not in self.models_:

            self.models_.append(model_id)
            self.workspaces_[model_id] = {}
            self.statics_[model_id] = {}
            self.workspace_views_[model_id] = {}
            self.cpu_arrays_[model_id] = {}

            for dev_id in self.device_ids_:
                self.workspaces_[model_id][dev_id] = {}
                self.workspace_views_[model_id][dev_id] = {}
                self.statics_[model_id][dev_id] = {}

            return True
        return False

    # register either A or B
    def register_gpu_workspace(self, model_id, device_id, name, size):
        if name not in self.workspaces_[model_id][device_id]:
            self.workspaces_[model_id][device_id][name] = {"size": size, "ptr": None}
            return True
        return False

    def register_gpu_static(self, model_id, device_id, name, shape, dtype):
        if name not in self.statics_:
            self.statics_[model_id][device_id][name] = {
                "size": size_in_bytes(shape, dtype),
                "shape": shape,
                "dtype": dtype,
                "ptr": None,
            }
            return True
        return False

    def register_gpu_ws_group(self, model_id, device_id, ws_name, group_ios):
        offset = 0
        for io in group_ios:
            print(self.workspace_views_)
            self.workspace_views_[model_id][device_id][io.name] = (
                ws_name,
                offset,
                io.dtype,
                io.shape,
            )
            offset += size_in_bytes(io.shape, io.dtype)

    def register_cpu_group(self, model_id, group_ios):
        for io in group_ios:
            self.cpu_arrays_[model_id][io.name] = np.ndarray(io.shape, io.dtype)

    def register_cpu(self, model_id, name, shape, dtype):
        if name not in self.cpu_arrays_:
            self.cpu_arrays_[model_id][name] = np.ndarray(shape, dtype)

    # produce a new NDARRAY with the correct source pointer
    def get_gpu_workspace_view(self, model_id, device_id, name):
        view_info = self.workspace_views_[model_id][device_id][name]
        ws_name, ws_offset, dtype, shape = view_info
        ws_ptr = self.workspaces_[model_id][device_id][ws_name]["ptr"]
        new_ptr = MemoryPointer(ws_ptr.mem, ws_ptr.ptr + ws_offset)
        return cupy.ndarray(shape, dtype, memptr=new_ptr)

    # produce a new NDARRAY with the correct source pointer
    def get_gpu_static_array(self, model_id, device_id, name):
        static_info = self.statics_[model_id][device_id][name]
        static_shape = static_info["shape"]
        static_dtype = static_info["dtype"]
        static_ptr = static_info["ptr"]
        return cupy.ndarray(static_shape, static_dtype, memptr=static_ptr)

    def get_cpu_array(self, model_id, name):
        return self.cpu_arrays_[model_id][name]

    def populate(self):

        for device_id in self.device_ids_:

            # get a packing from the current pack function
            bytes_usable = self.pool_.memory_size(device_id)
            print(f"device: {device_id}, usable: {bytes_usable}")

            list_of_statics = {}
            list_of_workspaces = {}
            for model_id in self.models_:
                list_of_statics[model_id] = self.statics_[model_id][device_id].items()
                list_of_workspaces[model_id] = self.workspaces_[model_id][
                    device_id
                ].items()

            pack_strat = self.pack_fn(list_of_statics, list_of_workspaces,
                    bytes_usable)

            self.last_pack_strat_ = pack_strat

            # compute workspace dependencies
            self.workspace_deps[device_id] = self.determine_dependencies(
                pack_strat.workspace_stack
            )

            # process new statics
            for level in pack_strat.static_stack:
                for alloc in level:
                    a_name = alloc.name
                    model_id = alloc.model_id
                    if self.statics_[model_id][device_id][a_name]["ptr"] == None:
                        new_static_ptr = self.pool_.alloc_static(
                            device_id, alloc.end - alloc.start
                        )

                        self.statics_[model_id][device_id][a_name]["ptr"] = new_static_ptr

            ws_bytes = pack_strat.ws_bytes
            self.pool_.free_workspace(device_id)
            assert ws_bytes <= self.pool_.memory_available(device_id)

            workspace_root_ptr = self.pool_.alloc_workspace(
                        device_id, ws_bytes)

            for level in pack_strat.workspace_stack:
                for alloc in level:
                    a_name = alloc.name
                    model_id = alloc.model_id
                    new_ws_ptr = MemoryPointer(workspace_root_ptr.mem,
                            workspace_root_ptr.ptr + alloc.start)
                    self.workspaces_[model_id][device_id][a_name]["ptr"] = new_ws_ptr


    def determine_dependencies(self, parallel_allocs):

        deps = {}

        # for all the parallel alloc rows
        #   for each allocation in the row
        #       check all of the previous rows and find what overlaps
        #       and add edges to the graph

        # 0.A 0.B
        # 1.A 1.B

        for pa_row in range(len(parallel_allocs)):
            for alloc in parallel_allocs[pa_row]:
                if alloc.model_id not in deps:
                    deps[alloc.model_id] = set()
                for prev_row in range(pa_row):
                    for prev_alloc in parallel_allocs[prev_row]:
                        if alloc.overlaps(prev_alloc):
                            deps[alloc.model_id].add(prev_alloc.model_id)
                            deps[prev_alloc.model_id].add(alloc.model_id)

        return deps

    def greedy_pack(list_of_statics, list_of_workspaces, bytes_usable):

        statics = []
        static_sum = 0
        for model_id in list_of_statics.keys():
            for static_name, static_info in list_of_statics[model_id]:
                static_sz = static_info["size"]
                statics.append(
                    AllocInterval(
                        model_id, static_name, static_sum, static_sum + static_sz
                    )
                )
                static_sum += static_sz

        print(bytes_usable, static_sum)
        mem_remaining = bytes_usable - static_sum
        assert mem_remaining > 0

        workspace_sum = 0
        for model_id in list_of_workspaces.keys():
            for workspace_name, workspace_info in list_of_workspaces[model_id]:
                workspace_sum += workspace_info["size"]

        if workspace_sum <= mem_remaining:
            # Convert to allocations
            allocs = []
            offset = 0
            for model_id in list_of_workspaces.keys():
                for ws_name, ws_info in list_of_workspaces[model_id]:
                    allocs.append(
                        AllocInterval(
                            model_id, ws_name, offset, offset + ws_info["size"]
                        )
                    )
                    offset += ws_info["size"]

            return AllocStrat(workspace_sum, [statics], [allocs])
        else:
            # NEED TO MODIFY TO MAKE SURE THAT EACH MODEL IS ON A SINGLE ROW

            sums = []
            for model_id in list_of_workspaces.keys():
                sums.append(
                    (
                        reduce(
                            lambda x, y: x[1]["size"] + y[1]["size"],
                            list_of_workspaces[model_id],
                        ),
                        model_id
                    )
                )

            sums = sorted(sums, key=itemgetter(0), reverse=True)

            if max(sums[0]) > mem_remaining:
                raise Exception("Out of memory")

            allocs = []
            max_loop_sum = 0
            while len(sums) > 0:
                allocs_this_round = []
                loop_sum = 0

                while loop_sum < mem_remaining and len(sums) > 0:
                    size, model_id = sums[-1]  # peek smallest model
                    if size + loop_sum < mem_remaining:
                        for ws_name, ws_info in list_of_workspaces[model_id]:
                            allocs_this_round.append(
                                AllocInterval(
                                    model_id, ws_name, loop_sum, loop_sum + ws_info["size"]
                                )
                            )
                            loop_sum += ws_info["size"]  # add size to loop sum
                        sums.pop()
                    else:
                        break

                    if loop_sum > max_loop_sum:
                        max_loop_sum = loop_sum

                allocs.append(allocs_this_round)

            return AllocStrat(max_loop_sum, [statics], allocs)


"""
WE HAVE 5 SPACES

[YY,ZZ,A]
[JJJ,LL] YY, ZZ !
    JJ -> YY,ZZ
    LL -> ZZ,A
[XXX] JJJ,YY,ZZ !
[QQQQ] XXX,JJJ,YY,ZZ
[PPPPP] QQQQ,XXX,JJJ,YY,ZZ,A

d(X) -> d(x-1) U {x-1} U {

XXX JJJ | YY ZZ A

QQQQ -> Q,J,Y

PPPPP -> P,J,Y,Z


YY,ZZ,A
JJJ -> YY,ZZ,XXX
XXX -> XXX


YY,ZZ,A
e = {}

JJJ
e = {JJJ-YY,JJJ-ZZ}

XXX
e = {JJJ-YY,JJJ-ZZ,XXX-JJJ,XXX-YY,XXX-ZZ}
"""


"""
                   A ----- workspace = model1, name = A, buffer = 0, group=(-1)
                 /   \    \
                 B    C   D workspace = model1, name = B, buffer = 1, group=
                            workspace = model1, name = C, buffer = 1,


"""
