from cupy.cuda import MemoryPointer, Memory

# memory pointer
# MemoryPointer(Memory, index)
# we grab a new buffer
# then we can produce new pointers by combining buffer and index

class CupyMemPool():

    def __init__(self, device_ids, size):
        self.total_size = total_size
        self.mem = {}

        for device_id in device_ids:
            with cupy.cuda.device(device_id):
                self.mem[device_id] = Memory(size)


    # alloc weights from 0
    def alloc_weights(num_bytes):
        return 

    # alloc workspaces from size-1
    def alloc_workspace(num_bytes):

