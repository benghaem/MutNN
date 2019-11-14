import parla.array as parray

def add_cpu(node, alloc_map):

    x_io = node.inputs["X"]
    y_io = node.inputs["Y"]
    z_io = node.outputs["Z"]

    x = x_io.get_data(alloc_map)
    y = y_io.get_data(alloc_map)
    z = z_io.get_data(alloc_map)

    def fn():
        parray.copy(z, x + y)
    return fn
