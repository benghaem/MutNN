class Node:
    def __init__(self, node_id, operator, inputs, outputs, attrs, instance_id):

        # Identification
        self.node_id = node_id
        self.instance_id = instance_id

        # Operational
        self.operator = operator
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs

        # Parla
        self.fn = None
        self.device_type = None
        self.device_id = None
        self.last_task_obj = None
        self.last_launch_batch_id = -1

    def get_operator(self):
        return self.operator

    def replace_io_for_input_buffer(self, buf_name, new_io):
        name_to_replace = None
        for name, io in self.inputs.items():
            if io.name == buf_name:
                name_to_replace = name
                break

        if name_to_replace is None:
            raise ValueError(f"buffer {buf_name} is not an input for {self}")

        self.inputs[name_to_replace] = new_io

    def get_device(self):
        return (self.device_type, self.device_id)

    def get_input(self, inp):
        real_io = self.inputs.get(inp)
        if not real_io:
            fake_io = InOut("__fake", "__fake", None, None)
            return fake_io
        return real_io

    def get_output(self, out):
        real_io = self.outputs.get(out)
        if not real_io:
            fake_io = InOut("__fake", "__fake", None, None)
            return fake_io
        return real_io

    def get_input_name(self, inp):
        return self.inputs[inp].name

    def get_output_name(self, out):
        return self.outputs[out].name

    def get_attr(self, attr, default=None):
        return self.attrs.get(attr, default)

    def set_attr(self, attr, value):
        self.attrs[attr] = value
        return True

    def __str__(self):
        return (
            f"[{self.node_id}@{self.device_type}.{self.device_id}]" f" {self.operator}"
        )

    def pretty_print(self):
        print(self)
        print("inputs:")
        for i in self.inputs.items():
            print("\t", i[0], ": ", i[1])
        print("outputs:")
        for o in self.outputs.items():
            print("\t", o[0], ": ", o[1])
        print("attrs:")
        for a in self.attrs.items():
            print("\t", a)


def node_stringizer(value):

    """
    Support function for networkx graph export
    """

    if isinstance(value, Node):
        return value.operator + "_" + str(value.device_type)
    else:
        return str(value)


class InOut:

    """
    Represents an undirected graph edge and an associated buffer
    for data

    Static data is stored directly in the IO while dynamic data must
    be looked up in the alloc map
    """

    def __init__(self, name, kind, data, shape):
        self.name = name
        self.kind = kind
        self.data = data
        self.shape = shape

    def get_data(self, alloc_map):

        """
        Get the actual data associated with the IO

        Static data is accessed directly while pointer and
        dynamic data is looked up in the alloc map
        """

        if self.kind == "pointer":
            return alloc_map[self.name]
        if self.kind == "dynamic":
            return alloc_map[self.name]
        if self.kind == "static":
            return self.data
        return None

    def __str__(self):
        return f"{self.name} {{ {self.kind} < {self.shape} > }}"
