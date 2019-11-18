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

    def get_operator(self):
        return self.operator

    def get_input_name(self, inp):
        return self.inputs[inp].name

    def get_output_name(self, out):
        return self.outputs[out].name

    def get_attr(self, attr):
        return self.attrs[attr]

    def set_attr(self, attr, value):
        self.attrs[attr] = value
        return True

    def __str__(self):
        return f"[{self.node_id}@{self.device_type}.{self.device_id}] {self.operator}"

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


class InOut:
    def __init__(self, name, kind, data, shape):
        self.name = name
        self.kind = kind
        self.data = data
        self.shape = shape

    def get_data(self, alloc_map):
        if self.kind == "pointer":
            return alloc_map[self.name]
        if self.kind == "dynamic":
            return alloc_map[self.name]
        if self.kind == "static":
            return self.data
        return None

    def __str__(self):
        return f"{self.name} {{ {self.kind} < {self.shape} > }}"
