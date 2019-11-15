class Node:
    def __init__(self, operator, inputs, outputs, attrs, instance_id):

        # Onnx
        self.operator = operator
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs

        # Parla
        self.fn = None
        self.device_type = None
        self.device_id = None
        self.task_id = None

        self.instance_id = instance_id

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
        return "{} @ {} -> {}".format(self.operator, self.device_type, self.fn)


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

