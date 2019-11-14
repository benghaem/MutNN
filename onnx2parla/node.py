class Node():

    def __init__(self, operator, inputs, outputs, attrs):

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

    def get_operator(self):
        return operator

    def get_input_name(self, inp):
        return inputs[inp].name

    def get_output_name(self, out):
        return outputs[out].name

    def get_attr(self, attr):
        return attrs[attr]

    def __str__(self):
        return "{} @ {} -> {}".format(self.operator, self.device_type, self.fn)

class InOut():

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

