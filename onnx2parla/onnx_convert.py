import onnx

onnx_conversion_info = {
    "Add": {"in": ["A", "B"], "out": ["C"]},
    "Conv": {"in": ["X", "W", "B"], "out": ["Y"]},
    "Transpose": {"in": ["data"], "out": ["transposed"]},
    "MatMul": {"in": ["A", "B"], "out": ["Y"]},
    "BatchNormalization": {
        "in": ["X", "scale", "B", "mean", "var"],
        "out": ["Y", "mean", "var", "saved_mean", "saved_var"],
    },
    "Clip": {"in": ["input", "min", "max"], "out": ["output"]},
    "Flatten": {"in": ["input"], "out": ["output"]},
    "ReduceMean": {"in": ["data"], "out": ["reduced"]},
    "Gemm": {"in": ["A", "B", "C"], "out": ["Y"]},
    "GlobalAveragePool": {"in": ["X"], "out": ["Y"]},
    "MaxPool": {"in": ["X"], "out": ["Y", "Indices"]},
    "Relu": {"in": ["X"], "out": ["Y"]},
    "Reshape": {"in": ["data", "shape"], "out": ["reshaped"]},
    "Pad": {"in": ["data"], "out": ["output"]},
    "AveragePool": {"in": ["X"], "out": ["Y"]},
    "Dropout": {"in": ["data"], "out": ["output", "mask"]},
}


def get_op_input_info(op):
    return onnx_conversion_info[op]["in"]


def get_op_output_info(op):
    return onnx_conversion_info[op]["out"]


def get_op_attr_info(op):
    return onnx_conversion_info[op]["attr"]


def convert_attr(attr):
    if attr.type == onnx.AttributeProto.INTS:
        return tuple(attr.ints)
    if attr.type == onnx.AttributeProto.INT:
        return int(attr.i)
    if attr.type == onnx.AttributeProto.FLOAT:
        return float(attr.f)
