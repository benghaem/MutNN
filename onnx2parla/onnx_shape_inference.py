import onnx2parla.operators as ops

def identity(shape_x):
    return shape_x

def conv_like(shape_x, shape_k, stride, padding, dialation, groups):

    shape = [0,0,0,0]
    shape[0] = shape_x[0] #N
    shape[1] = shape_k[0] #C
    shape[2] = ((shape_x[2] + 2*padding - shape_k[2] - ((shape_k[2] -1) *
        (dialation - 1))) // stride) + 1
    shape[3] = ((shape_x[3] + 2*padding - shape_k[3] - ((shape_k[3] -1) *
        (dialation - 1))) // stride) + 1

    return tuple(shape)

def gemm_like(shape_x, shape_w, trans_x, trans_w):

    trans_x = (trans_x == 1)
    trans_w = (trans_w == 1)

    if trans_x and not trans_w:
        return (shape_x[1], shape_w[1])
    elif not trans_x and trans_w:
        return (shape_x[0], shape_w[0])
    elif trans_x and trans_w:
        return (shape_x[1], shape_w[0])
    else:
        return (shape_x[0], shape_w[1])

def reshape_like(shape_x, shape):
    if shape[-1] == -1:
        x_prod = 1
        for xx in shape_x:
            x_prod = x_prod * xx

        s_prod = 1
        for ss in shape[:-1]:
            s_prod = s_prod * ss

        last = x_prod // s_prod

        shape_list = list(shape)
        shape_list[-1] = last

        return(tuple(shape_list))

    else:
        return (shape)

def flatten_like(shape_x, n_axis):

    first_half_product = 1
    second_half_product = 1

    for fh in shape_x[0:n_axis]:
        first_half_product = first_half_product * fh

    for sh in shape_x[n_axis:]:
        second_half_product = second_half_product * sh

    return (first_half_product, second_half_product)

def infer_shape(node):

    print("INFER SHAPE for:")
    node.pretty_print()
    print("----------------")

    if node.operator == ops.ADD:
        shape_x = node.inputs["A"].shape
        node.outputs["C"].shape = identity(shape_x)

    if node.operator == ops.RELU:
        shape_x = node.inputs["X"].shape
        node.outputs["Y"].shape = identity(shape_x)

    if node.operator == ops.BATCH_NORM:
        shape_x = node.inputs["X"].shape
        node.outputs["Y"].shape = identity(shape_x)

    if node.operator == ops.DROPOUT:
        shape_x = node.inputs["data"].shape
        node.outputs["output"].shape = identity(shape_x)
        if node.outputs["mask"]:
            node.outputs["mask"].shape = identity(shape_x)

    if node.operator == ops.GEMM:
        shape_x = node.inputs["A"].shape
        shape_w = node.inputs["B"].shape
        trans_x = node.get_attr("transA",0)
        trans_w = node.get_attr("transB",0)

        node.outputs["Y"].shape = gemm_like(shape_x, shape_w, trans_x, trans_w)

    if node.operator == ops.CONV:
        shape_x = node.inputs["X"].shape
        shape_k = node.inputs["W"].shape
        stride = node.get_attr("strides", [1])[0]
        padding = node.get_attr("pads",[0])[0]
        dialation = node.get_attr("dialation",[1])[0]
        groups =node.get_attr("group",1)

        conv_shape = conv_like(shape_x, shape_k, stride,padding, dialation, groups)
        node.outputs["Y"].shape = conv_shape

    if node.operator == ops.GLOBALAVERAGEPOOL:
        shape_x = node.inputs["X"].shape
        shape_k = (shape_x[1], 0, shape_x[2],shape_x[3])
        padding = 0
        dialation = 1
        groups = 1
        stride = 1
        conv_shape = conv_like(shape_x,
                               shape_k,
                               stride, padding, dialation, groups)
        node.outputs["Y"].shape = conv_shape


    if node.operator in [ops.AVERAGE_POOL, ops.MAXPOOL]:
        shape_x = node.inputs["X"].shape
        mini_kernel = node.get_attr("kernel_shape")

        # fiddle with the values here to get the correct shape
        shape_k = (shape_x[1],0,mini_kernel[0],mini_kernel[1])

        padding = node.get_attr("pads", [0])[0]
        stride = node.get_attr("strides",[1])[0]
        dialation = 1
        groups = 1
        conv_shape = conv_like(shape_x,
                               shape_k,
                               stride, padding, dialation, groups)
        node.outputs["Y"].shape = conv_shape

    if node.operator == ops.RESHAPE:
        shape_x = node.inputs["data"].shape
        shape = node.inputs["shape"].get_data({})

        node.outputs["reshaped"].shape = reshape_like(shape_x, shape)

    if node.operator == ops.FLATTEN:
        shape_x = node.inputs["input"].shape
        n_axis = node.get_attr("axis",1)

        node.outputs["output"].shape = flatten_like(shape_x, n_axis)

    node.pretty_print()

    return

