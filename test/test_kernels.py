import unittest
import numpy as np
import cupy

from onnx2parla.node import Node, InOut
from onnx2parla.config import Config
from onnx2parla import kernels
from onnx2parla import operators as ops

import chainer
import torch

class TestGPUKernels(unittest.TestCase):

    def test_copy(self):

        c = Config(None, None, 4, 4)

        size = (4,3,224,224)

        io_in = InOut("in", "static", np.random.random(size), size)

        io_gpu = InOut("gpu", "dynamic", None, size)

        io_return = InOut("return", "dynamic", None, size)

        with cupy.cuda.Device(0):
            gpu_buffer = cupy.ndarray((size))

        am = {"gpu": gpu_buffer,
              "return": np.ndarray((size))}

        inp_c0 = {"X": io_in}
        oup_c0 = {"Z": io_gpu}


        inp_c1 = {"X": io_gpu}
        oup_c1 = {"Z": io_return}

        c0 = Node(0, ops.O2P_COPY, inp_c0, oup_c0, {}, 0)
        c0.device_id = 0
        c1 = Node(0, ops.O2P_COPY, inp_c1, oup_c1, {}, 0)
        c1.device_id = 0

        fn_c0 = kernels.copy(c0, am, c)
        fn_c1 = kernels.copy(c1, am, c)

        #copy to gpu
        fn_c0()

        #execute +1
        cupy.copyto(gpu_buffer,gpu_buffer + 1)

        #copy back
        fn_c1()

        ref_plus_one = io_in.get_data(am) + 1

        cupy.testing.assert_array_equal(io_gpu.get_data(am), ref_plus_one)
        np.testing.assert_equal(io_return.get_data(am), ref_plus_one)


class TestCPUKernels(unittest.TestCase):
    def test_relu(self):

        c = Config(None, None, 4, 4)

        io_in = InOut("in", "static", np.array([1, 2, 3, 4]), (4))

        io_out = InOut("out", "dynamic", None, (4))

        am = {"out": np.ndarray((4))}
        inp = {"X": io_in}
        oup = {"Y": io_out}

        n = Node(0, ops.RELU, inp, oup, {}, 0)

        fn = kernels.relu_cpu(n, am, c)

        # eval
        fn()
        np.testing.assert_array_equal(io_out.get_data(am), [1, 2, 3, 4])

        # copy new static input in
        np.copyto(io_in.data, [-2, 2, -1, 1])

        fn()
        np.testing.assert_array_equal(io_out.get_data(am), [0, 2, 0, 1])

        np.copyto(io_in.data, [-2, -2, -1, -100000])

        fn()
        np.testing.assert_array_equal(io_out.get_data(am), [0, 0, 0, 0])

    def test_conv_no_attrs(self):

        c = Config(None, None, 4, 4)

        io_in = InOut("in", "static", np.ndarray((4, 3, 22, 22)), (4, 3, 22, 22))

        io_kern = InOut("kern", "static", np.ndarray((1, 3, 3, 3)), (1, 3, 3, 3))

        io_bias = InOut("bias", "static", np.ndarray((1)), (1))

        io_out = InOut("out", "dynamic", None, (4, 1, 10, 10))

        i = np.random.random(np.shape(io_in.data))
        w = np.random.random(np.shape(io_kern.data))
        b = np.random.random(np.shape(io_bias.data))

        np.copyto(io_in.data, i)
        np.copyto(io_kern.data, w)
        np.copyto(io_bias.data, b)

        # ---TEST 1: X,W,B no attrs
        am = {"out": np.ndarray((4, 1, 20, 20))}
        inp = {"X": io_in, "W": io_kern, "B": io_bias}
        oup = {"Y": io_out}
        attrs = {}

        n = Node(0, ops.CONV, inp, oup, attrs, 0)
        fn = kernels.conv_cpu(n, am, c)

        # chainer with onnx default
        o = chainer.functions.convolution_2d(
            i, w, b=b, stride=(1, 1), pad=(0, 0), dilate=(1, 1), groups=1
        ).array
        fn()

        np.testing.assert_array_almost_equal(o, io_out.get_data(am))

    def test_conv_default_attrs(self):

        c = Config(None, None, 4, 4)

        io_in = InOut("in", "static", np.ndarray((4, 3, 22, 22)), (4, 3, 22, 22))

        io_kern = InOut("kern", "static", np.ndarray((1, 3, 3, 3)), (1, 3, 3, 3))

        io_bias = InOut("bias", "static", np.ndarray((1)), (1))

        io_out = InOut("out", "dynamic", None, (4, 1, 10, 10))

        i = np.random.random(np.shape(io_in.data))
        w = np.random.random(np.shape(io_kern.data))
        b = np.random.random(np.shape(io_bias.data))

        np.copyto(io_in.data, i)
        np.copyto(io_kern.data, w)
        np.copyto(io_bias.data, b)

        # ---TEST 2: X,W,B default attrs
        am = {"out": np.ndarray((4, 1, 20, 20))}
        inp = {"X": io_in, "W": io_kern, "B": io_bias}
        oup = {"Y": io_out}
        attrs = {
            "dilations": (1, 1),
            "group": (1),
            "kernel_shape": (3, 3),
            "pads": (0, 0, 0, 0),
            "strides": (1, 1, 1, 1),
        }

        n = Node(0, ops.CONV, inp, oup, attrs, 0)
        fn = kernels.conv_cpu(n, am, c)

        # chainer with previous config
        o = chainer.functions.convolution_2d(
            i, w, b=b, stride=(1, 1), pad=(0, 0), dilate=(1, 1), groups=1
        ).array
        fn()

        np.testing.assert_array_almost_equal(o, io_out.get_data(am))

    def test_conv_stride(self):

        c = Config(None, None, 4, 4)

        io_in = InOut("in", "static", np.ndarray((4, 3, 22, 22)), (4, 3, 22, 22))

        io_kern = InOut("kern", "static", np.ndarray((1, 3, 3, 3)), (1, 3, 3, 3))

        io_bias = InOut("bias", "static", np.ndarray((1)), (1))

        io_out = InOut("out", "dynamic", None, (4, 1, 10, 10))

        i = np.random.random(np.shape(io_in.data))
        w = np.random.random(np.shape(io_kern.data))
        b = np.random.random(np.shape(io_bias.data))

        np.copyto(io_in.data, i)
        np.copyto(io_kern.data, w)
        np.copyto(io_bias.data, b)

        # ---TEST 3: X,W,B default attrs
        am = {"out": np.ndarray((4, 1, 10, 10))}
        inp = {"X": io_in, "W": io_kern, "B": io_bias}
        oup = {"Y": io_out}
        attrs = {
            "dilations": (1, 1),
            "group": (1),
            "kernel_shape": (3, 3),
            "pads": (0, 0, 0, 0),
            "strides": (2, 2, 2, 2),
        }

        n = Node(0, ops.CONV, inp, oup, attrs, 0)
        fn = kernels.conv_cpu(n, am, c)

        # chainer with previous config
        o = chainer.functions.convolution_2d(
            i, w, b=b, stride=(2, 2), pad=(0, 0), dilate=(1, 1), groups=1
        ).array
        fn()

        np.testing.assert_array_almost_equal(o, io_out.get_data(am))

    def test_maxpool_defaults(self):

        B = 4
        C = 4
        H = 22
        W = 22

        K_size = (3, 3)

        in_shape = (B, C, H, W)
        out_shape = (B, C, 20, 20)

        c = Config(None, None, B, B)

        io_in = InOut("in", "static", np.ndarray(in_shape), in_shape)
        io_out = InOut("out", "dynamic", None, out_shape)

        i = np.random.random(np.shape(io_in.data))

        np.copyto(io_in.data, i)

        ref_mod = torch.nn.MaxPool2d(
            K_size, stride=1, dilation=1, padding=0, ceil_mode=False
        )

        torch_i = torch.tensor(i)
        ref = ref_mod(torch_i).numpy()

        am = {"out": np.ndarray(out_shape)}
        inp = {"X": io_in}
        oup = {"Y": io_out}
        attrs = {"kernel_shape": K_size}

        n = Node(0, ops.MAXPOOL, inp, oup, attrs, 0)

        test_fn = kernels.maxpool_cpu(n, am, c)

        test_fn()

        np.testing.assert_array_almost_equal(io_out.get_data(am), ref)

    def test_maxpool_big_stride(self):

        B = 4
        C = 4
        H = 22
        W = 22

        K_size = (3, 3)

        in_shape = (B, C, H, W)
        out_shape = (B, C, 7, 7)

        c = Config(None, None, B, B)

        io_in = InOut("in", "static", np.ndarray(in_shape), in_shape)
        io_out = InOut("out", "dynamic", None, out_shape)

        i = np.random.random(np.shape(io_in.data))

        np.copyto(io_in.data, i)

        ref_mod = torch.nn.MaxPool2d(
            K_size, stride=3, dilation=1, padding=0, ceil_mode=False
        )

        torch_i = torch.tensor(i)
        ref = ref_mod(torch_i).numpy()

        am = {"out": np.ndarray(out_shape)}
        inp = {"X": io_in}
        oup = {"Y": io_out}
        attrs = {"kernel_shape": K_size, "strides": (3, 3, 3, 3)}

        n = Node(0, ops.MAXPOOL, inp, oup, attrs, 0)

        test_fn = kernels.maxpool_cpu(n, am, c)

        test_fn()

        np.testing.assert_array_almost_equal(io_out.get_data(am), ref)

    def test_batchnorm_defaults(self):

        B = 4
        C = 4
        H = 22
        W = 22

        K_size = (3, 3)

        in_shape = (B, C, H, W)
        out_shape = (B, C, H, W)

        c = Config(None, None, B, B)

        io_in = InOut("in", "static", np.ndarray(in_shape), in_shape)
        io_scale = InOut("scale", "static", np.ndarray((C)), (C))
        io_B = InOut("B", "static", np.ndarray((C)), (C))
        io_mean = InOut("mean", "static", np.ndarray((C)), (C))
        io_var = InOut("var", "static", np.ndarray((C)), (C))
        io_out = InOut("out", "dynamic", None, out_shape)

        np.random.seed(123)

        i = np.random.random(np.shape(io_in.data))
        s = np.random.random(np.shape(io_scale.data))
        b = np.random.random(np.shape(io_B.data))
        mean = np.random.random(np.shape(io_mean.data))
        var = np.random.random(np.shape(io_var.data))

        np.copyto(io_in.data, i)
        np.copyto(io_scale.data, s)
        np.copyto(io_B.data, b)
        np.copyto(io_mean.data, mean)
        np.copyto(io_var.data, var)

        eps = 1e-05
        momentum_torch = 0.5
        momentum_test = 0.4

        torch_i = torch.tensor(i)
        torch_w = torch.tensor(s)
        torch_b = torch.tensor(b)
        torch_mean = torch.tensor(mean)
        torch_var = torch.tensor(var)

        ref = torch.nn.functional.batch_norm(
            torch_i,
            torch_mean,
            torch_var,
            weight=torch_w,
            bias=torch_b,
            training=False,
            momentum=momentum_torch,
            eps=eps,
        ).numpy()

        ref_chainer = chainer.functions.fixed_batch_normalization(
            i,
            s,
            b,
            mean,
            var,
            eps=eps,
        ).array

        am = {"out": np.ndarray(out_shape)}
        inp = {"X": io_in, "scale": io_scale, "B": io_B, "mean": io_mean, "var": io_var}
        oup = {"Y": io_out}
        attrs = {"epsilon": eps, "momentum": momentum_test}

        n = Node(0, ops.BATCH_NORM, inp, oup, attrs, 0)

        test_fn = kernels.batchnorm_cpu(n, am, c)

        test_fn()

        #np.testing.assert_array_almost_equal(ref, ref_chainer)
        np.testing.assert_array_almost_equal(io_out.get_data(am), ref_chainer)


if __name__ == "__main__":
    unittest.main()
