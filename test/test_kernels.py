import unittest
import numpy as np

from onnx2parla.node import Node, InOut
from onnx2parla.config import Config
from onnx2parla import kernels
from onnx2parla import operators as ops

import chainer


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

        io_in = InOut("in", "static", np.ndarray((4, 3, 22, 22)), (4))

        io_kern = InOut("kern", "static", np.ndarray((1, 3, 3, 3)), (4))

        io_bias = InOut("bias", "static", np.ndarray((1)), (4))

        io_out = InOut("out", "dynamic", None, (4))

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
        o = chainer.functions.convolution_2d(i, w, b=b,
                                             stride=(1,1),
                                             pad=(0,0),
                                             dilate=(1,1),
                                             groups=1)
        fn()

        np.testing.assert_array_almost_equal(o, io_out.data)


    def test_conv_default_attrs(self):

        c = Config(None, None, 4, 4)

        io_in = InOut("in", "static", np.ndarray((4, 3, 22, 22)), (4))

        io_kern = InOut("kern", "static", np.ndarray((1, 3, 3, 3)), (4))

        io_bias = InOut("bias", "static", np.ndarray((1)), (4))

        io_out = InOut("out", "dynamic", None, (4))

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
        attrs = {"dilations": (1,1),
                 "group": (1),
                 "kernel_shape": (3,3),
                 "pads": (0,0,0,0),
                 "strides": (1,1,1,1)}

        n = Node(0, ops.CONV, inp, oup, attrs, 0)
        fn = kernels.conv_cpu(n, am, c)

        # chainer with previous config
        o = chainer.functions.convolution_2d(i, w, b=b,
                                             stride=(1,1),
                                             pad=(0,0),
                                             dilate=(1,1),
                                             groups=1)
        fn()

        np.testing.assert_array_almost_equal(o, io_out.data)


    def test_conv_stride(self):

        c = Config(None, None, 4, 4)

        io_in = InOut("in", "static", np.ndarray((4, 3, 22, 22)), (4))

        io_kern = InOut("kern", "static", np.ndarray((1, 3, 3, 3)), (4))

        io_bias = InOut("bias", "static", np.ndarray((1)), (4))

        io_out = InOut("out", "dynamic", None, (4))

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
        attrs = {"dilations": (1,1),
                 "group": (1),
                 "kernel_shape": (3,3),
                 "pads": (0,0,0,0),
                 "strides": (2,2,2,2)}

        n = Node(0, ops.CONV, inp, oup, attrs, 0)
        fn = kernels.conv_cpu(n, am, c)

        # chainer with previous config
        o = chainer.functions.convolution_2d(i, w, b=b,
                                             stride=(2,2),
                                             pad=(0,0),
                                             dilate=(1,1),
                                             groups=1)
        fn()

        np.testing.assert_array_almost_equal(o, io_out.data)




if __name__ == "__main__":
    unittest.main()
