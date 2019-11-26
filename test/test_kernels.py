import unittest
import numpy as np

from onnx2parla.node import Node, InOut
from onnx2parla.config import Config
from onnx2parla import kernels
from onnx2parla import operators as ops

class TestCPUKernels(unittest.TestCase):

    def test_relu(self):

        c = Config(None,None,4,4)

        io_in = InOut("in",
                      "static",
                      np.array([1,2,3,4]),
                      (4))

        io_out = InOut("out",
                       "dynamic",
                       None,
                       (4))

        am = {
                "out": np.ndarray((4))
                }
        inp = {"X": io_in}
        oup = {"Y": io_out}


        n = Node(0,ops.RELU,inp,oup,{},0)

        fn = kernels.relu_cpu(n,am,c)

        # eval
        fn()
        np.testing.assert_array_equal(io_out.get_data(am), [1,2,3,4])

        # copy new static input in
        np.copyto(io_in.data, [-2,2,-1,1])

        fn()
        np.testing.assert_array_equal(io_out.get_data(am), [0,2,0,1])

        np.copyto(io_in.data, [-2,-2,-1,-100000])

        fn()
        np.testing.assert_array_equal(io_out.get_data(am), [0,0,0,0])


if __name__ == "__main__":
    unittest.main()
