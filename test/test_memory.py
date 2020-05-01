import unittest
import cupy

from onnx2parla.memory import CupyMemPool


class TestCupyMempool(unittest.TestCase):

    def test_simple_allocation(self):

        size = 1024 * 10**6
        pool0 = CupyMemPool([0], size)

