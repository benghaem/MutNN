import unittest
import cupy
import time

from onnx2parla.memory import CupyMemPool, AllocInterval, AllocMap

class TestCupyMempool(unittest.TestCase):

    def test_1gb_allocation(self):

        size = 1024 * 10**6
        pool0 = CupyMemPool([0], size)

        assert(pool0.memory_available(0) == size)

    def test_failing_allocation(self):
        size = 1024 * 10**9

        ee = None

        try:
            pool0 = CupyMemPool([0], size)
        except Exception as e:
            ee = e

        assert(type(ee) == cupy.cuda.runtime.CUDARuntimeError)


    def test_weight_allocation(self):
        size = 1024 * 10**6
        pool0 = CupyMemPool([0], size)

        small_alloc = 1024
        static_0 = pool0.alloc_static(0,small_alloc)
        static_1 = pool0.alloc_static(0,small_alloc)
        static_2 = pool0.alloc_static(0,small_alloc)

        assert(static_0.ptr != static_1.ptr != static_2.ptr)
        assert(pool0.memory_available(0) == (size - 3*small_alloc))


    def test_workspace_allocation(self):
        size = 1024 * 10**6
        pool0 = CupyMemPool([0], size)

        small_alloc = 1024
        workspace_0 = pool0.alloc_workspace(0,small_alloc)
        workspace_1 = pool0.alloc_workspace(0,small_alloc)
        workspace_2 = pool0.alloc_workspace(0,small_alloc)

        assert(workspace_0.ptr != workspace_1.ptr != workspace_2.ptr)
        assert(pool0.memory_available(0) == (size - 3*small_alloc))


    def test_mixed_alloc(self):
        size = 1024 * 10**6
        pool0 = CupyMemPool([0], size)

        small_alloc = 1024
        workspace_0 = pool0.alloc_workspace(0,small_alloc)
        static_0 = pool0.alloc_static(0,small_alloc)

        assert(pool0.memory_available(0) == (size - 2*small_alloc))


    def test_failing_mixed_alloc(self):
        size = 1600
        pool0 = CupyMemPool([0], size)

        small_alloc = 1024
        workspace_0 = pool0.alloc_workspace(0, small_alloc)

        assert(workspace_0 != None)

        static_0 = pool0.alloc_static(0,small_alloc)

        assert(static_0 == None)

        workspace_1 = pool0.alloc_workspace(0, small_alloc)

        assert(workspace_1 == None)


    def test_failing_alloc_then_free(self):
        size = 1600
        pool0 = CupyMemPool([0], size)

        small_alloc = 1024
        workspace_0 = pool0.alloc_workspace(0, small_alloc)

        assert(workspace_0 != None)

        static_0 = pool0.alloc_static(0,small_alloc)

        assert(static_0 == None)

        pool0.free_workspace(0)

        static_0 = pool0.alloc_static(0,small_alloc)
        assert(static_0 != None)

class TestAllocInterval(unittest.TestCase):

    def test_overlaps(self):
        #| ---- |
        #| -------|

        ai0 = AllocInterval("dummy",0,5)
        ai1 = AllocInterval("dummy",0,10)

        assert(ai0.overlaps(ai1))
        assert(ai1.overlaps(ai0))

        #|------|
        #  |--|

        ai2 = AllocInterval("dummy",-5,5)
        ai3 = AllocInterval("dummy",-3,3)

        assert(ai2.overlaps(ai3))
        assert(ai3.overlaps(ai2))


        #|------|
        #   |---|
        ai4 = AllocInterval("dummy",-10,10)
        ai5 = AllocInterval("dummy",0,10)

        assert(ai4.overlaps(ai5))
        assert(ai5.overlaps(ai4))

        #|------|
        #       |------|


        ai6 = AllocInterval("dummy",-10,0)
        ai7 = AllocInterval("dummy",0,10)
        assert(not ai6.overlaps(ai7))
        assert(not ai7.overlaps(ai6))

        #|-------|
        #            |-------|

        ai8 = AllocInterval("dummy",0,10)
        ai9 = AllocInterval("dummy",20,30)
        assert(not ai8.overlaps(ai9))
        assert(not ai9.overlaps(ai8))

class TestAllocMap(unittest.TestCase):

    def test_determine_deps(self):

        deps_row_0 = [AllocInterval("C",0,2),
                      AllocInterval("B",2,4),
                      AllocInterval("A",4,5)]

        deps_row_1 = [AllocInterval("E",0,3),
                      AllocInterval("D",3,5)]

        deps_row_2 = [AllocInterval("F",0,5)]

        rows = [deps_row_0, deps_row_1, deps_row_2]

        deps = AllocMap.determine_dependencies(rows)

        print(deps)

        assert(len(deps["A"]) == 2)
        assert(len(deps["B"]) == 3)
        assert(len(deps["C"]) == 2)
        assert(len(deps["D"]) == 3)
        assert(len(deps["E"]) == 3)
        assert(len(deps["F"]) == 5)

        assert("A" not in deps["B"])
        assert("A" not in deps["C"])
        assert("A" in deps["D"])
        assert("A" not in deps["E"])
        assert("A" in deps["F"])

        assert("B" not in deps["A"])
        assert("B" not in deps["C"])
        assert("B" in deps["D"])
        assert("B" in deps["E"])

