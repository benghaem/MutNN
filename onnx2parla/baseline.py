import parla
from parla import tasks
from parla import array
from parla import cpu
from parla import cuda
from parla import ldevice
import numpy as np
import logging
import copy


def add_kernel(out, a, b, i):
    def fn():
        array.copy(out[i : i + 1], a[i] + b[i])

    return fn


def build_kernel(operand, i, partial_sums, a_part, b_part, dep, loc):
    if operand == "+":
        return add_kernel(partial_sums, a_part, b_part, i)
    if operand == "*":

        def fn():
            array.copy(partial_sums[i : i + 1], a_part[i] * b_part[i])

        return fn

    return None


def build_graph(buffers, info):
    # return [ (kernel_fns, dependencies), .... ]
    pass


async def execute_graph(output, partial_sums, a_part, b_part, divisions):
    async with tasks.finish():
        for i in range(divisions):

            fn = build_kernel("+", i, partial_sums, a_part, b_part, None, None)
            tasks.spawn(placement=cpu.cpu(0))(fn)

        # manual copy
    for i, v in enumerate(partial_sums):
        output[i] = v


def main():

    divisions = 2
    a = np.random.rand(divisions)
    b = np.random.rand(divisions)

    logging.basicConfig(filename="out.log", level=logging.DEBUG)

    devs = list(cpu.cpu.devices)
    print(devs)

    mapper = ldevice.LDeviceSequenceBlocked(divisions, devices=devs)

    a_part = mapper.partition_tensor(a)
    b_part = mapper.partition_tensor(b)

    result = np.empty(divisions)

    output = np.empty(divisions)
    partial_sums = np.empty(divisions)

    tasks.spawn(placement=cpu.cpu(0))(
        execute_graph(output, partial_sums, a_part, b_part, divisions)
    )
    print(output, a, b)
    print(partial_sums)


if __name__ == "__main__":
    main()
