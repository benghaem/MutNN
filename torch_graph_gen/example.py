import torch
import torch.onnx

import chainer
import numpy as np


class example_model(torch.nn.Module):
    def __init__(self):

        super(example_model, self).__init__()
        self.layer0 = torch.nn.Linear(10, 5)
        self.layer1 = torch.nn.Linear(5, 1)

    def forward(self, x):
        l0 = self.layer0(x)
        l1 = self.layer1(l0)

        return l1

class example_conv(torch.nn.Module):
    def __init__(self):

        super(example_conv, self).__init__()
        self.layer0 = torch.nn.Conv2d(3,3,3)
        self.layer1 = torch.nn.Conv2d(3,3,3)
        self.layer2 = torch.nn.Conv2d(3,1,3)

    def forward(self, x):
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)

        return l2


m = example_model()

x = torch.tensor(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ), dtype=torch.float32)
torch.onnx.export(m, (x), "example.onnx", verbose=True,
        keep_initializers_as_inputs=True)

print(m(torch.ones((4, 10))))

m2 = example_conv()
np.random.seed(123)
inp = np.random.random((4,3,224,224)).astype(np.float32)

print(inp)

inp_torch = torch.tensor(inp, dtype=torch.float32)
torch.onnx.export(m2, (inp_torch), "example_conv.onnx", verbose=True,
        keep_initializers_as_inputs=True)

print(m2(inp_torch))

