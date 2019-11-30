import torch
import torch.onnx

import chainer
import numpy as np


class one_stage_resnet(torch.nn.Module):
    def __init__(self):

        super(one_stage_resnet, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 64, 7, padding=3, stride=2)
        self.bn0   = torch.nn.BatchNorm2d(64)
        self.relu0 = torch.nn.ReLU()
        self.mxp   = torch.nn.MaxPool2d(3, padding=1, stride=2)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.gap   = torch.nn.AvgPool2d(56)
        self.flat  = torch.nn.Flatten()
        self.flin  = torch.nn.Linear(64, 10)

    def forward(self, x):
        c0 = self.conv0(x)
        b0 = self.bn0(c0)
        r0 = self.relu0(b0)
        mxp0 = self.mxp(r0)

        # conv branch
        c1 = self.conv1(mxp0)
        r1 = self.relu1(c1)
        c2 = self.conv2(r1)

        conv_out = mxp0 + c2
        gap_o = self.gap(conv_out)
        flat_o = self.flat(gap_o)
        final = self.flin(flat_o)

        return final


m2 = one_stage_resnet()
np.random.seed(123)
inp = np.random.random((2, 3, 224, 224)).astype(np.float32)


inp_torch = torch.tensor(inp, dtype=torch.float32)
torch.onnx.export(
    m2, (inp_torch), "one_stage_resnet.onnx", verbose=True, keep_initializers_as_inputs=True
)

out = m2(torch.tensor(inp))

print(out)
