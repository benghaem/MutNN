import torch

class example_model(torch.nn.Module):
    def __init__(self):

        super(example_model, self).__init__()
        self.layer0 = torch.nn.MaxPool2d((3,3), stride=1)

    def forward(self, x):
        l0 = self.layer0(x)

        return l0

m = example_model()

i = torch.rand((4,4,22,22))

print(i)

o = m(i)

print(o)
print(o.shape)
