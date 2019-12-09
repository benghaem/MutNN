# ONNX2Parla

ONNX2Parla is an experimental ONNX runtime that uses
[Parla](https://github.com/ut-parla/Parla.py) as a task queue.

ONNX2Parla supports seamless multi-GPU graph execution on CUDA GPUs and provides
baseline implementations of both model and data parallelism.

__WARNING: This is alpha software that relies on alpha software. Have fun!__

Developed by Benjamin Ghaemmaghami & Saraubh Gupta

## Usage

```python
import onnx2parla as o2p
import numpy as np

def load_batch(start, end):
    batch_size = end - start
    return np.random.random((batch_size,3,224,224))

def store_batch(res):
    print(res)

# config with batch_size = 16, total_batches = 256
cfg = o2p.Config(store_batch, load_batch, 16, 256)
o2p_model = o2p.build("resnet18v1.onnx", config)

o2p_model.run()

```

## Operator Support

Current operator support is limited with most of the focus placed on
supporting CNNs.

The full list is [here](https://docs.google.com/spreadsheets/d/1veBajIq8DIdUIYmdqAsWaS6feH-y4oqRgem2Eglg4G8/edit?usp=sharing)


## Where can I get ONNX graphs?

The [ONNX Model Zoo](https://github.com/onnx/models) is a great place to find a
wide selection of pre-trained neural network graphs. All of the major
frameworks also support some form of support for conversion to ONNX Graphs

## Known Issues / Limitations

* Datatypes are assumed to be float32
* No support for multiple graph inputs/outputs
* Conv operator can occasionally crash when run on GPU
* No ONNX versioning support, most operators support only the newest version


## Diagnosing Problems

First check your ONNX graph using something like 
[Netron](https://github.com/lutzroeder/netron) to ensure all operators in your
graph are supported by ONNX2Parla

Next, enable pass debugging mode

```python
cfg = o2p.Config(...)
cfg.debug_passes = True
```

and inspect the generated GML files to see if your graph is correctly being
processed by the system.
