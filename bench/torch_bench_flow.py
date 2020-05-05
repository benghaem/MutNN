import torch
import torch.onnx
from torch.utils.data import DataLoader
import torchvision.models as tvmodels
import onnxruntime as ort
import onnx2parla as o2p
import onnx2parla.vision_dataloaders as vidl
import sys
import datetime
import time
import gc
import numpy as np
from numpy.random import default_rng

rng = default_rng()

models = [tvmodels.mobilenet_v2, tvmodels.resnet50, tvmodels.vgg16]
targets = ["pytorch","o2p","ort","o2p_model","generate"]

target_runtime = int(sys.argv[1])
model_idx = int(sys.argv[2])
batch_len = int(sys.argv[3])
total_len = int(sys.argv[4])
outfile = str(sys.argv[5])

with open(outfile, "a+") as f:

    model_fn = models[model_idx]
    target = targets[target_runtime]
    model_name = str(model_fn.__name__)

    num_gpus = torch.cuda.device_count()

    print(target, model_name)

    model = model_fn(pretrained=True)
    # pytorch --> onnx model conversion
    if (target != "pytorch"):
        # write model out to onnx
        ref_input = torch.tensor(vidl.get_random(0,batch_len*num_gpus))
        torch.onnx.export(model, (ref_input), "bench_out.onnx",
                keep_initializers_as_inputs=True, verbose=True, opset_version=10)

        so = ort.SessionOptions()
        so.optimized_model_filepath = "bench_out.onnx.opt"
        session = ort.InferenceSession("bench_out.onnx", so)

        del session
        del so

    if (target == "generate"):
        sys.exit()

    res = None
    if (num_gpus > 0):
        scaled_batch_len = batch_len*num_gpus
    else:
        scaled_batch_len = batch_len
    latency = []

    # PYTORCH BENCH
    if target == "pytorch":
        # cudaify model
        if (num_gpus > 0):
            model.to('cuda')

        #model = torch.nn.DataParallel(model, device_ids=torch_devices)
        #torch_devices = [torch.device(f"cuda:{ii}") for ii in range(num_gpus)]

        if (num_gpus > 0):
            RandomLoader = DataLoader(vidl.RandomDataset(6144),
                                      batch_size=scaled_batch_len,
                                      num_workers = 8,
                                      pin_memory=True)
        else:
            RandomLoader = DataLoader(vidl.RandomDataset(6144),
                                      batch_size=scaled_batch_len,
                                      num_workers = 8)
             

        st = datetime.datetime.now()
        time.sleep(20)
        with torch.no_grad():
            for batch in RandomLoader:
                time.sleep(rng.exponential(1) / 100)
                lt_st = datetime.datetime.now()
                if (num_gpus > 0):
                    out = model(batch.cuda()).cpu().numpy()
                else:
                    out = model(batch).numpy()
                lt_end = datetime.datetime.now()
                latency.append((lt_st,lt_end))
        end = datetime.datetime.now()

        res = end - st

    # ONNXRUNTIME BENCH
    if target == "ort":
        ort_session = ort.InferenceSession("bench_out.onnx")

        st = datetime.datetime.now()
        for batch_id in range(0,total_len,scaled_batch_len):
            batch = {"input.1": vidl.get_random(batch_id, batch_id + scaled_batch_len)}
            out = ort_session.run(None, batch)
        end = datetime.datetime.now()

        res = end-st

    # ONNX2PARLA BENCH
    if target == "o2p":
        config = o2p.Config(vidl.nop_store,vidl.get_random,batch_len,total_len)
        config.debug_passes = False
        o2p_model = o2p.build("bench_out.onnx", config)

        st = datetime.datetime.now()
        o2p_model.run()
        end = datetime.datetime.now()

        res = end-st

    # ONNX2PARLA BENCH
    if target == "o2p_model":
        config = o2p.Config(vidl.nop_store,vidl.get_random,batch_len,total_len)
        config.debug_passes = False
        config.use_simple_model_para = True
        config.use_data_para = False
        o2p_model = o2p.build("bench_out.onnx", config)

        st = datetime.datetime.now()
        o2p_model.run()
        end = datetime.datetime.now()

        res = end-st

    latency_comp = [(end - start).total_seconds() for start, end in latency]
    
    mean_latency = np.mean(latency_comp)
    latency_99p = np.percentile(latency_comp, 99.0)


    f.write("{},{},{},{},{},{},{}\n".format(model_name,target,batch_len,total_len,res.total_seconds(),mean_latency,
        latency_99p))

