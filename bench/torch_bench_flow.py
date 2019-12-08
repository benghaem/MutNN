import torch
import torch.onnx
import torchvision.models as tvmodels
import onnxruntime as ort
import onnx2parla as o2p
import onnx2parla.vision_dataloaders as vidl
import sys
import datetime
import time
import gc

models = [tvmodels.mobilenet_v2, tvmodels.resnet50, tvmodels.vgg16]
targets = ["pytorch","o2p","ort","o2p_model"]

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
    ref_input = torch.tensor(vidl.get_random(0,batch_len*num_gpus))

    print(target, model_name)

    model = model_fn(pretrained=True)
    # write model out to onnx
    torch.onnx.export(model, (ref_input), "bench_out.onnx",
            keep_initializers_as_inputs=True, verbose=True, opset_version=10)

    so = ort.SessionOptions()
    so.optimized_model_filepath = "bench_out.onnx.opt"
    session = ort.InferenceSession("bench_out.onnx", so)

    del session
    del so

    res = None
    scaled_batch_len = batch_len*num_gpus

    # PYTORCH BENCH
    if target == "pytorch":
        # cudaify model
        model.to('cuda')
        torch_devices = [torch.device(f"cuda:{ii}") for ii in range(num_gpus)]

        model = torch.nn.DataParallel(model, device_ids=torch_devices)

        st = datetime.datetime.now()
        with torch.no_grad():
            for batch_id in range(0,total_len,scaled_batch_len):
                batch = torch.tensor(vidl.get_random(batch_id,batch_id+scaled_batch_len)).to('cuda')
                out = model(batch).cpu().numpy()
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


    f.write("{},{},{},{},{}\n".format(model_name,target,batch_len,total_len,res.total_seconds()))
