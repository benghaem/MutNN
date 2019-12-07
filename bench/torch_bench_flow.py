import torch
import torch.onnx
import torchvision.models as tvmodels
import onnxruntime as ort
import onnx2parla as o2p
import onnx2parla.vision_dataloaders as vidl
import sys
import datetime

models = [tvmodels.resnet18, tvmodels.vgg16, tvmodels.mobilenet_v2]

batch_len = int(sys.argv[1])
total_len = int(sys.argv[2])

ref_input = torch.tensor(vidl.get_random(0,batch_len))

for model_fn in models:

    name = str(model_fn.__name__)
    model = model_fn()
    # write model out to onnx
    torch.onnx.export(model, (ref_input), "bench_out.onnx", keep_initializers_as_inputs=True, verbose=True)

    so = ort.SessionOptions()
    so.optimized_model_filepath = "bench_out.onnx.opt"
    session = ort.InferenceSession("bench_out.onnx", so)

    # PYTORCH BENCH
    st = datetime.datetime.now()

    # cudaify model
    model.to('cuda')
    model = torch.nn.DataParallel(model)

    with torch.no_grad():
        for batch_id in range(0,total_len,batch_len):
            batch = torch.tensor(vidl.get_random(batch_id,batch_id+batch_len)).to('cuda')
            out = model(batch).cpu().numpy()

    end = datetime.datetime.now()

    pytorch_res = end - st

    del model

    # ONNXRUNTIME BENCH
    ort_session = ort.InferenceSession("bench_out.onnx")

    st = datetime.datetime.now()
    for batch_id in range(0,total_len,batch_len):
        batch = {"input.1": vidl.get_random(batch_id, batch_id + batch_len)}
        out = ort_session.run(None, batch)
    end = datetime.datetime.now()

    ort_res = end-st

    del ort_session

    # ONNX2PARLA BENCH

    config = o2p.Config(vidl.nop_store,vidl.get_random,batch_len,total_len)
    config.debug_passes = True
    o2p_model = o2p.build("bench_out.onnx", config)

    st = datetime.datetime.now()
    o2p_model.run()
    end = datetime.datetime.now()

    o2p_res = end-st

    del o2p_model

    print(name, pytorch_res, ort_res, o2p_res)
