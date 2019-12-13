import torch
import torch.onnx
import onnx2parla.resnet_data as rn_loader
import onnxruntime as ort
import datetime
import sys

model = torch.hub.load("pytorch/vision:v0.4.2", "resnet18", pretrained=True)

batch_len = 64
total_size = 128 * 12 * 4

torch.onnx.export(
    model,
    (torch.tensor(rn_loader.get_random(0, batch_len))),
    "torch_resnet_18.onnx",
    verbose=True,
    keep_initializers_as_inputs=True,
)

# sys.exit(1)

model.to("cuda")

st = datetime.datetime.now()
with torch.no_grad():
    for batch_id in range(0, total_size, batch_len):
        batch = torch.tensor(rn_loader.get_random(batch_id, batch_id + batch_len)).to(
            "cuda"
        )
        out = model(batch).cpu().numpy()
        # print(out)

end = datetime.datetime.now()

print("torch:", end - st)

ort_session = ort.InferenceSession("torch_resnet_18.onnx")

st = datetime.datetime.now()
for batch_id in range(0, total_size, batch_len):
    batch = {"input.1": rn_loader.get_random(batch_id, batch_id + batch_len)}
    out = ort_session.run(None, batch)
end = datetime.datetime.now()


print("ort:", end - st)
