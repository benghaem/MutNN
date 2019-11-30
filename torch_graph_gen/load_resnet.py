import onnxruntime as ort
import numpy as np
import imageio
import matplotlib.pyplot as plt



def preprocess(img_data):
    mean_vec = np.array([.485, 0.456, 0.406])
    stdev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype(np.float32)

    # for all channels in the image
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stdev_vec[i]

    return norm_img_data

def get_test(s, e):
    d = imageio.imread("dog.jpg")
    d = d.transpose(2,0,1)
    d = preprocess(d)
    d = np.expand_dims(d,0)
    print(np.shape(d))
    return d

ort_session = ort.InferenceSession("../onnx2parla/test_models/resnet18v1.onnx")

input_data = get_test(0,1)

o = ort_session.run(None,{"data":input_data.astype(np.float32)})

print(np.shape(o))

res = o[0][0].tolist()

print(sorted(zip(res,range(len(res))), reverse=True)[0:5])



