import numpy as np
import logging

import imageio

def preprocess(img_data):
    mean_vec = np.array([.485, 0.456, 0.406])
    stdev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype(np.float32)

    # for all channels in the image
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stdev_vec[i]

    return norm_img_data

def get_test(s, e):
    batch_size = e - s
    d = imageio.imread("input/dog.jpg")
    d = d.transpose(2,0,1)
    dp = preprocess(d)
    dp = np.tile(dp, (batch_size,1,1,1))
    #d = np.zeros((1,3,224,224))
    print(np.shape(dp))
    return dp

def get_random(s, e):
    batch_size = e - s
    d = np.random.random((batch_size,3,224,224)).astype(np.float32)
    return d

def get_fixed_random(s, e):
    batch_size = e - s
    np.random.seed(123)
    d = np.random.random((batch_size,3,224,224)).astype(np.float32)
    return d

def store(x):
    print(x)

def echo_top5(x):
    for batch_el in x:
        t5 = sorted(zip(batch_el,range(len(batch_el))),reverse=True)[0:5]
        print(t5)
