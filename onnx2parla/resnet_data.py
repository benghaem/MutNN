import numpy as np
import logging

import imageio

def get_test(s, e):
    batch_size = e - s
    #d = imageio.imread("../torch_example/dog.jpg")
    #d = d.transpose(2,0,1)
    #d = np.expand_dims(d,0)
    d = np.zeros((1,3,224,224))
    print(np.shape(d))
    return d


