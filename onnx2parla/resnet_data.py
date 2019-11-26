import numpy as np
import logging

def get_test(s, e):
    batch_size = e - s
    np.random.seed(123)
    d = np.random.random((batch_size,3,8,8)).astype(np.float32)
    logging.log(logging.INFO,d)
    return d


