import numpy as np
def normalize(signal,eps=1e-8):
    mean=np.mean(signal,keepdims=True)
    std=np.std(signal,keepdims=True)
    return (signal-mean)/(std+eps)
