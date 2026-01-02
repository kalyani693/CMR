import numpy as np

def sliding_window(signal,window_size,step):
    windows=[]
    indices=[]
    for i in range(0,len(signal)-window_size,step):
        windows.append(signal[i:i+window_size])
        indices.append((i,i+window_size))
    return np.array(windows) , np.array(indices)

