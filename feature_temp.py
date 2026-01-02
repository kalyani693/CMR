import numpy as np

def feature_extraction(window):
    feature=[]
    for w in window:
        w=w.flatten()
        t=np.arange(len(w))
        slope=np.polyfit(t,w,1)[0]
        feature.append([np.mean(w),slope])
    return np.array(feature)