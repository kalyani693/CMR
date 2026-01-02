import numpy as np
from scipy.stats import kurtosis,skew


def extract_features(windows):
    feats=[]

    for w in windows:
        hist,_=np.histogram(w,bins=10,density=True)
        hist=hist+1e-8
        entropy=-np.sum(hist*np.log(hist))
        zcr=np.sum(np.diff(np.sign(w))!=0)
        feats.append([np.mean(w),np.std(w),
                      np.max(w),np.min(w),
                      np.ptp(w),np.sum(w**2),
                      skew(w),
                      kurtosis(w),
                      zcr,
                      entropy])
                     
    return np.array(feats)    

   