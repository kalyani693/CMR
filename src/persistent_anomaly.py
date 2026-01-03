import numpy as np

def compute_persistent(anomaly_flags, K):
    persistent = np.zeros_like(anomaly_flags, dtype=bool)

    start = None
    for i, flag in enumerate(anomaly_flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= K:
                persistent[start:i] = True
            start = None

    # handle case where anomaly continues till end
    if start is not None and len(anomaly_flags) - start >= K:
        persistent[start:] = True

    return persistent

