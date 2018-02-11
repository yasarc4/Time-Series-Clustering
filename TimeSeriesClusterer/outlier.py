import numpy as np

def right_outlier_iqr(x,n_iqr = 1.5):
    q1,q3 = np.percentile(x, [25, 75])
    iqr = q3-q1
    return q3 + iqr*n_iqr

def right_outlier_z(x, n_sd=3):
    return np.mean(x) + np.std(x)*n_sd
