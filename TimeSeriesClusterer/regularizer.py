import numpy as np

def min_max(series):
    min_ = np.min(series)
    max_ = np.max(series)
    range_ = max_-min_
    return (series-min_)*100.0/range_

def z_transform(series):
    mean_ = np.mean(series)
    sd_ = np.std(series)
    return (series-mean_)/sd_
