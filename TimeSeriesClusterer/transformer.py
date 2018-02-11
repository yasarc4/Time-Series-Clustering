import pandas as pd
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA


def roll_mean(series, window = 3, center = True):
    return pd.rolling_mean(series,window=window,center=center).fillna(np.mean(series))

def weighted_roll_mean(series, window=3, center = True):
    return ((pd.rolling_sum(series,window=window,center=center) + series)/(window+1.0)).fillna(np.mean(series))

def difference(series):
    shifted = roll_mean(series, window=24, center=False).shift(1)
    return series - shifted.fillna(np.mean(series))

def get_pca_features(df, n_components = 2):
    pca_model = sklearnPCA(n_components=n_components)
    return pca_model.fit_transform(df)
