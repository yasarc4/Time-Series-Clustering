from sklearn.cluster import KMeans
from datetime import datetime as dt
from slugify import slugify
from sklearn.cluster import KMeans
from sklearn import metrics
from transformer import *



class KMeansHelper:
    def __init__(self, optimal_k = None, n_iterations = 500, optimal_k_function = None):
        self.optimal_k = None
        self.all_state = {}
        self.all_k = []
        self._n_iterations = n_iterations
        self._optimal_k_function = optimal_k_function

    def get_all_state(self):
        return self.all_state

    def fit(self, df, random_state = 1):
        self._random_state = random_state
        self._df = df
        self._pca_df = get_pca_features(df)
        self.all_datapoints = df.index
        if len(df)<5:
            raise('The given data already has only {} data points.'.format(len(df)))
        if self.optimal_k == None:
            min_k = max(int(len(df)**0.33), 2)
            max_k = int(np.ceil(len(df)**0.5)+1)
            self.all_k = range(min_k, max_k)
        else:
            self.all_k = [self.optimal_k]
        for k in self.all_k:
            self.all_state[k] = self.get_model(k)
        self.get_optimal_k()

    def get_optimal_model(self):
        return self.all_state[self.optimal_k]

    def get_model(self, k):
        model = KMeans(n_clusters=k, random_state=self._random_state).fit(self._df)
        result = Clustering_Result(model, self._df)
        return result

    def get_optimal_k(self):
        if self._optimal_k_function == None:
            self.optimal_k = max(self.get_weighted_silhoutte_score(), key = lambda x: x[1])[0]
        else:
            self.optimal_k = self._optimal_k_function(self.all_k)

    def get_weighted_silhoutte_score(self):
        for k,v in self.all_state.items():
            yield (k,v.silhoutte_score**(1/k))


class Clustering_Result:
    def __init__(self, model, df):
        self._slug =slugify(str(dt.now()))
        self.model = model
        self.labels = model.labels_
        self.silhoutte_score = metrics.silhouette_score(df, self.labels, metric='euclidean')
        self.calinski_score = metrics.calinski_harabaz_score(df, self.labels)
        self.plot_name = type(model).__name__ + self._slug + '.png'
        self.maps = dict(zip(df.index,self.labels))
