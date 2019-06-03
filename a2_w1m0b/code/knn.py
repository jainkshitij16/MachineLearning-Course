"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        X = self.X
        y = self.y
        k = self.k

        T,D = Xtest.shape
        N,D = X.shape
        distance = utils.euclidean_dist_squared(X,Xtest)
        sort = np.argsort(distance,axis=0)
        y_pred = np.empty(T)
        for n in range (T):
            y_pred[n] =  stats.mode(y[sort[:k,n]])[0][0]
        return y_pred