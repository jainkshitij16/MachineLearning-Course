import numpy as np
from scipy import stats
from scipy.stats import mode
from utils import euclidean_dist_squared
from random_tree import RandomTree



class RandomForest:
	def __init__(self, num_trees, max_depth):
		self.num_trees = num_trees
		self.max_depth = max_depth



	def fit(self, X, y):
		N = X.shape[0]
		self.submodel = [RandomTree] * self.num_trees
		for nt in range(self.num_trees):
			self.submodel[nt] = RandomTree(self.max_depth)
			self.submodel[nt].fit(X,y)



	def predict(self, Xtest):
		N = Xtest.shape[0]
		y_pred = np.array([np.zeros(N)]*self.num_trees)
		for nt in range(self.num_trees):
				y_pred[nt] = self.submodel[nt].predict(Xtest)
		y_pred = y_pred.T
		value = np.zeros(N)
		for n in range(N):
			value[n] = stats.mode(y_pred[n])[0][0]
		return value
