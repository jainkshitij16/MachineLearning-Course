import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

class logRegL1:
    # Logistic Regression
    def __init__(self, verbose=1, maxEvals=100, lammy=1):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.lammy = lammy

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.lammy,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

class logRegL2:
    # Logistic Regression
    def __init__(self, verbose=1, maxEvals=100, lammy=1):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.lammy = lammy

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + (self.lammy/2 * w.T.dot(w))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy*w

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)



class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1
        minvalue = np.inf
        index_minimum = -2

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the seected set
                w_new,value = minimize(list(selected_new))
                if value < minvalue:
                    minvalue = value
                    index_minimum = i

                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature
            selected.add(index_minimum)
            minLoss = minvalue

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

class logLinearClassifier:
    def __init__(self, verbose, maxEvals):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1.0 + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1.0 + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))
        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # self.w = np.zeros(d)
            (self.W[i], f) = findMin.findMin(self.funObj, self.W[i], self.maxEvals, X, ytmp, verbose=self.verbose)
                                         
    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)
                                         
class softmaxClassifier:
    def __init__(self, maxEvals):
        self.maxEvals = maxEvals
    
    def funObj(self, w, X, y):
        # Calculate the function value

        w= w.reshape((self.n_classes, X.shape[1]))

        d_num = np.zeros((X.shape[0]))
        for i in range(X.shape[0]):
            for c in range(self.n_classes):
                d_num[i] += np.exp(w[c].T.dot(X[i]))

        f=0
        for i in range(X.shape[0]):
            f += -w[y[i]].T.dot(X[i]) + np.log(d_num[i])

        # Calculate the gradient value

        grad = np.zeros((self.n_classes,X.shape[1]))

        for c in range(self.n_classes):
            for j in range(X.shape[1]):
                for i in range(X.shape[0]):
                        grad[c,j] += X[i,j] * (np.exp(w[c].T.dot(X[i])) / d_num[i] - (y[i] == c))

        grad = grad.flatten()
        return f, grad

    def fit(self, X, y):
        n, d = X.shape    
        self.n_classes = np.unique(y).size
        self.w = np.zeros((self.n_classes,d))
        self.w = self.w.flatten()
        # Initial guess

        # utils.check_gradient(self, X, y)
        (self.w, _) = findMin.findMin(self.funObj, self.w,
                                          self.maxEvals, X, y)

    def predict(self, X):
        self.w= self.w.reshape((self.n_classes, X.shape[1]))
        return np.argmax(X@self.w.T, axis=1)   
