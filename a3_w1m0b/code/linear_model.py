import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        outliers = np.full((400,), 1)
        dataPoints = np.full((100,), .1)
        linear = np.concatenate((outliers,dataPoints), axis=0)
        V = np.diag(linear)
        multiT = np.dot(X.T, V)
        multiXw = np.dot(multiT,X)
        multiXwIn = np.linalg.inv(multiXw)
        finalMulti = np.dot(multiXwIn,multiT)
        self.w = (np.dot(finalMulti,y))
        # raise NotImplementedError()

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''
        # Calculate the function value
        #f = 0.5*np.sum((X@w - y)**2)
        f = np.sum(np.log(np.exp(X*w-y)+ np.exp(y-X*w)), axis=0)

        # Calculate the gradient value
        #g = X.T@(X@w-y)
        g = np.sum(X.T.dot((np.exp(X.dot(w)-y)-np.exp(y-X.dot(w)))/(np.exp(X.dot(w)-y)+np.exp(y-X.dot(w)))),axis=0)

        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        ''' YOUR CODE HERE '''
        numberRows, numberColumns = X.shape
        rows = numberRows
        ones = np.ones((numberRows,1))
        addones = np.concatenate((X,ones), axis=1)
        self.w = solve((np.dot(addones.T, addones)),(np.dot(addones.T, y)))
        w = self.w
        #raise NotImplementedError()

    def predict(self, X):
        ''' YOUR CODE HERE '''
        w = self.w
        row, col = X.shape
        ones = np.ones((row,1))
        addones = np.concatenate((X,ones), axis=1)
        y = np.dot(addones,w)
        return y
        #raise NotImplementedError()

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        ''' YOUR CODE HERE '''
        basis = self.__polyBasis(X)
        self.w = solve((np.dot(basis.T,basis)),(np.dot(basis.T,y)))

       # raise NotImplementedError()

    def predict(self, X):
        ''' YOUR CODE HERE '''
        w = self.w
        basis = self.__polyBasis(X)
        yhat = np.dot(basis,w)
        return yhat

       # raise NotImplementedError()

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        ''' YOUR CODE HERE '''
        rows = X.shape[0]
        columns = self.p + 1
        ones = np.ones((rows,columns))
        for i in range (self.p):
            ones[:,i+1]=(X**(i+1))[:,0]
        return ones

       # raise NotImplementedError()
