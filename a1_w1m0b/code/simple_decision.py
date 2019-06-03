import numpy as np
from decision_stump import DecisionStumpErrorRate
from decision_tree import decisiontree

class SimpleDecision(decisiontree)

    def predict(self, X):
        M, D = X.shape
        y = np.zeros(M)

        for vector in range[M]
            if X[vector,0] <=-85: 
                if X[vector,1] <=35:
                    y[vector] = 1

            else:
                if X[vector,1] <=40:
                    y[vector] = 0

        return y