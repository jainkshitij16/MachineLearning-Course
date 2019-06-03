# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        training_error_array = []
        testing_error_array = []

        for n in range(1,16):
            model = DecisionTreeClassifier(criterion='entropy', max_depth=n)
            model.fit(X,y)
            predicted_y = model.predict(X)
            training_error = np.mean(predicted_y != y)
            training_error_array.append(training_error)

            predicted_y = model.predict(X_test)
            testing_error = np.mean(predicted_y != y_test)
            testing_error_array.append(testing_error)
        
        plt.plot(training_error_array)
        plt.plot(testing_error_array)
        plt.show()

    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape



    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

    

    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        model = KNN(1)
        #model = KNN(3)
        #model = KNN(10)

        model.fit(X,y)
        
        y_training = model.predict(X)
        y_testing = model.predict(Xtest)
        training_error =  np.mean(y_training != y)
        testing_error =   np.mean(y_testing != ytest)
       
        print(training_error)
        print(testing_error)
 
        utils.plotClassifier(model, Xtest, ytest)

        fname = os.path.join("..", "figs", "q3.3_KNN.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        model = KNeighborsClassifier(1)
        model.fit(X, y)
        y_pred = model.predict(X)
        training_error = np.mean(y_pred != y)
        utils.plotClassifier(model,X, y)

        fname = os.path.join("..", "figs", "q3.3_KNeighbors.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        print("Random Tree")
        evaluate_model(RandomTree(np.inf))
        print("Random Forest 50 trees")
        evaluate_model(RandomForest(50,np.inf))
        print("Random Forest classifier")
        evaluate_model(RandomForestClassifier())
        
        



    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        error_value = model.error(X)
        #print(error_value)
        print(model.error(X))
        #print("\nError for the dataset = " % error_value)

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        #Finds the minimum error in kmeans after running 50 times
        min_error = 123456123
        model_min = None
        for i in range(0,50):
            model = Kmeans(4)
            model.fit(X)
            value_error = model.error(X)
            if value_error < min_error:
                plt.scatter(X[:,0], X[:,1], c = model.predict(X))
                model_min = model
                min_error = value_error
        plt.scatter(X[:,0], X[:,1], c = model_min.predict(X))
        fname = "../figs/Kmeans(50).png"
        plt.savefig(fname)
        print("Figure saved as ", fname)
        print(min_error)
        plt.show()


    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']



    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']

        min_error = 123456123
        model_min = None
        track_error = np.zeros(10)
        for j in range(1,11):
            for i in range(0,50):
                model = Kmeans(k=j)
                model.fit(X)
                error_val = model.error(X)
                if error_val < min_error:
                    model_min = model
                    min_error = error_val
            
            track_error[j-1] = min_error
            min_error = 123456123
        
        k_val = np.arange(1,11)
        
        plt.plot(k_val, track_error, label="Minimum error", linestyle='-', marker='o')
        plt.xlabel("k")
        plt.ylabel("Minimum Error")
        plt.xticks(np.arange(min(k_val), max(k_val)+1, 1.0))
        plt.legend()
        fname = "../figs/KValueMinimumError.png"
        plt.savefig(fname)
        print("Figure saved as ", fname)
        plt.show()


    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=15, min_samples=30)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))
        
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        plt.xlim(-25,25)
        plt.ylim(-15,30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)
        
    else:
        print("Unknown question: %s" % question)
