
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0
        
        # YOUR CODE HERE FOR Q1.1.1

        dataset = ratings['item'].iloc[0]
        
        maxRatings = 0
        curRatings = 0 
        for item,rating in zip(ratings.item, ratings.rating):
            if (item == dataset):
                 curRatings = curRatings + rating
                 dataset = item
            else: 
               if (curRatings > maxRatings):
                   maxRatings = curRatings
                   maxRatingItem = dataset
               curRatings = 0
               curRatings = rating + curRatings
               dataset = item
        
        print(maxRatings)
        print(maxRatingItem)
        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))

        # YOUR CODE HERE FOR Q1.1.2

        print(ratings.user.mode())
        print(n)
        print("\n")
        print(ratings['user'].value_counts())
        print(ratings['rating'].value_counts())
        print(ratings['item'].value_counts())


        # YOUR CODE HERE FOR Q1.1.3

        plt.hist(ratings['rating'],bins=10)
        #plt.yscale('log',nonposy='clip')
        plt.title("Number of Ratings VS Stars of Ratings")
        fname = os.path.join("..", "figs", "histRatings.png")
        plt.savefig(fname)

        ratings.rating *=1/n
        #print(ratings['rating'])
        #print(ratings['rating'].value_counts())
        plt.hist(ratings['rating'],bins=10)
        plt.yscale('log',nonposy='clip')
        plt.title("Number of Ratings per user")
        fname = os.path.join("..", "figs", "histRatingsPerUser.png")
        plt.savefig(fname)

        ratings.rating *=1/d
        #print(ratings['rating'])
        #print(ratings['rating'].value_counts())
        plt.hist(ratings['rating'],bins=10)
        plt.yscale('log',nonposy='clip')
        plt.title("Number of Ratings per item")
        fname = os.path.join("..", "figs", "histRatingsPerItem.png")
        plt.savefig(fname)




    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]

        print(url_amazon % grill_brush)

        # YOUR CODE HERE FOR Q1.2



        # YOUR CODE HERE FOR Q1.3


    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        model = linear_model.WeightedLeastSquares()
        model.fit(X,y,1)
        print(model.w)

        XSample = np.linspace(np.min(X), np.max(X), 1000)[:,None]
        yhat = model.predict(XSample)
        utils.test_and_plot(model,X,y,title="Weighted Least Squares", filename="Weighted_least_squares.png")

        # YOUR CODE HERE

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']
        rows,columns = X.shape
        test = Xtest.shape[0]
        model = linear_model.LeastSquaresBias()
        model.fit(X,y)
        
        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, bias",filename="least_squares_bias.pdf")


        # YOUR CODE HERE

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']
        rows, columns = X.shape
        test = Xtest.shape[0]

        for p in range(11):
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)
            yhat = model.predict(X)
            print("p=%d" % p)
            utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, polynomial",filename="least_squares_polynomial.pdf")

            # YOUR CODE HERE

    else:
        print("Unknown question: %s" % question)

