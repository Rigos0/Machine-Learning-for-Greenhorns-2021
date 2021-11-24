#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request
import sklearn
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import numpy as np

class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        data = train.data
        target = train.target


        train_data, test_data, train_target, test_target = \
            sklearn.model_selection.train_test_split(data, target, test_size=0.20)
        #
        # print(train_data.shape)
        # print(test_data.shape)



        # TODO: Train a model on the given dataset and store it in `model`.
        pipe = sklearn.pipeline.Pipeline(steps=[
                                            ('features', PolynomialFeatures()),
                                            ('scaler', StandardScaler()),
                                            ("logistic_regression", LogisticRegression())])

        param_grid = [{'features__degree': [3],
                       'logistic_regression__C': [500],
                       'logistic_regression__max_iter': [100],
                       'logistic_regression__solver': ['lbfgs'],
                       'logistic_regression__penalty': ['l2']}]

        model = sklearn.model_selection.GridSearchCV(pipe, param_grid)

        model.fit(train_data, train_target)

        predictions = model.predict(test_data)
        correctly_classified = 0
        print(model.best_params_)

        for i, p in enumerate(predictions):
            if p == test_target[i]:
                correctly_classified += 1
        test_accuracy = correctly_classified / len(predictions)
        print(test_accuracy)


        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)