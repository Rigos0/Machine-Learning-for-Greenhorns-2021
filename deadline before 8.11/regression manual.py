#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the Diabetes dataset
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.
    # TODO: Append a new feature to all input data, with value "1"
    dataset.data = np.c_[dataset.data, np.ones(442)]

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(dataset.data,
                                                                                dataset.target,
                                                                                test_size=args.test_size,
                                                                                random_state=args.seed)

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    Xt = np.transpose(X_train)
    Xt_X = (np.dot(Xt, X_train))
    X_inv = np.linalg.inv(Xt_X)

    X_inv_Xt = np.dot(X_inv, Xt)
    weights = np.dot(X_inv_Xt, Y_train)

    # TODO: Predict target values on the test set.

    target = np.dot(X_test, weights)


    # TODO: Compute root mean square error on the test set predictions.
    sum = 0
    in_test = np.shape(Y_test)[0]
    for i in range(in_test):
        diff_squared = (target[i] - Y_test[i])**2
        sum += diff_squared
    MSE = sum/in_test
    rmse = MSE**0.5

    return rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))