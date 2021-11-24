#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import pylab

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=13, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def get_rmse(predictions, true_data, weights, l):
    sum_of_diffs = 0
    N = len(predictions)
    for i in range(N):
        sum_of_diffs += (predictions[i]-true_data[i])**2

    wei = 0
    for w in weights:
        wei+= w**2
    wei **= 0.5

    norm_squared = wei**2
    return sum_of_diffs/N

def main(args: argparse.Namespace):
    # Load the Diabetes dataset
    dataset = sklearn.datasets.load_diabetes()

    train_data, test_data, train_target, test_target = \
        sklearn.model_selection.train_test_split(dataset.data, dataset.target,
                                                 test_size=args.test_size, random_state=args.seed)

    lambdas = np.geomspace(0.01, 10, num=500)
    # TODO: Using `sklearn.linear_model.Ridge`, fit the train set using
    # L2 regularization, employing above defined lambdas.
    # For every model, compute the root mean squared error and return the
    # lambda producing lowest RMSE and the corresponding RMSE.
    rmses = []
    smallest_rmse = 99999
    index = 999
    for i, l in enumerate(lambdas):
        model = sklearn.linear_model.Ridge(alpha=l).fit(train_data, train_target)
        predictions = model.predict(test_data)
        mse = get_rmse(predictions, test_target, model.coef_, l)
        rmse = mse ** 0.5
        if rmse < smallest_rmse:
            smallest_rmse = rmse
            index = i
        rmses.append(rmse)

    best_lambda = lambdas[index]
    best_rmse = smallest_rmse

    if args.plot:
        # This block is not required to pass in ReCodEx, however, it is useful
        # to learn to visualize the results.

        # If you collect the respective results for `lambdas` to an array called `rmses`,
        # the following lines will plot the result if you add `--plot` argument.
        import matplotlib.pyplot as plt
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength")
        plt.ylabel("RMSE")
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))