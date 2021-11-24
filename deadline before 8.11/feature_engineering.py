#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.compose import ColumnTransformer

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def isinteger(arr):
    return np.equal(np.mod(arr, 1), 0)


def get_columns_with_features(data):
    for i in range(len(data[0])):
        for x in range(len(data[0])):
            if x >= i:
                result = np.multiply(data[:, i], data[:, x])
                yield result.reshape((-1, 1))

def process(data, training, tranformer = None):
    categorical_columns_indeces = []
    real_columns_indeces = []
    "For every column, find if the column is categorical or real"
    for i in range(len(data[0])):
        column = data[:, i]
        is_int_arr = isinteger(column)
        if is_categorical(is_int_arr):
            categorical_columns_indeces.append(i)

        else:
            real_columns_indeces.append(i)

    cat_transformer = ColumnTransformer([("cat", sklearn.preprocessing.OneHotEncoder(
        sparse=False, handle_unknown="ignore"), categorical_columns_indeces)])

    normaliser_transformer = ColumnTransformer(transformers=[("norm", sklearn.preprocessing.StandardScaler(),
                                                              real_columns_indeces)])
    if not training:
        categorised = cat_transformer.fit_transform(data)
        normalised = normaliser_transformer.fit_transform(data)

        categorised = np.concatenate((categorised, normalised), axis=1)

        for column in get_columns_with_features(normalised):
            categorised = np.concatenate((categorised, column), axis=1)

        return categorised, normaliser_transformer
    else:

        normaliser_transformer = tranformer
        # categorised = cat_transformer.transform(data)
        normalised = normaliser_transformer.transform(data)

        # categorised = np.concatenate((categorised, normalised), axis=1)

        for column in get_columns_with_features(normalised):
            normalised = np.concatenate((normalised, column), axis=1)

        return normalised



def is_categorical(arr):
    for boolean in arr:
        if not boolean:
            return False
    return True


def main(args: argparse.Namespace):
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data = \
        sklearn.model_selection.train_test_split(dataset.data, test_size=args.test_size, random_state=args.seed)


    # TODO: Process the input columns in the following way:
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general, integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of an exercise). Encode the values with one-hot encoding
    #   using `sklearn.preprocessing.OneHotEncoder` (note that its output is by
    #   default sparse, you can use `sparse=False` to generate dense output;
    #   also use `handle_unknown="ignore"` to ignore missing values in test set).

    train_data, trans = process(train_data, False)
    test_data = process(test_data, True, tranformer=trans)

    #
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; use `sklearn.preprocessing.StandardScaler`.
    #
    # In the output, first there should be all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.

    # TODO: To the current features, append polynomial features of order 2.
    # If the input values are [a, b, c, d], you should append
    # [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]. You can generate such polynomial
    # features either manually, or you can generate them with
    # `sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)`.

    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.

    # TODO: Fit the feature processing steps on the training data.
    # Then transform the training data into `train_data` (you can do both these
    # steps using `fit_transform`), and transform testing data to `test_data`.

    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))))