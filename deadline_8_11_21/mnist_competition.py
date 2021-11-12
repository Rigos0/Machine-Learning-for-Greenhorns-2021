#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request
import sklearn
from sklearn.neural_network import MLPClassifier
import sklearn.pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
import numpy as np

class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")


def is_it_right_otaznik(probability_1, target_value):
    max_index = np.argmax(probability_1)

    if max_index == target_value:
        return 1
    else:
        return 0

def get_accuracy(data, target_data, model):
    number_of_correctly_classified = 0
    predictions = model.predict(data)

    for i, p in enumerate(predictions):
        number_of_correctly_classified += is_it_right_otaznik(p, target_data[i])

        # number of weight updates
    accuracy = number_of_correctly_classified / data.shape[0]

    return accuracy

def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        train_data, test_data, train_target, test_target = \
            sklearn.model_selection.train_test_split(train.data, train.target, test_size=0.20)


        # TODO: Train a model on the given dataset and store it in `model`.
        pipe = sklearn.pipeline.Pipeline(steps=[
            ('scaler', StandardScaler()),
            ("logistic_regression", MLPClassifier(verbose=True, solver='adam',
                                                  hidden_layer_sizes=(1000), learning_rate_init = 0.01))])

        model = pipe
        model.fit(train_data, train_target)

        test_accuracy = get_accuracy(test_data, test_target, model)
        print(test_accuracy)


        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLPClassifier is in `mlp` variable.
        mlp = model

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