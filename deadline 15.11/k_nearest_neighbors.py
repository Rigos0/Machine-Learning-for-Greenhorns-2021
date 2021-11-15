#!/usr/bin/env python3
import argparse
import os
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.metrics

from collections import Counter



class MNIST:
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
parser.add_argument("--k", default=3, type=int, help="K nearest neighbors to consider")
parser.add_argument("--p", default=2, type=int, help="Use L_p as distance metric")
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=100, type=int, help="Train set size")
parser.add_argument("--weights", default="uniform", type=str, help="Weighting to use (uniform/inverse/softmax)")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> float:
    # Load MNIST data, scale it to [0, 1] and split it to train and test.
    mnist = MNIST(data_size=args.train_size + args.test_size)
    mnist.data = sklearn.preprocessing.MinMaxScaler().fit_transform(mnist.data)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        mnist.data, mnist.target, test_size=args.test_size, random_state=args.seed)

    OHE = sklearn.preprocessing.OneHotEncoder(sparse=False)
    encoded_train_target = OHE.fit_transform(train_target.reshape(-1, 1))

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    class NearestNeighbors:
        """This class always holds the information about the current K closest
        neighbors during the fitting"""
        def __init__(self, k, p):
            self.k = k
            self.p = p

        """Finds the most common element in given list, which is the prediction"""
        @staticmethod
        def get_prediction(targets, encoded,  weights):
            weights = np.asarray(weights)
            list_ = [0,0,0,0,0,0,0,0,0,0]
            if args.weights=="uniform":
                for t in targets:
                    list_[t] += 1

                highest = -999
                index= 999
                for i in range(len(list_)):
                    if list_[i] > highest:
                        index = i
                        highest = list_[i]

                return index


                return data.most_common(1)[0][0]

            elif args.weights=="inverse":
                weights = 1/weights
            elif args.weights=="softmax":
                weights = softmax(-1*weights)

            distances_sum = np.sum(weights)

            s = np.zeros(encoded[0].shape)

            for i, w in enumerate(weights):
                s += (w / distances_sum) * encoded[i]

            return np.argmax(s)

        def find_targets_of_k_smallest_distances(self, distances_vector):
            x = np.argpartition(distances_vector, self.k)
            # First k elements in x are the indices of the k closest elements from train_data
            """Sort the first k elements """
            k_distances = []
            indices = []
            for i in range(self.k):
                indices.append(x[i])
                k_distances.append(distances_vector[x[i]])
            sorted_distances, sorted_indices = (list(t) for t in zip(*sorted(zip(k_distances, indices))))

            targets = []
            encoded_targets = []
            for index in sorted_indices:
                targ = train_target[index]
                targets.append(targ)
                encoded_targets.append(encoded_train_target[index])

            return targets, np.asarray(encoded_targets), sorted_indices, sorted_distances

        def L_p(self, test, train):
            """Calculates the L_p norm of one test data value and all train data"""
            """returns a vector with distances from each train data value"""
            x = np.linalg.norm(train-test, self.p, axis=1)
            return x

    # TODO: Generate `test_predictions` with classes predicted for `test_data`.
    # Find `args.k` nearest neighbors, choosing the ones with smallest train_data
    # indices in case of ties. Use the most frequent class (optionally weighted
    # by a given scheme described below) as prediction, choosing the one with the
    # smallest class index when there are multiple classes with the same frequency.
    #
    # Use L_p norm for a given p (either 1, 2 or 3) to measure distances.
    #
    # The weighting can be:
    # - "uniform": all nearest neighbors have the same weight
    # - "inverse": `1/distances` is used as weights
    # - "softmax": `softmax(-distances)` is used as weights
    #
    # If you want to plot misclassified examples, you need to also fill `test_neighbors`
    # with indices of nearest neighbors; but it is not needed for passing in ReCodEx.

    nearest_neighbors = NearestNeighbors(args.k, args.p)
    test_neighbors = []
    test_predictions = []
    """for every test instance, find the distances from every train instance"""
    for jedno_test_dato in test_data:
        distances_vector = nearest_neighbors.L_p(jedno_test_dato, train_data)
        targets, enc, indices, distances = nearest_neighbors.find_targets_of_k_smallest_distances(distances_vector)
        # get the prediction for the current jedno_test_dato and append it to predictions
        prediction = nearest_neighbors.get_prediction(targets, enc, distances)
        test_predictions.append(prediction)
        test_neighbors.append(indices)

    accuracy = sklearn.metrics.accuracy_score(test_predictions, test_target)

    if args.plot:
        import matplotlib.pyplot as plt
        examples = [[] for _ in range(10)]
        for i in range(len(test_predictions)):
            if test_predictions[i] != test_target[i] and not examples[test_target[i]]:
                examples[test_target[i]] = [test_data[i], *train_data[test_neighbors[i]]]
        examples = [[img.reshape(28, 28) for img in example] for example in examples if example]
        examples = [[example[0]] + [np.zeros_like(example[0])] + example[1:] for example in examples]
        plt.imshow(np.concatenate([np.concatenate(example, axis=1) for example in examples], axis=0), cmap="gray")
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("K-nn accuracy for {} nearest neighbors, L_{} metric, {} weights: {:.2f}%".format(
        args.k, args.p, args.weights, 100 * accuracy))