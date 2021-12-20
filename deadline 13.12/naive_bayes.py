#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="multinomial", type=str, help="NB type to use")
parser.add_argument("--classes", default=3, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> float:
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.

    def get_empty_matrix():
        return np.zeros((train_data.shape)[1])

    def gaussian_mean_and_variance(args, X_train, y_train):
        means_for_each_class = []
        variances_for_each_class = []
        priors = []
        for class_index in range(args.classes):
            # calculate the number of train data in given class
            N_k = np.count_nonzero(y_train == class_index)
            priors.append(N_k/len(y_train))

            # get the mean of all train data with the same class
            data_sum = get_empty_matrix()
            for train_data_index, train_instance in enumerate(X_train):
                if y_train[train_data_index] == class_index:
                    data_sum += X_train[train_data_index]
            mean = data_sum / N_k
            means_for_each_class.append(mean)

            # get the variances
            variances_sum = get_empty_matrix()
            for train_data_index, train_instance in enumerate(X_train):
                if y_train[train_data_index] == class_index:
                    variances_sum += (X_train[train_data_index] - mean)**2
            variance = variances_sum / N_k

            # smoothing
            variance += args.alpha
            variances_for_each_class.append(variance)
        return means_for_each_class, variances_for_each_class, priors

    #   During prediction, you can compute probability density function of a Gaussian
    #   distribution using `scipy.stats.norm`, which offers `pdf` and `logpdf`
    #   methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.

    def gaussian_predict(args, test_data, means, variances, prior):
        predictions = []
        prior = np.asarray(prior)

        for dato in test_data:
            p_for_all_classes = []
            for c_i in range(args.classes):
                stat = scipy.stats.norm(means[c_i], np.sqrt(variances[c_i]))
                p_for_all_classes.append(np.prod(stat.pdf(dato))*prior[c_i])

            predictions.append(np.argmax(p_for_all_classes))

        return predictions

    def binarize_vector(vector):
        new = vector.copy()
        new[new < 8] = 0
        new[new >= 8] = 1
        return new

    def bernoulli_probabilities(args, X_train, y_train):
        probabilities_for_each_class = []
        priors = []
        for class_index in range(args.classes):
            # calculate the number of train data in given class
            N_k = np.count_nonzero(y_train == class_index)
            priors.append(N_k / len(y_train))

            # get the probs of all train data with the same class
            data_sum = get_empty_matrix()
            for t_i, train_instance in enumerate(X_train):
                if y_train[t_i] == class_index:
                    binary = binarize_vector(train_instance)
                    data_sum += binary

            # smoothing
            prob = (data_sum + args.alpha) / (N_k + 2*args.alpha)

            probabilities_for_each_class.append(prob)

        return probabilities_for_each_class, priors

    def bernoulli_predict(args, test_data, p, prior):
        predictions = []
        prior = np.asarray(prior)

        for dato in test_data:
            x = binarize_vector(dato)
            p_for_all_classes = []
            for c_i in range(args.classes):
                prob = np.log(prior[c_i]) + np.sum(x*np.log(p[c_i]/(1-p[c_i])) + np.log(1-p[c_i]))
                p_for_all_classes.append(prob)

            predictions.append(np.argmax(p_for_all_classes))

        return predictions

    def multinomial_probabilities(args, X_train, y_train):
        probabilities_for_each_class = []
        priors = []
        for class_index in range(args.classes):
            # calculate the number of train data in given class
            N_k = np.count_nonzero(y_train == class_index)
            priors.append(N_k / len(y_train))

            # get the probs of all train data with the same class
            data_sum = get_empty_matrix()
            lambda_ = 0
            for t_i, train_instance in enumerate(X_train):
                if y_train[t_i] == class_index:
                    data_sum += train_instance
                    lambda_ += np.sum(train_instance)

            # smoothing                                            # the number of features
            prob = (data_sum + args.alpha) / (lambda_ + args.alpha * X_train.shape[1])

            probabilities_for_each_class.append(prob)

        return probabilities_for_each_class, priors

    def multinomial_predict(args, test_data, p, prior):
        predictions = []
        prior = np.asarray(prior)

        for dato in test_data:
            p_for_all_classes = []
            for c_i in range(args.classes):
                prob = np.log(prior[c_i]) + np.sum(dato * np.log(p[c_i]))
                p_for_all_classes.append(prob)

            predictions.append(np.argmax(p_for_all_classes))

        return predictions

    def naive_bayes(args, X_train, y_train):
        if args.naive_bayes_type == "gaussian":
            gaussian_mean_and_variance(args, X_train, y_train)
            means, variances, priors = gaussian_mean_and_variance(args, X_train, y_train)
            preds = gaussian_predict(args, test_data, means, variances, priors)

        elif args.naive_bayes_type == "bernoulli":
            probabilities, priors = bernoulli_probabilities(args, X_train, y_train)
            preds = bernoulli_predict(args, test_data, probabilities, priors)

        elif args.naive_bayes_type == "multinomial":
            probabilities, priors = multinomial_probabilities(args, X_train, y_train)
            preds = multinomial_predict(args, test_data, probabilities, priors)

        from sklearn.metrics import accuracy_score
        test_accuracy = accuracy_score(preds, test_target)

        return test_accuracy
    # TODO: Predict the test data classes and compute test accuracy.

    return naive_bayes(args, train_data, train_target)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))