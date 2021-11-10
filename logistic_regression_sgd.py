#!/usr/bin/env python3
import argparse


import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import math


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=95, type=int, help="Data size")
parser.add_argument("--epochs", default=9, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.7, type=float, help="Learning rate")
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=45, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical classification dataset
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data
    ones = np.ones((data.shape[0], 1))
    data = np.concatenate((data, ones), axis=1)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = \
        sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial logistic regression weights
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))


    def is_it_right_otaznik(probability_1, target_value):
        difference = abs(target_value - probability_1)

        if difference <= 0.5:
            return 1
        else:
            return 0

    def get_loss_accuracy(data, target_data, weights):
        number_of_correctly_classified = 0
        loss_sum = 0

        predictions = np.dot(data, weights)
        for i, p in enumerate(predictions):
            prob_pred = sigmoid(p)
            number_of_correctly_classified += is_it_right_otaznik(prob_pred, target_data[i])

            loss = target_data[i] * math.log(prob_pred, math.e) + (1 - target_data[i]) * math.log(1 - prob_pred, math.e)
            loss_sum += loss

            # number of weight updates
        loss = - loss_sum / data.shape[0]
        accuracy = number_of_correctly_classified / data.shape[0]

        return loss, accuracy

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        gradient_sum = 0

        for counter, i in enumerate(permutation):
            data_point = train_data[i]
            target_value = train_target[i]

            linear_prediction = np.dot(data_point, weights)
            probability_1 = sigmoid(linear_prediction)

            gradient = np.dot((probability_1 - target_value), data_point)
            gradient_sum += gradient

            # for every batch, update the weights
            if ((counter+1) % args.batch_size) == 0:
                average_gradient = gradient_sum / args.batch_size
                # update weights
                weights -= args.learning_rate * average_gradient
                gradient_sum = 0

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train set and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.
        train_loss, train_accuracy = get_loss_accuracy(train_data, train_target, weights)
        test_loss, test_accuracy = get_loss_accuracy(test_data, test_target, weights)


        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                if not epoch: plt.figure(figsize=(6.4*3, 4.8*(args.epochs+2)//3))
                plt.subplot(3, (args.epochs+2)//3, 1 + epoch)
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=21, cmap=plt.cm.RdBu, alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="P", label="train", cmap=plt.cm.RdBu)
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap=plt.cm.RdBu)
            plt.legend(loc="upper right")
            if args.plot is True: plt.show()
            else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, [(train_loss, train_accuracy), (test_loss, test_accuracy)]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))