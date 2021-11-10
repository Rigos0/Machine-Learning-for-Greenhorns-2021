#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import math

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    OHE = sklearn.preprocessing.OneHotEncoder(sparse=False)
    encoded_train_target = OHE.fit_transform(train_target.reshape(-1, 1))

    # Generate initial model weights
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    # Takes and return np array
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        gradient_sum = np.zeros(weights.shape)
        for counter, i in enumerate(permutation):
            data_point = train_data[i]
            target_value_encoded = encoded_train_target[i]
            raw_prediction = np.dot(data_point, weights)

            # Note that you need to be careful when computing softmax, because the exponentiation
            # in softmax can easily overflow. To avoid it, you should use the fact that
            # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
            # That way we only exponentiate values which are non-positive, and overflow does not occur.

            """using softmax(z) = softmax(z - maximum_of_z) to avoid overflows"""
            maximum = np.max(raw_prediction)
            softmaxed = softmax(raw_prediction - maximum)
            b = softmaxed - target_value_encoded

            gradient = np.dot(data_point.reshape(-1,1), b.reshape(1,-1))
            gradient_sum += gradient

            """ for every batch, update the weights """
            if ((counter + 1) % args.batch_size) == 0:
                average_gradient = gradient_sum / args.batch_size

                weights -= args.learning_rate * average_gradient
                gradient_sum = np.zeros(weights.shape)

        def is_it_right_otaznik(probability_1, target_value):
            max_index = np.argmax(probability_1)

            if max_index == target_value:
                return 1
            else:
                return 0

        def get_loss_accuracy(data, target_data, weights):
            number_of_correctly_classified = 0
            loss_sum = 0

            predictions = np.dot(data, weights)
            for i, p in enumerate(predictions):
                prob_pred = softmax(p)
                number_of_correctly_classified += is_it_right_otaznik(prob_pred, target_data[i])

                target_prob = prob_pred[target_data[i]]
                loss = math.log(target_prob, math.e)
                loss_sum += loss

                # number of weight updates
            loss = - loss_sum / data.shape[0]
            accuracy = number_of_correctly_classified / data.shape[0]

            return loss, accuracy

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.

        train_loss, train_accuracy = get_loss_accuracy(train_data, train_target, weights)
        test_loss, test_accuracy = get_loss_accuracy(test_data, test_target, weights)


        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))


    return weights, [(train_loss, train_accuracy), (test_loss, test_accuracy)]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:", *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")