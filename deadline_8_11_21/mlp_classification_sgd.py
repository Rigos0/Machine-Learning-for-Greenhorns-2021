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
parser.add_argument("--epochs", default=3, type=int, help="Number of SGD training epochs")
parser.add_argument("--hidden_layer", default=20, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    OHE = sklearn.preprocessing.OneHotEncoder(sparse=False)
    encoded_train_target = OHE.fit_transform(train_target.reshape(-1, 1))

    # Generate initial model weights
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]

    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    # Takes and returns np array
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    # Takes and returns np array
    def relu(x):
        return np.maximum(x, 0)

    def is_it_right_otaznik(probability_1, target_value):
        max_index = np.argmax(probability_1)

        if max_index == target_value:
            return 1
        else:
            return 0

    def get_accuracy(data, target_data, weights, biases):
        number_of_correctly_classified = 0
        _, predictions = forward(data, biases, weights)
        for i, p in enumerate(predictions):
            number_of_correctly_classified += is_it_right_otaznik(p, target_data[i])

            # number of weight updates
        accuracy = number_of_correctly_classified / data.shape[0]

        return accuracy

    def forward(input_data, biases, weights):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.

        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where ReLU(x) = max(x, 0), and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as ReLU(inputs @ weights[0] + biases[0]).
        # The value of the output layer is computed as softmax(hidden_layer @ weights[1] + biases[1]).
        hidden_layer = relu(input_data @ weights[0] + biases[0])

        output_layer = hidden_layer @ weights[1] + biases[1]
        maximum = np.max(output_layer)
        output_layer = softmax(output_layer - maximum)
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.
        return hidden_layer, output_layer

    gradient_sum = [np.zeros(weights[0].shape), np.zeros(weights[1].shape)]
    biases_sum = [np.zeros(biases[0].shape), np.zeros(biases[1].shape)]

    def relu_derivative(x):
        x = (x > 0) * 1
        return x

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        for counter, i in enumerate(permutation):
            data_point = train_data[i]
            hidden_layer, output_layer = forward(data_point, biases, weights)
            target_value_encoded = encoded_train_target[i]

            L__y_in = output_layer - target_value_encoded
            L__bias = L__y_in

            L__w_y = hidden_layer.reshape(-1, 1) @ L__y_in.reshape(1, -1)

            L__h = weights[1] @ L__y_in.reshape(-1, 1)
            h__h_in = relu_derivative(hidden_layer)
            L__h_in = L__h * h__h_in.reshape(-1, 1)

            L__w_h = data_point.reshape(-1, 1) @ L__h_in.reshape(1, -1)

            biases_sum[1] += L__bias
            gradient_sum[0] += L__w_h
            gradient_sum[1] += L__w_y
            biases_sum[0] += L__h_in[0]

            # for every batch, update the weights
            if ((counter + 1) % args.batch_size) == 0:
                """very clean code, google please hire me"""
                average_gradient_0 = gradient_sum[0] / args.batch_size
                average_gradient_1 = gradient_sum[1] / args.batch_size
                b_0 = biases_sum[0] / args.batch_size
                b_1 = biases_sum[1] / args.batch_size

                weights[0] -= args.learning_rate * average_gradient_0
                weights[1] -= args.learning_rate * average_gradient_1
                biases[0] -= args.learning_rate * b_0
                biases[1] -= args.learning_rate * b_1

                gradient_sum[0] = np.zeros(weights[0].shape)
                gradient_sum[1] = np.zeros(weights[1].shape)
                biases_sum = [np.zeros(biases[0].shape), np.zeros(biases[1].shape)]

        # The gradient used in SGD has now four parts, gradient of weights[0] and weights[1]
        # and gradient of biases[0] and biases[1].
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of -log P(target | data), or you can compute
        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer
        # - compute the derivative with respect to weights[1] and biases[1]
        # - compute the derivative with respect to the hidden layer output
        # - compute the derivative with respect to the hidden layer input
        # - compute the derivative with respect to weights[0] and biases[0]

        # TODO: After the SGD epoch, measure the accuracy for both the
        # train test and the test set.

        _, y_pred = forward(train_data, biases, weights)
        train_accuracy = np.mean(train_target == np.argmax(y_pred, axis=-1))
        _, test_pred = forward(test_data, biases, weights)
        test_accuracy = np.mean(test_target == np.argmax(test_pred, axis=-1))

        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [train_accuracy, test_accuracy]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(args)
    print("Learned parameters:", *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:20]] + ["..."]) for ws in parameters), sep="\n")