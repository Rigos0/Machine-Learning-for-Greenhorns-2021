#!/usr/bin/env python3
import argparse
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=2, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=50, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.35, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

def kernel(args, x, y):
    # TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    if args.kernel == "poly":
        return polynomial_kernel(x, y, degree=args.kernel_degree, gamma=args.kernel_gamma)
    else:
        return rbf_kernel(x, y, gamma=args.kernel_gamma)

def get_sum(kernel, index_radku, a, train_target):
    sum_s = np.sum(a * kernel[index_radku] * train_target)
    return sum_s

def get_sum_predict(kernel, index_radku, weights):
    sum_s = np.sum(weights * kernel[index_radku])
    return sum_s

def smo(args, train_data, train_target, test_data, test_target):
    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)
    train_kernel = kernel(args, train_data, train_data)
    test_kernel = kernel(args, test_data, train_data)
    tol = args.tolerance
    C = args.C

    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i #lol, clean
            j = j + (j >= i)

            # TODO: Check that a[i] fulfils the KKT conditions, using `args.tolerance` during comparisons.
            E_i = get_sum(train_kernel, i, a, train_target)+b - train_target[i]
            if (a[i] < C - tol and train_target[i]*E_i < -tol) or (a[i] > tol and train_target[i]*E_i > tol):
                # If the conditions do not hold, then
                # - compute the updated unclipped a_j^new.
                E_j = get_sum(train_kernel, j, a, train_target)+b - train_target[j]
                second_derivative_a_j = 2 * train_kernel[i][j] - train_kernel[i][i] - train_kernel[j][j]
                a_j_new = a[j] - train_target[j] * ((E_i - E_j) / second_derivative_a_j)

            #   If the second derivative of the loss with respect to a[j]
            #   is > -`args.tolerance`, do not update a[j] and continue
            #   with next i.
                # protoze bychom delili nulou
                if second_derivative_a_j > -tol:
                    continue

            # - clip the a_j^new to suitable [L, H].
                if train_target[i] == train_target[j]:
                    L, H = max((0, a[i]+a[j]-C)), min((C, a[i]+a[j]))
                else:
                    L, H = max((0, a[j]-a[i])), min((C, C+a[j]-a[i]))
                if a_j_new < L:
                    a_j_new = L
                elif a_j_new > H:
                    a_j_new = H

            #   If the clipped updated a_j^new differs from the original a[j]
            #   by less than `args.tolerance`, do not update a[j] and continue
            #   with next i.
                if abs(a_j_new - a[j]) < tol:
                    continue

                # - update a[j] to a_j^new, and compute the updated a[i] and b.
                a_i_new = a[i] - train_target[i]*train_target[j]*(a_j_new-a[j])

                #   During the update of b, compare the a[i] and a[j] to zero by
                #   `> args.tolerance` and to C using `< args.C - args.tolerance`.
                if tol < a_i_new < C - tol:
                    b_new = b - E_i - train_target[i]*(a_i_new-a[i])*train_kernel[i][i]-train_target[j]*\
                            (a_j_new - a[j]) * train_kernel[j][i]
                elif tol < a_j_new < C-tol:
                    b_new = b - E_j - train_target[i]*(a_i_new-a[i])*train_kernel[i][j]-train_target[j]*\
                            (a_j_new - a[j]) * train_kernel[j][j]
                else:
                    # oh god
                    b_new = ((b - E_i - train_target[i] * (a_i_new - a[i]) * train_kernel[i][i] - train_target[j] *
                      (a_j_new - a[j]) * train_kernel[j][i]) + (b - E_j - train_target[i]*(a_i_new-a[i])*
                        train_kernel[i][j]-train_target[j]*(a_j_new - a[j]) * train_kernel[j][j])) / 2
                a[j] = a_j_new
                a[i] = a_i_new
                b = b_new

                # - increase `as_changed`
                as_changed += 1

        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.

        train_predictions = []
        for i in range(len(train_target)):
            raw_pred = get_sum(train_kernel, i, a, train_target) + b
            if raw_pred > 0:
                pred = 1
            else: pred = -1
            train_predictions.append(pred)
        train_accs.append(sklearn.metrics.accuracy_score(train_target, train_predictions))

        test_predictions = []
        for i in range(len(test_target)):
            raw_pred = get_sum(test_kernel, i, a, train_target) + b
            if raw_pred > 0:
                pred = 1
            else:
                pred = -1
            test_predictions.append(pred)
        test_accs.append(sklearn.metrics.accuracy_score(test_target, test_predictions))

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    support_vectors = []
    support_vector_weights = []
    for i, a_but_shorter in enumerate(a):
        if a_but_shorter > tol:
            support_vectors.append(train_data[i])
            support_vector_weights.append(a_but_shorter*train_target[i])

    print("Done, iteration {}, support vectors {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), len(support_vectors), 100 * train_accs[-1], 100 * test_accs[-1]))

    return support_vectors, support_vector_weights, b

def main(args: argparse.Namespace) -> float:
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    def get_data(i, j, train, target):
        new_train_target = []
        new_train_data = []
        for index, dato in enumerate(target):
            if dato == i:
                new_train_target.append(1)
                new_train_data.append(train[index])
            elif dato == j:
                new_train_target.append(-1)
                new_train_data.append(train[index])
        return new_train_data, new_train_target

    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes $i < j$, using the `smo` method.
    #
    # When training a classifier for classes $i < j$:
    # - keep only the training data of these classes, in the same order
    #   as in the input dataset;
    # - use targets 1 for the class $i$ and -1 for the class $j$.
    trained_classifiers = []
    corresponding_classes = []
    for i in range(args.classes):
        for j in range(args.classes):
            if i < j:
                print("Training classes {} and {}".format(i, j))
                X_train, y_train = get_data(i, j, train_data, train_target)
                X_test, y_test = get_data(i, j, test_data, test_target)
                support_vectors, weights, bias = smo(args, X_train, y_train, X_test, y_test)
                trained_classifiers.append([np.asarray(support_vectors), np.asarray(weights), bias])
                corresponding_classes.append([i, j])

    # TODO: Classify the test set by majority voting of all the trained classifiers,
    # using the lowest class index in the case of ties.

    # Note that during prediction, only the support vectors returned by the `smo`
    # should be used, not all training data.
    predictions = np.zeros((len(test_data), args.classes))
    for index, model in enumerate(trained_classifiers):
        # support_vectors = model[0]
        # w = model[1]
        # bias = model[2]
        class_1, class_2 = corresponding_classes[index][0], corresponding_classes[index][1]
        test_kernel = kernel(args, test_data, model[0])

        for i in range(len(test_data)):
            raw_pred = get_sum_predict(test_kernel, i, model[1]) + model[2]
            if raw_pred > 0:
                pred = class_1
            else:
                pred = class_2
            predictions[i][pred] += 1

    argmaxed_preds = []
    for p in predictions:
        argmaxed_preds.append(np.argmax(p))

    # Finally, compute the test set prediction accuracy.
    test_accuracy = sklearn.metrics.accuracy_score(argmaxed_preds, test_target)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))