#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="entropy", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="digits", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=7, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=None, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace):
    # Use the given dataset
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    classes = np.max(train_target) + 1

    # TODO: Manually create a decision tree on the training data.

    def calculate_criterion(targets_indices, criterion_type="gini"):
        length = len(targets_indices)
        targets = train_target[targets_indices]
        sum_ = 0

        if criterion_type == "gini":
            # for every class
            for c in range(classes):
                class_probability = np.count_nonzero(targets == c) / length
                sum_ += class_probability * (1 - class_probability)

            criterion = length * sum_
        else:
            for c in range(classes):
                class_count = np.count_nonzero(targets == c)
                if class_count != 0:
                    class_probability = class_count / length

                    sum_ += class_probability * np.log(class_probability)
            criterion = -length * sum_

        return criterion

    def get_split_points(f_index, data_indices):
        features = []
        # get the column of corresponding feature
        for i in data_indices:
            features.append(train_data[i][f_index])
        features.sort()

        # find the split point
        split_points = []
        length = len(features)

        for i in range(length):
            if i + 1 == length:
                break
            if features[i] != features[i+1]:
                split_points.append((features[i] + features[i+1])/2)

        return split_points

    def split_by_split_points(f_index, split_points, data_indices):
        for split_value in split_points:
            left = []
            right = []
            for i in data_indices:
                if train_data[i][f_index] < split_value:
                    left.append(i)
                else:
                    right.append(i)

            yield left, right, split_value

    class Tree:
        def __init__(self, root):
            self.root = root

        def build(self, args):
            if args.max_leaves:
                self.split_by_max_leaves(self.root, args.max_leaves, args.min_to_split, args.max_depth)
            else:
                self.recursive_split(self.root, args.min_to_split, args.max_depth)

        # not ideal but I am tired
        def split_by_max_leaves(self, root, max_leaves, min_to_split, max_depth):
            nodes_to_be_splat = []
            corresponding_criterions = []

            left, right = root.split(args)
            leaves = 2

            # initialize the first two leaves
            nodes_to_be_splat.extend([left, right])
            corresponding_criterions.extend((left.try_split(args), right.try_split(args)))

            def remove_from_queue(index):
                corresponding_criterions.pop(index)
                nodes_to_be_splat.pop(index)

            while True:
                # find the node to split
                index_to_split = corresponding_criterions.index(min(corresponding_criterions))

                # split it
                node = nodes_to_be_splat[index_to_split]
                if min_to_split:
                    if len(node.data_indices) < min_to_split:
                        remove_from_queue(index_to_split)
                        continue

                if node.depth == max_depth:
                    remove_from_queue(index_to_split)
                    continue

                node.left = node.fake_splits[0]
                node.right = node.fake_splits[1]# rewrite children
                nodes_to_be_splat.extend((node.left, node.right))
                corresponding_criterions.extend((node.left.try_split(args), node.right.try_split(args)))
                leaves += 1

                # remove the splat node from the queue
                remove_from_queue(index_to_split)

                if leaves == max_leaves:
                    return

        def recursive_split(self, node, min_to_split=None, max_depth=None):
            if min_to_split:
                if len(node.data_indices) < min_to_split:
                    return

            if max_depth:
                if node.depth == max_depth:
                    return

            left, right = node.split(args)
            self.recursive_split(left, min_to_split, max_depth)
            self.recursive_split(right, min_to_split, max_depth)

        def predict(self, x_test):
            predictions = []
            for dato in x_test:
                predictions.append(self.root.predict(dato))
            return predictions

    class Node:
        def __init__(self, indices, depth):
            self.depth = depth
            self.data_indices = indices
            self.classification_feature: int
            self.cut_value: float
            self.criterion = calculate_criterion(self.data_indices, args.criterion)
            self.left = None
            self.right = None

            self.fake_splits = [] # for max leaves approach (ugh, now its not beautiful)

        def find_split(self, args):
            lowest = 99999
            # for each feature, find split points
            for f_index in range(len(train_data[0])):
                split_points = get_split_points(f_index, self.data_indices)
                for left, right, split_value in split_by_split_points(f_index, split_points, self.data_indices):
                    diff = (calculate_criterion(left, args.criterion) + calculate_criterion(right, args.criterion) - self.criterion)
                    if diff < lowest:
                        self.cut_value = split_value
                        self.classification_feature = f_index
                        lowest = diff

        def split(self, args):
            self.find_split(args)
            for new_left, new_right, _ in split_by_split_points(self.classification_feature,
                                                                [self.cut_value], self.data_indices):
                left_kid = Node(new_left, self.depth + 1)
                right_kid = Node(new_right, self.depth + 1)
                self.left = left_kid
                self.right = right_kid

                return left_kid, right_kid

        def predict(self, dato):
            # if leaf node, find the most common class in leaf
            if self.left == None:
                class_occurences = np.zeros((classes,))
                targets = train_target[self.data_indices]
                for c in range(classes):
                    class_occurences[c] = np.count_nonzero(targets == c)
                return np.argmax(class_occurences)
            # else traverse the tree
            else:
                if dato[self.classification_feature] < self.cut_value:
                    return self.left.predict(dato)
                else:
                    return self.right.predict(dato)

        def try_split(self, args):
            self.find_split(args)
            for new_left, new_right, _ in split_by_split_points(self.classification_feature,
                                                                [self.cut_value], self.data_indices):
                left_kid = Node(new_left, self.depth + 1)
                right_kid = Node(new_right, self.depth + 1)
                total_criterion = left_kid.criterion + right_kid.criterion - self.criterion
                self.fake_splits.extend((left_kid, right_kid))

            return total_criterion

    # create the root node
    all_indices = []
    for i in range(len(train_data)):
        all_indices.append(i)

    root_node = Node(all_indices, 0)
    tree = Tree(root_node)
    tree.build(args)

    # - For each node, predict the most frequent class (and the one with
    #   smallest index if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split decreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (e.g., for four instances
    #   with values 1, 7, 3, 3 the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not None, its depth must be less than `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    # TODO: Finally, measure the training and testing accuracy.
    from sklearn.metrics import accuracy_score

    train_accuracy = accuracy_score(tree.predict(train_data), train_target)
    test_accuracy = accuracy_score(tree.predict(test_data), test_target)

    return train_accuracy, test_accuracy


# import cProfile
# cProfile.run("main(parser.parse_args([] if '__file__' not in globals() else None))")
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))