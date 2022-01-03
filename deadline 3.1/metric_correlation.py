#!/usr/bin/env python3
import argparse
import dataclasses

import numpy as np

class ArtificialData:
    @dataclasses.dataclass
    class Sentence:
        """ Information about a single dataset sentence."""
        gold_edits: int # Number of required edits to be performed.
        predicted_edits: int # Number of edits predicted by a model.
        predicted_correct: int # Number of correct edits predicted by a model.
        human_rating: int # Human rating of the model prediction.

    def __init__(self, args: argparse.Namespace):
        generator = np.random.RandomState(args.seed)

        self.sentences = []
        for _ in range(args.data_size):
            gold = generator.poisson(2)
            correct = generator.randint(gold + 1)
            predicted = correct + generator.poisson(0.5)
            human_rating = max(0, int(100 - generator.uniform(5, 8) * (gold - correct) - generator.uniform(8, 13) * (predicted - correct)))
            self.sentences.append(self.Sentence(gold, predicted, correct, human_rating))


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrap_samples", default=100, type=int, help="Bootstrap samples")
parser.add_argument("--data_size", default=1000, type=int, help="Data set size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.

def main(args: argparse.Namespace):
    # Create the artificial data
    data = ArtificialData(args)

    def mean(l):
        return sum(l) / len(l)

    def pearson(l1, l2):
        mean_l1 = mean(l1)
        mean_l2 = mean(l2)
        to_nahore = 0
        for i in range(len(l1)):
            to_nahore += (l1[i] - mean_l1) * (l2[i] - mean_l2)
        x_sum_dole = 0
        y_sum_dole = 0
        for i in range(len(l1)):
            x_sum_dole += (l1[i] - mean_l1) ** 2
            y_sum_dole += (l2[i] - mean_l2) ** 2
        to_dole = x_sum_dole ** 0.5 * y_sum_dole ** 0.5

        return to_nahore / to_dole

    # Create `args.bootstrap_samples` bootstrapped samples of the dataset by
    # sampling sentences of the original dataset, and for each compute
    # - average of human ratings
    # - TP, FP, FN counts of the predicted edits

    human_ratings, predictions = [], []
    generator = np.random.RandomState(args.seed)
    for _ in range(args.bootstrap_samples):
        # Bootstrap sample of the dataset
        sentences = generator.choice(data.sentences, size=len(data.sentences), replace=True)

        # TODO: Append the average of human ratings of `sentences` to `humans`.
        all_human_rating = 0
        for s in sentences:
            all_human_rating += s.human_rating
        average = all_human_rating / len(data.sentences)
        human_ratings.append(average)

        # TODO: Compute TP, FP, FN counts of predicted edits in `sentences`
        TP, FP, FN = 0, 0, 0
        for s in sentences:
            TP += s.predicted_correct
            FP += s.predicted_edits - s.predicted_correct
            FN += s.gold_edits - s.predicted_correct
        predictions.append([TP, FP, FN])

    # Compute Pearson correlation between F_beta score and human ratings
    # for betas between 0 and 2.
    betas, correlations = [], []
    for beta in np.linspace(0, 2, 201):
        betas.append(beta)

        # TODO: For each bootstap dataset, compute the F_beta score using
        # the counts in `predictions` and then manually compute the Pearson
        # correlation between the computed scores and `human_ratings`. Append
        # the result to `correlations`.
        f_betas = []

        for p in predictions:
            F_beta = ((1 + beta**2) * p[0]) / ((1 + beta**2) * p[0] + beta**2 * p[2] + p[1])
            f_betas.append(F_beta)

        correlations.append(pearson(f_betas, human_ratings))

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(betas, correlations)
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"Pearson correlation of $F_\beta$-score and human ratings")
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    # TODO: Assign the highest correlation to `best_correlation` and
    # store corresponding beta to `best_beta`.
    best_correlation = max(correlations)
    index = correlations.index(best_correlation)
    best_beta = betas[index]

    return best_beta, best_correlation

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_beta, best_correlation = main(args)

    print("Best correlation of {:.3f} was found for beta {:.2f}".format(
        best_correlation, best_beta))