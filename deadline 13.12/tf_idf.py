#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request
import re

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
from collections import Counter
from copy import deepcopy
from math import log

class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=True, action="store_true", help="Use IDF weights")
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=37, type=int, help="Random seed")
parser.add_argument("--tf", default=True, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.

def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a feature for every word that is present at least twice
    # in the training data. A word is every maximal sequence of at least 2 word characters,
    # where a word character corresponds to a regular expression `\w`.
    def get_words(document):
        return re.sub(r"[^\w]", ' ', document).split()

    """Creates a list in the form:
    [(number_of_words, {'word':number_of_occurences_in_doc, 'word2':....,}), ({....]  where each dict in the list corresponds to 
    one document AND a dict with words and corresponding indices and in how many documents the word occurs 
    {'word':(0, in_how_many_documents_the_word_occurs...} """

    def process_dataset(args, data):
        processed = []
        all_words = {}
        words_found_once_so_far = []
        words_counter = 0

        # for each document create a dict with
        for d_index, document in enumerate(data):
            just_a_dict = {}
            words = get_words(document)
            number_of_words_in_doc = len(words)

            for w in words:
                # if word isnt in the dictionary yet
                if len(w) > 1:
                    # the occurrences in single document
                    first_occurrence_in_doc = False
                    if w not in just_a_dict.keys():
                        just_a_dict[w] = 1
                        first_occurrence_in_doc = True
                    just_a_dict[w] += 1

                    # the dict with all words, its index and number of occurrences for IDF
                    if w not in all_words.keys():
                        all_words[w] = [words_counter, 1]
                        words_counter += 1
                    else:
                        if first_occurrence_in_doc:
                            all_words[w][1] += 1 # the word occurs in another document

            processed.append((number_of_words_in_doc, just_a_dict))

        return processed, all_words, words_counter

    processed_documents, all_words, number_of_words = process_dataset(args, train_data)

    def idf_formula(number_of_documents, number_of_documents_containing_t):
        return log(number_of_documents / (number_of_documents_containing_t + 1))

    # computes idfs of all words in the dataset
    def get_idf_of_all_words():
        all_idfs = []  # contains idfs of all words
        number_of_documents = len(train_data)
        for word, word_info in all_words.items():
            all_idfs.append(idf_formula(number_of_documents, word_info[1]))

        return all_idfs

    def get_empty_array():
        return np.zeros(number_of_words,)

    # computes TFs of all words in a document, returns array representation
    def get_tf(document_info, test=False):
        array_representation = get_empty_array()
        number_of_words_in_document = document_info[0]
        document_dict = document_info[1]
        for word, occurrences in document_dict.items():
            tf_value = occurrences/number_of_words_in_document
            if not test:
                array_representation[all_words[word][0]] = tf_value
            else:
                if word in all_words.keys():
                    array_representation[all_words[word][0]] = tf_value

        return array_representation

    # computes IDFs of all words in a document, returns array representation
    def get_idf(document_info, all_idfs, test=False):
        array_representation = get_empty_array()
        document_dict = document_info[1]
        for word in document_dict.keys():
            # if we are using this function for test data, the word may not exist in all_words
            if not test:
                word_index = all_words[word][0]
                array_representation[word_index] = all_idfs[word_index]
            else:
                if word in all_words.keys():
                    word_index = all_words[word][0]
                    array_representation[word_index] = all_idfs[word_index]

        return array_representation

    # get binary representation of a doc
    def get_binary_representation(document_info, test=False):
        array_representation = get_empty_array()
        document_dict = document_info[1]
        for word in document_dict.keys():
            if not test:
                word_index = all_words[word][0]
                array_representation[word_index] = 1
            else:
                if word in all_words.keys():
                    word_index = all_words[word][0]
                    array_representation[word_index] = 1

        return array_representation

    # similar to process_dataset, but done here for better readability
    def process_test_data(data):
        processed = []

        # for each document create a dict with
        for d_index, document in enumerate(data):
            just_a_dict = {}
            words = get_words(document)
            number_of_words_in_doc = len(words)

            for w in words:
                # if word isnt in the dictionary yet
                if len(w) > 1:
                    # the occurrences in single document
                    if w not in just_a_dict.keys():
                        just_a_dict[w] = 1
                    just_a_dict[w] += 1

            processed.append((number_of_words_in_doc, just_a_dict))

        return processed

    processed_test_data = process_test_data(test_data)


    # TODO: Weight the selected features using
    # - term frequency (TF), if `args.tf` is set;
    # - inverse document frequency (IDF), if `args.idf` is set; use
    #   the variant which contains `+1` in the denominator;
    # - TF * IDF, if both `args.tf` and `args.idf` are set;
    # - binary indicators, if neither `args.tf` nor `args.idf` are set.
    # Note that IDFs are computed on the train set and then reused without
    # modification on the test set, while TF is computed for every document separately.

    def get_data_representation(args, processed_docs, all_idfs=None, test=False):
        representation = []
        if args.tf and args.idf:
            for doc in processed_docs:
                doc_tf = get_tf(doc, test=test)
                doc_idf = get_idf(doc, all_idfs, test=test)
                doc_tf_idf = doc_tf * doc_idf
                representation.append(doc_tf_idf)

        elif args.idf:
            for doc in processed_docs:
                doc_idf = get_idf(doc, all_idfs, test=test)
                representation.append(doc_idf)

        elif args.tf:
            for doc in processed_docs:
                doc_tf = get_tf(doc,test=test)
                representation.append(doc_tf)

        else:
            for doc in processed_docs:
                doc_binary = get_binary_representation(doc, test=test)
                representation.append(doc_binary)

        return representation

    if args.idf:
        all_idf = get_idf_of_all_words()
    else:
        all_idf = None
    X_train = get_data_representation(args, processed_documents, all_idfs=all_idf)
    X_test = get_data_representation(args, processed_test_data, all_idfs=all_idf, test=True)

    # TODODODODOODD:  # Finally, for each document L2-normalize its features.

    # TODO: Perform classification of the test set using the k-NN algorithm
    # from sklearn (pass the `algorithm="brute"` option), with `args.k` nearest
    # neighbors determined using the cosine similarity, where
    #   cosine_similarity(x, y) = x^T y / (||x|| * ||y||).
    # Note that for L2-normalized data (which we have), the nearest neighbors
    # are equivalent to using the usual Euclidean distance (L2 distance).

    # def cosine_sim(x, y):
    #     return x * y.t / (|| x || * || y ||)

    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=args.k, weights='distance', algorithm='brute')
    model.fit(X_train, train_target)
    prediction = model.predict(X_test)
    from sklearn.metrics import f1_score as f1
    # TODO: Evaluate the performance using macro-averaged F1 score.
    f1_score = f1(prediction, test_target, average='macro')

    return f1_score
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}, k={}: {:.1f}%".format(args.tf, args.idf, args.k, 100 * f1_score))