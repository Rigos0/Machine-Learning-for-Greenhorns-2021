#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np
import re
from math import log
from sklearn.ensemble import GradientBoostingRegressor

class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")

def main(args: argparse.Namespace):
    def get_words(document):
            return document.lower().split()
            # return re.findall(r"('\w+)|(\w+'\w+)|(\w+')|(\w+)", document.lower())

    def process_dataset(args, data, all_words_in_dataset):
        processed = []

        for document in data:
            single_doc_dict = {}
            words = get_words(document)
            number_of_words_in_doc = 0

            # Iterate over all words in the document
            for w in words:

                number_of_words_in_doc += 1
                # the occurrences in the single document
                if w not in single_doc_dict.keys():
                    single_doc_dict[w] = 1
                    first_occurrence_in_doc = True
                else:
                    single_doc_dict[w] += 1
                    first_occurrence_in_doc = False

                if first_occurrence_in_doc:
                    if w in all_words_in_dataset.keys():
                        all_words_in_dataset[w][1] += 1  # the word occurs in another document

            processed.append((number_of_words_in_doc, single_doc_dict))
        # processed train data, dict with all words in the dataset, number of words in the dataset
        return processed, all_words_in_dataset

    def idf_formula(number_of_documents, number_of_documents_containing_t):
        return log(number_of_documents / (number_of_documents_containing_t + 1))

        # computes idfs of all words in the dataset

    def get_idf_of_all_words():
        all_idfs = []  # contains idfs of all words
        number_of_documents = len(X_train)
        for word, word_info in all_words.items():
            all_idfs.append(idf_formula(number_of_documents, word_info[1]))

        return all_idfs

    def get_empty_array():
        return np.zeros(number_of_words, )

        # computes TFs of all words in a document, returns array representation

    def get_tf(document_info):
        array_representation = get_empty_array()
        number_of_words_in_document = document_info[0]
        document_dict = document_info[1]
        for word, occurrences in document_dict.items():
            tf_value = occurrences / number_of_words_in_document
            if word in all_words.keys():
                array_representation[all_words[word][0]] = tf_value

        return array_representation

        # computes IDFs of all words in a document, returns array representation

    def get_idf(document_info, all_idfs):
        array_representation = get_empty_array()
        document_dict = document_info[1]
        for word in document_dict.keys():
            # if we are using this function for test data, the word may not exist in all_words
            if word in all_words.keys():
                array_representation[all_words[word][0]] = all_idfs[all_words[word][0]]

        return array_representation

    def get_data_representation(args, processed_docs, all_idfs):
        representation = []
        for doc in processed_docs:
            doc_tf = get_tf(doc)
            doc_idf = get_idf(doc, all_idfs)
            doc_tf_idf = doc_tf * doc_idf
            representation.append(doc_tf_idf)

        return representation

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        # Create train-test split.
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_target, test_target = train_test_split(
            train.data, train.target, random_state=args.seed)

        X_train = train_data
        y_train = train_target
        # X_train = train.data
        # y_train = train.target

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        def find_all_words_in_the_dataset(data):
            words_occurrences = {}
            # count the words in the dataset
            for document in data:
                words = get_words(document)

                for w in words:
                    if w not in words_occurrences:
                        words_occurrences[w] = 1
                    else:
                        words_occurrences[w] += 1

            # keep the words that are at least once in the dataset
            all_words_in_data = {}
            word_counter = 0  # this will later become the index of the feature of the word
            for key in words_occurrences.keys():
                if words_occurrences[key] != 1:
                    # in the place of zero, we will later count the number of docs the word is in
                    all_words_in_data[key] = [word_counter, 0]
                    word_counter += 1
            return all_words_in_data, word_counter

        all_words, number_of_words = find_all_words_in_the_dataset(X_train)
        processed_documents, all_words = process_dataset(args, X_train, all_words)
        for x in all_words:
            print(x)

        all_idf = get_idf_of_all_words()

        np.save('train_info.npy', np.asarray([all_words, number_of_words, all_idf]))

        X_train = np.asarray(get_data_representation(args, processed_documents, all_idfs=all_idf))
        print(X_train.shape)

        processed_documents.clear()

        from sklearn.neural_network import MLPClassifier
        # TODO: Train a model on the given dataset and store it in `model`.
        mlp = MLPClassifier(hidden_layer_sizes=(50), verbose=True, max_iter=20, learning_rate='adaptive')
        # TODO: Train a model on the given dataset and store it in `model`.
        model = mlp.fit(X_train, y_train)

        # TRY:
        from sklearn.metrics import f1_score as f1
        processed_documents, all_words = process_dataset(args, test_data, all_words)
        X_test = np.asarray(get_data_representation(args, processed_documents, all_idfs=all_idf))

        preds = model.predict(X_test)
        print(f1(preds, test_target))


        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)
        X_test = test.data

        train_info = np.load('train_info.npy', allow_pickle=True)
        all_idf = train_info[2]

        all_words, number_of_words = train_info[0], train_info[1]
        processed_documents, all_words = process_dataset(args, X_test, all_words)
        X_test = np.asarray(get_data_representation(args, processed_documents, all_idfs=all_idf))

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(X_test)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)