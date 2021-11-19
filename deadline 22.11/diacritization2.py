#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


class Text:
    def __init__(self, text, target_text, n):
        self.text = text
        self.target_text = target_text
        nodia = "acdeeinorstuuyz"
        self.nodia = nodia + nodia.upper()
        carky = "áéíóúý"
        self.carky = carky + carky.upper()
        self.carky_convert_dict = self.create_convert_dict("aeiouy", "áéíóúý")
        hacky = "čěňřšžďť"
        self.hacky = hacky + hacky.upper()
        self.hacky_convert_dict = self.create_convert_dict("cenrszdt", "čěňřšžďť")
        self.krouzky = "ůŮ"
        self.letters_in_front = n
        self.junk = [".", ",", "\n", "-", ":"]

    # nektera pismena ani nemaji diakritiku, model nezajimaji
    def needed_to_predict_this_letter(self, letter):
        if letter in self.nodia:
            return True
        else:
            return False

    # for creating targets from a letter
    def get_letter_class(self, letter):
        if letter in self.carky:
            letter_class = 1
        elif letter in self.hacky:
            letter_class = 2
        elif letter in self.krouzky:
            letter_class = 3
        else:
            letter_class = 0
        return letter_class

    def convert_piece_of_text_to_np_array(self, text_piece):
        text_piece = text_piece.lower()
        arr = np.empty((self.letters_in_front*2 + 1))

        for i, letter in enumerate(text_piece):
            if letter in self.junk:
                letter = " "
            unicode = ord(letter)
            arr[i] = unicode
        return arr

    """Get train data from the train text"""
    def convert_text_to_train_data(self):
        space = " "
        delka_textu = len(self.text)

        for letter_index in range(delka_textu):
            if not self.needed_to_predict_this_letter(self.text[letter_index]):
                continue

            string = ""
            # zacatek kousku
            if letter_index < self.letters_in_front:
                string += space * self.letters_in_front
            else:
                for i in reversed(range(self.letters_in_front)):
                    string += self.text[(letter_index - i - 1)]

            string += self.text[letter_index]

            #konec kousku
            if delka_textu - letter_index - 1 < self.letters_in_front:
                string += space * self.letters_in_front
            else:

                for i in range(self.letters_in_front):
                    string += self.text[letter_index + 1 + i]

            yield self.convert_piece_of_text_to_np_array(string)

    """Get train targets from the target text with diacritics"""
    def convert_to_train_target(self):
        delka_textu = len(self.target_text)
        all = self.carky + self.hacky + self.krouzky
        for letter_index in range(delka_textu):
            l = self.target_text[letter_index]
            if l not in all and not self.needed_to_predict_this_letter(l):
                continue
            else:
                yield np.array([self.get_letter_class(self.target_text[letter_index])])

    """Get train data and train target from the text"""
    def create_train(self, train_only=False):
        train_data = []
        for dato in self.convert_text_to_train_data():
            train_data.append(dato)
        train_data = np.asarray(train_data)

        if not train_only:
            train_target = []
            for target in self.convert_to_train_target():
                train_target.append(target)
            train_target = np.asarray(train_target)

            return train_data, train_target
        else:
            return train_data

    """Given letter and predicted class, get letter with predicted diacritics"""
    def get_new_letter(self, letter, pred_class):
        if pred_class == 1 and letter in self.carky_convert_dict.keys():
            return self.carky_convert_dict[letter]

        if pred_class == 2 and letter in self.hacky_convert_dict.keys():
            return self.hacky_convert_dict[letter]

        if pred_class == 3:
            if letter == "u":
                return "ů"
            if letter == "U":
                return "Ů"
        return letter

    """For converting letters with predictions, return a dict in a form:
    {"e":"é",...}"""
    @staticmethod
    def create_convert_dict(nodia, dia):
        d = {}
        nodia = nodia + nodia.upper()
        dia = dia + dia.upper()
        for i in range(len(nodia)):
            d[nodia[i]] = dia[i]

        return d

    """Given output from NN to given text, convert it to human readable form"""
    def get_output(self, text, predictions):
        pred_counter = 0
        new_text = ""
        for i, letter in enumerate(text):
            if self.needed_to_predict_this_letter(letter):
                pred = predictions[pred_counter]
                pred_counter += 1
                pred_class = np.argmax(pred)
                if pred_class != 0:
                    # update the letter in text
                    new_letter = self.get_new_letter(letter, pred_class)
                else:
                    new_letter = text[i]
            else:
                new_letter = text[i]
            new_text += new_letter

        return new_text

    def train_data_from_dictionary(self, variants):
        padding = self.letters_in_front * " "
        # iterate over all words in the dictionary
        for w, diacritised in variants.items():
            padded_word = padding + w + padding

            for i, letter in enumerate(padded_word):
                if not self.needed_to_predict_this_letter(letter):
                    continue

                # get the piece of text that will be included in the dataset
                # note: there is no need to care about index errors, as we added
                # the padding
                else:
                    s = ""
                    # the front
                    for x in reversed(range(self.letters_in_front)):
                        s += padded_word[i - x - 1]
                    # the actual letter we care about
                    s += letter
                    # the back
                    for x in range(self.letters_in_front):
                        s += padded_word[i + x + 1]

                    # # we will include the word multiple times in the dataset if it has
                    # # multiple diacritisations
                    # for z in range(len(diacritised)):
                        yield self.convert_piece_of_text_to_np_array(s)

    def label_from_dictionary(self, variants):
        all = self.carky + self.hacky + self.krouzky

        for diacritisations in variants.values():
            # for word in diacritisations:
                for letter in diacritisations[0]:
                    if letter not in all and not self.needed_to_predict_this_letter(letter):
                        continue
                    else:
                        yield np.array([self.get_letter_class(letter)])

    def create_train_from_dict(self, dicti, train_only=False):
        train_data = []
        for dato in self.train_data_from_dictionary(dicti):
            train_data.append(dato)
        train_data = np.asarray(train_data)

        if not train_only:
            train_target = []
            for target in self.label_from_dictionary(dicti):
                train_target.append(target)
            train_target = np.asarray(train_target)

            return train_data, train_target
        else:
            return train_data

def main(args: argparse.Namespace):
    # number of letters in front and after to consider
    n = 3

    # this could be done better, but for now I am defining the categories "by hand"
    # otherwise, more complex logic would be needed, because there could be letters missing in
    # each column in the training data, so when applying the model to unseen data, the required
    # input shapes wouldn't match
    cats = np.array([32.0, 33.0, 34.0, 39.0, 40.0, 41.0, 48.0, 49.0,
                     50.0, 52.0, 54.0, 56.0, 57.0, 59.0, 63.0, 97.0,
                     98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
                     105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                     112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0,
                     119.0, 120.0, 121.0, 122.0])

    categor = []
    for i in range(n * 2 + 1):
        categor.append(cats)

    OHE = OneHotEncoder(sparse=False, categories=categor, handle_unknown='ignore')

    # OHE = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
    OHE_label = OneHotEncoder(sparse=False)

    if args.predict is None:
        # We are training a model.

        np.random.seed(args.seed)
        train = Dataset()
        t = Text(train.data, train.target, n)

        train_data, train_target = t.create_train()

        from diacritization_dictionary import Dictionary

        d = Dictionary()

        train_dict, target_dict = t.create_train_from_dict(d.variants, train_only=False)

        train_data = np.concatenate((train_dict, train_data))

        train_data = OHE.fit_transform(train_data)
        # all_c = []
        # for feat in OHE.categories_:
        #     for c in feat:
        #         if c not in all_c:
        #             all_c.append(c)
        #
        # print(sorted(all_c))
        train_target = OHE_label.fit_transform(np.concatenate((target_dict, train_target)))

        print(train_data.shape, train_target.shape)

        MLP = MLPClassifier(hidden_layer_sizes=300, verbose=True, max_iter=200)
        # TODO: Train a model on the given dataset and store it in `model`.
        model = MLP.fit(train_data, train_target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        t = Text(test.data, None, n)
        test_data = t.create_train(train_only=True)
        test_data = OHE.fit_transform(test_data)
        model_predictions = model.predict(test_data)
        predictions = t.get_output(test.data, model_predictions)
        print(predictions)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)