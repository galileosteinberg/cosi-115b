# Galileo Steinberg
# CS115B Spring 2024 Homework 2
# Logistic Regression Classifier

import os
from typing import Sequence, DefaultDict, Dict

import numpy as np
from collections import defaultdict
from math import ceil
from random import Random
from scipy.special import expit # logistic (sigmoid) function


class LogisticRegression():

    def __init__(self):
        self.class_dict = {}
        # use of self.feature_dict is optional for this assignment
        self.feature_dict = {}
        self.n_features = None
        self.theta = None # weights (and bias)

    def make_dicts(self, train_set_path: str) -> None:
        '''
        Given a training set, fills in self.class_dict (and optionally,
        self.feature_dict), as in HW1.
        Also sets the number of features self.n_features and initializes the
        parameter vector self.theta.
        '''
        # iterate over training documents
        class_set = set()
        feature_set = set()
        for root, dirs, files in os.walk(train_set_path):
            for name in files:
                if name == '.DS_Store':
                    continue  # Skip .DS_Store files
                with open(os.path.join(root, name)) as f:
                    words = f.read().strip().split()
                    class_name = os.path.basename(root)
                    class_set.add(class_name)
                    for word in words:
                        feature_set.add(word)
        self.class_dict = {class_name: i for i, class_name in enumerate(class_set)}
        self.feature_dict = {feature: i for i, feature in enumerate(feature_set)}
        self.n_features = len(self.feature_dict)
        self.theta = np.zeros(self.n_features + 1)

        # fill in class_dict, (feature_dict,) n_features, and theta
        # the following are used for testing with the toy corpus from the lab 3
        # exercise
        # Comment this out and replace with your code
        # self.class_dict = {'action': 0, 'comedy': 1}
        # self.feature_dict = {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}
        # self.n_features = 4
        # self.theta = np.zeros(self.n_features + 1)   # +1 for bias

    def load_data(self, data_set_path: str):
        '''
        Loads a dataset. Specifically, returns a list of filenames, and dictionaries
        of classes and documents such that:
        classes[filename] = class of the document
        documents[filename] = feature vector for the document (use self.featurize)
        '''
        filenames = []
        classes = dict()
        documents = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set_path):
            for name in files:
                if name == '.DS_Store':
                    continue  # Skip .DS_Store files
                with open(os.path.join(root, name)) as f:
                    words = f.read().strip().split()
                    class_name = os.path.basename(root)
                    filepath = os.path.join(root, name)
                    filenames.append(filepath)
                    classes[filepath] = self.class_dict[class_name]
                    documents[filepath] = self.featurize(words)
        return filenames, classes, documents

    def featurize(self, document: Sequence[str]) -> np.array:
        '''
        Given a document (as a list of words), returns a feature vector.
        Note that the last element of the vector, corresponding to the bias, is a
        "dummy feature" with value 1.
        '''
        vector = np.zeros(self.n_features + 1)   # + 1 for bias
        for word in document:
            if word in self.feature_dict:
                vector[self.feature_dict[word]] = 1
        vector[-1] = 1   # bias
        return vector

    def train(self, train_set_path: str, batch_size=3, n_epochs=1, eta=0.1) -> None:
        '''
        Trains a logistic regression classifier on a training set.
        '''
        filenames, classes, documents = self.load_data(train_set_path)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]
                # create and fill in matrix x and vector y
                x = np.array([documents[filename] for filename in minibatch])
                y = np.array([classes[filename] for filename in minibatch])
                # compute y_hat
                y_hat = expit(np.dot(x, self.theta))
                # applying clipping
                epsilon = 1e-7
                y_hat_clipped = np.clip(y_hat, epsilon, 1 - epsilon)
                # update loss
                loss += np.sum(-y * np.log(y_hat_clipped) - (1 - y) * np.log(1 - y_hat_clipped))
                # compute gradient
                gradient = (np.dot(x.T, (y_hat - y))) / len(minibatch)
                # update weights (and bias)
                self.theta -= eta * gradient
            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
            # randomize order
            Random(epoch).shuffle(filenames)

    def test(self, dev_set_path: str) -> DefaultDict[str, Dict[str, int]]:
        '''
        Tests the classifier on a development or test set.
        Returns a dictionary of filenames mapped to their correct and predicted
        classes such that:
        results[filename]['correct'] = correct class
        results[filename]['predicted'] = predicted class
        '''
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set_path)
        for name in filenames:
            # get most likely class (recall that P(y=1|x) = y_hat)
            y_hat_prob = expit(np.dot(documents[name], self.theta))
            y_pred = 1 if y_hat_prob > 0.5 else 0
            results[name]['correct'] = classes[name]
            results[name]['predicted'] = y_pred
        return results

    def evaluate(self, results: DefaultDict[str, Dict[str, int]]) -> None: ## make sure to change back to None
        '''
        Given results, calculates the following:
        Precision, Recall, F1 for each class
        Accuracy overall
        Also, prints evaluation metrics in readable format.
        '''
        # you can copy and paste your code from HW1 here
        index_to_class = {index: class_name for class_name, index in self.class_dict.items()}
        confusion_matrix = np.zeros((len(self.class_dict), len(self.class_dict)))
        for name in results:
            correct_class_name = index_to_class[results[name]['correct']]
            predicted_class_name = index_to_class[results[name]['predicted']]
            correct_idx = self.class_dict[correct_class_name]
            predicted_idx = self.class_dict[predicted_class_name]
            confusion_matrix[predicted_idx, correct_idx] += 1

        total_correct_predictions = np.trace(confusion_matrix)
        total_predictions = np.sum(confusion_matrix)
        accuracy = (total_correct_predictions / total_predictions) if (total_correct_predictions / total_predictions) > 0 else 0

        for class_name, class_idx in self.class_dict.items():
            precision = confusion_matrix[class_idx, class_idx] / np.sum(confusion_matrix[class_idx, :]) if (
                        np.sum(confusion_matrix[class_idx, :]) > 0) else 0
            recall = confusion_matrix[class_idx, class_idx] / np.sum(confusion_matrix[:, class_idx]) if np.sum(
                confusion_matrix[:, class_idx]) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"Class: {class_name}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1_score}")
            print('\n')

        print(f"Overall accuracy: {accuracy}")
        # return accuracy


if __name__ == '__main__':
    lr = LogisticRegression()
    # batch_sizes = [1, 2, 4, 8, 16]
    # n_epoch_vals = [1, 2, 4, 8, 16]
    # etas = [0.025, 0.05, 0.1, 0.2, 0.4]
    # make sure these point to the right directories
    lr.make_dicts('/Users/GSteinberg/Desktop/CS_115_NLP2/HW2/movie_reviews/train')

    # best_accuracy = 0
    # best_params = {}
    # results_list = []
    #
    # for batch_size in batch_sizes:
    #     for n_epochs in n_epoch_vals:
    #         for eta in etas:
    #             # print(f"Training with batch_size={batch_size}, n_epochs={n_epochs}, eta={eta}")
    #             lr.train('/Users/GSteinberg/Desktop/CS_115_NLP2/HW2/movie_reviews/train', batch_size=batch_size, n_epochs=n_epochs,
    #                      eta=eta)
    #             results = lr.test('/Users/GSteinberg/Desktop/CS_115_NLP2/HW2/movie_reviews/dev')
    #             current_accuracy = lr.evaluate(results)
    #
    #             if current_accuracy > best_accuracy:
    #                 best_accuracy = current_accuracy
    #                 best_params = {'batch_size': batch_size, 'n_epochs': n_epochs, 'eta': eta}
    #             results_list.append({'batch_size': batch_size, 'n_epochs': n_epochs, 'eta': eta, 'accuracy': current_accuracy})
    #
    # sorted_results = sorted(results_list, key=lambda x: x['accuracy'], reverse=True)
    # for result in sorted_results:
    #     print(result)
    # print(f"best params are: Batch Size={best_params['batch_size']}, Epochs={best_params['n_epochs']}, Learning Rate={best_params['eta']}")
    # print(f"Best model accuracy: {best_accuracy}")

    #lr.make_dicts('/Users/GSteinberg/Desktop/CS_115_NLP2/HW2/movie_reviews_small/train')
    #lr.train('/Users/GSteinberg/Desktop/CS_115_NLP2/HW2/movie_reviews_small/train', batch_size=3, n_epochs=1, eta=0.1)
    #results = lr.test('/Users/GSteinberg/Desktop/CS_115_NLP2/HW2/movie_reviews_small/test')

    lr.train('/Users/GSteinberg/Desktop/CS_115_NLP2/HW2/movie_reviews/train', batch_size=1, n_epochs=8, eta=0.025)
    results = lr.test('/Users/GSteinberg/Desktop/CS_115_NLP2/HW2/movie_reviews/dev')
    lr.evaluate(results)
