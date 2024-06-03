# Galileo Steinberg
# CS114B Spring 2023 Homework 1
# Naive Bayes in Numpy

import os
import numpy as np
from collections import defaultdict, Counter

class NaiveBayes():

    def __init__(self):
        self.class_dict = {}
        self.feature_dict = {}
        self.prior = None
        self.likelihood = None

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[feature, class] = log(P(feature|class))
    '''
    def train(self, train_set):
        class_counts = Counter()
        feature_counts = defaultdict(Counter)
        vocab = set()

        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                if name == '.DS_Store':
                    continue  # Skip .DS_Store files
                with open(os.path.join(root, name)) as f:
                    # collect class counts and feature counts
                    words = f.read().strip().split()
                    class_name = os.path.basename(root)
                    class_counts[class_name] += 1
                    for word in words:
                        vocab.add(word)
                        feature_counts[class_name][word] += 1
        # fill in class_dict and feature_dict
        self.class_dict = {class_name: idx for idx, class_name in enumerate(class_counts.keys())}
        self.feature_dict = {feature: idx for idx, feature in enumerate(vocab)}

        # normalize counts to probabilities, and take logs
        num_classes = len(self.class_dict)
        num_features = len(self.feature_dict)

        self.prior = np.log([class_counts[class_name] / sum(class_counts.values()) for class_name in class_counts.keys()])
        self.likelihood = np.zeros((num_features, num_classes))

        for class_name, class_idx in self.class_dict.items():
            total_words = sum(feature_counts[class_name].values())
            for word, word_idx in self.feature_dict.items():
                word_count = feature_counts[class_name][word]
                self.likelihood[word_idx, class_idx] = np.log((word_count + 1) / (total_words + num_features))

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                if name == '.DS_Store':
                    continue  # Skip .DS_Store files

                feature_vector = np.zeros(len(self.feature_dict))
                with open(os.path.join(root, name)) as f:
                    # create feature vectors for each document
                    words = f.read().strip().split()
                    for word in words:
                        if word in self.feature_dict:
                            feature_vector[self.feature_dict[word]] += 1
                log_probs = np.dot(feature_vector, self.likelihood) + self.prior
                # get most likely class
                predicted_class_idx = np.argmax(log_probs)
                predicted_class = list(self.class_dict.keys())[predicted_class_idx]
                correct_class = os.path.basename(root)

                results[name]['correct'] = correct_class
                results[name]['predicted'] = predicted_class
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you may find this helpful
        confusion_matrix = np.zeros((len(self.class_dict), len(self.class_dict)))
        for name in results:
            correct_idx = self.class_dict[results[name]['correct']]
            predicted_idx = self.class_dict[results[name]['predicted']]
            confusion_matrix[predicted_idx, correct_idx] += 1

        total_correct_predictions = np.trace(confusion_matrix)
        total_predictions = np.sum(confusion_matrix)
        accuracy = (total_correct_predictions / total_predictions) if (total_correct_predictions / total_predictions) > 0 else 0

        for class_name, class_idx in self.class_dict.items():
            precision = confusion_matrix[class_idx, class_idx] / np.sum(confusion_matrix[class_idx, :]) if (np.sum(confusion_matrix[class_idx, :]) > 0) else 0
            recall = confusion_matrix[class_idx, class_idx] / np.sum(confusion_matrix[:, class_idx]) if np.sum(confusion_matrix[:, class_idx]) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"Class: {class_name}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1_score}")
            print('\n')

        print(f"Overall accuracy: {accuracy}")


if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('/Users/GSteinberg/Desktop/CS_115_NLP2/HW1/movie_reviews/train')
    #nb.train('/Users/GSteinberg/Desktop/CS_115_NLP2/HW1/movie_reviews_small/train')
    results = nb.test('/Users/GSteinberg/Desktop/CS_115_NLP2/HW1/movie_reviews/dev')
    #results = nb.test('/Users/GSteinberg/Desktop/CS_115_NLP2/HW1/movie_reviews_small/test')
    nb.evaluate(results)
