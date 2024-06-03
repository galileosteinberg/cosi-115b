# Galileo Steinberg
# CS115B Spring 2024 Homework 3
# Part-of-speech Tagging with Structured Perceptrons

import os
import pickle

import numpy as np
from collections import defaultdict
from random import Random

class POSTagger():

    def __init__(self):
        # For testing with the toy corpus from the lab 7 exercise
        self.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
        self.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
                          'dwarf': 4, 'cheered': 5}
        self.initial = np.array([-0.3, -0.7, 0.3])
        self.transition = np.array([[-0.7, 0.3, -0.3],
                                    [-0.3, -0.7, 0.3],
                                    [0.3, -0.3, -0.7]])
        self.emission = np.array([[-0.3, -0.7, 0.3],
                                  [0.3, -0.3, -0.7],
                                  [-0.3, 0.3, -0.7],
                                  [-0.7, -0.3, 0.3],
                                  [0.3, -0.7, -0.3],
                                  [-0.7, 0.3, -0.3]])
        # Should raise an IndexError; if you come across an unknown word, you
        # Should treat the emission scores for that word as 0
        self.unk_index = np.inf

    def make_dicts(self, train_set):
        '''
        Fills in self.tag_dict and self.word_dict, based on the training data.
        '''
        # Iterate over training documents
        tag_set = set()
        word_set = set()
        for root, dirs, files in os.walk(train_set):
            for name in files:
                if name == '.DS_Store':
                    continue  # Skip .DS_Store files
                with open(os.path.join(root, name)) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        words = line.strip().split()
                        for word in words:
                            token, tag = word.split('/', 1)
                            word_set.add(token)
                            tag_set.add(tag)

        self.word_dict = {word: i for i, word in enumerate(word_set)}
        self.tag_dict = {tag: i for i, tag in enumerate(tag_set)}

    def load_data(self, data_set):
        '''
        Loads a dataset. Specifically, returns a list of sentence_ids, and
        dictionaries of tag_lists and word_lists such that:
        tag_lists[sentence_id] = list of part-of-speech tags in the sentence
        word_lists[sentence_id] = list of words in the sentence
        '''
        sentence_ids = []
        tag_lists = dict()
        word_lists = dict()
        sentence_id = 0
        # Iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                if name == '.DS_Store':
                    continue  # Skip .DS_Store files
                with open(os.path.join(root, name)) as f:
                    for line in f:
                        if not line.strip():
                            continue  # Skip empty lines

                        tags = []
                        words = []
                        tokens = line.strip().split()
                        for token in tokens:
                            word, tag = token.rsplit('/', 1)
                            word_index = self.word_dict.get(word, self.unk_index)
                            tag_index = self.tag_dict.get(tag, self.unk_index)

                            words.append(word_index)
                            tags.append(tag_index)


                        sentence_ids.append(sentence_id)
                        tag_lists[sentence_id] = tags
                        word_lists[sentence_id] = words
                        sentence_id += 1

        return sentence_ids, tag_lists, word_lists

    def viterbi(self, sentence):
        '''
        Implements the Viterbi algorithm.
        Use v and backpointer to find the best_path.
        '''
        T = len(sentence)
        N = len(self.tag_dict)
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        # Your code here
        # Initialization step
        start_word_idx = sentence[0]
        emission_scores_start = self.emission[start_word_idx, :] if start_word_idx < len(self.word_dict) else np.zeros(N)
        v[:, 0] = self.initial + emission_scores_start
        # Recursion step
        for t in range(1, T):
            word_index = sentence[t]
            emission_scores = self.emission[word_index, :] if word_index < len(self.word_dict) else np.zeros(N)

            scores = v[:, t - 1, None] + self.transition
            scores_max = np.max(scores, axis=0)

            v[:, t] = scores_max + emission_scores
            backpointer[:, t] = np.argmax(scores, axis=0)

        # Termination step
        best_last_tag = np.argmax(v[:, T-1])
        best_path = [best_last_tag]
        for t in range(T - 1, 0, -1):
            best_path.append(backpointer[best_path[-1], t])

        best_path.reverse()

        return best_path

    def train(self, train_set):
        '''
        Trains a structured perceptron part-of-speech tagger on a training set.
        '''
        self.make_dicts(train_set)
        sentence_ids, tag_lists, word_lists = self.load_data(train_set)
        Random(0).shuffle(sentence_ids)
        self.initial = np.zeros(len(self.tag_dict))
        self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict)))
        self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))
        for i, sentence_id in enumerate(sentence_ids):
            # Your code here
            sentence = word_lists[sentence_id]
            correct_tags = tag_lists[sentence_id]
            predicted_tags = self.viterbi(sentence)

            for t, word_index in enumerate(sentence):
                correct_tag = correct_tags[t]
                predicted_tag = predicted_tags[t]
                if word_index != np.inf and word_index < len(self.word_dict):
                    self.emission[word_index, correct_tag] += 1
                    self.emission[word_index, predicted_tag] -= 1

                if t == 0:
                    self.initial[correct_tag] += 1
                    self.initial[predicted_tag] -= 1
                    # Update transition weights
                if t > 0:  # Transition is relevant from the second word onwards
                    self.transition[correct_tags[t - 1], correct_tag] += 1
                    self.transition[predicted_tags[t - 1], predicted_tag] -= 1

            # Prints progress of training
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')

    def test(self, dev_set):
        '''
        Tests the tagger on a development or test set.
        Returns a dictionary of sentence_ids mapped to their correct and predicted
        sequences of part-of-speech tags such that:
        results[sentence_id]['correct'] = correct sequence of tags
        results[sentence_id]['predicted'] = predicted sequence of tags
        '''
        results = defaultdict(dict)
        sentence_ids, tag_lists, word_lists = self.load_data(dev_set)
        for i, sentence_id in enumerate(sentence_ids):
            # your code here
            sentence = word_lists[sentence_id]  # Get the list of words for the current sentence
            correct_tags = tag_lists[sentence_id]

            predicted_tags = self.viterbi(sentence)

            results[sentence_id]['correct'] = correct_tags
            results[sentence_id]['predicted'] = predicted_tags
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
        return results


    def evaluate(self, results):
        '''
        Given results, calculates overall accuracy.
        '''
        total_correct = 0
        total_words = 0

        for sentence_id, tags in results.items():
            correct_tags = tags['correct']
            predicted_tags = tags['predicted']

            # Increment the total number of words
            total_words += len(correct_tags)

            # Increment the total number of correct tags
            for correct_tag, predicted_tag in zip(correct_tags, predicted_tags):
                if correct_tag == predicted_tag:
                    total_correct += 1

        # Calculate accuracy
        accuracy = total_correct / total_words if total_words > 0 else 0.0
        return accuracy


if __name__ == '__main__':
    pos = POSTagger()
    # Make sure train and test point to the right directories

    # Small datasets
    # pos.train('/Users/GSteinberg/Desktop/CS_115_NLP2/HW3/data_small/train')
    # results = pos.test('/Users/GSteinberg/Desktop/CS_115_NLP2/HW3/data_small/test')
    # results = pos.test('data_small/test')

    # Full dataset
    # pos.train('/Users/GSteinberg/Desktop/CS_115_NLP2/HW3/brown/train')
    pos.train('brown/train')
    # Writes the POS tagger to a file
    with open('pos_tagger.pkl', 'wb') as f:
        pickle.dump(pos, f)
    # Reads the POS tagger from a file
    with open('pos_tagger.pkl', 'rb') as f:
        pos = pickle.load(f)

    results = pos.test('brown/dev')
    #results = pos.test('/Users/GSteinberg/Desktop/CS_115_NLP2/HW3/brown/dev')
    print('Accuracy:', pos.evaluate(results))
