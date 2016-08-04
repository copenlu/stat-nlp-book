import graphviz as gv
import functools
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import statnlpbook.util as util


def load_tweebank(filename):
    """
    Loads files in tweebank format (lines of word[TAB]tag format, empty new lines between
    sentences).
    Args:
        filename: the name of the file to load.
    Returns:
        a list of pairs (x,y) where x is a list of tokens, and y is a list of tags, both of
        same length.
    """
    result = []
    tweet = []
    with open(filename) as f:
        for line in f:
            if line.strip() == "":
                xs = tuple([x.strip() for x, _ in tweet])
                ys = tuple([y.strip() for _, y in tweet])
                result.append((xs, ys))
                tweet = []
            else:
                tweet.append(tuple(line.split("\t")))
    return result


def add_nodes(graph, nodes):
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph


def add_edges(graph, edges):
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph


graph = functools.partial(gv.Graph, format='svg')
digraph = functools.partial(gv.Digraph, format='svg')


def draw_local_fg(length=3):
    var_style = {'shape': 'circle'}
    factor_style = {'shape': 'box', 'fontcolor': 'white',
                    'style': 'filled', 'fillcolor': 'black', 'width': '0.2', 'height': '0.2'}

    ys = graph()
    result = graph()
    result.node("x", shape='circle', style='filled', fillcolor='lightgrey')
    for i in range(0, length):
        node_id = "y" + str(i)
        factor_id = "f" + str(i)
        ys.node(node_id, **var_style)
        result.node(factor_id, **factor_style)
        result.edge(node_id, factor_id)
        result.edge(factor_id, "x")

    result.subgraph(ys)
    return result


def draw_transition_fg(length=3):
    var_style = {'shape': 'circle'}
    factor_style = {'shape': 'box', 'fontcolor': 'white',
                    'style': 'filled', 'fillcolor': 'black', 'width': '0.2', 'height': '0.2'}

    ys = graph()
    result = graph()
    result.node("x", shape='circle', style='filled', fillcolor='lightgrey')
    for i in range(0, length):
        node_id = "y" + str(i)
        factor_id = "l" + str(i)
        ys.node(node_id, **var_style)
        result.node(factor_id, **factor_style)
        result.edge(node_id, factor_id)
        result.edge(factor_id, "x")

    for i in range(1, length):
        factor_id = "t" + str(i)
        ys.node(factor_id, **factor_style)
        ys.edge("y" + str(i), factor_id, constraint="false")
        ys.edge("y" + str(i - 1), factor_id, constraint="false")

    result.subgraph(ys)
    return result


def draw_memm_fg(length=3):
    var_style = {'shape': 'circle'}
    factor_style = {'shape': 'box', 'fontcolor': 'white',
                    'style': 'filled', 'fillcolor': 'black', 'width': '0.2', 'height': '0.2'}

    ys = graph()
    result = graph()
    result.node("x", shape='circle', style='filled', fillcolor='lightgrey')
    result.node("t0", **factor_style)
    for i in range(0, length):
        node_id = "y" + str(i)
        factor_id = "l" + str(i)
        ys.node(node_id, **var_style)
        # result.node(factor_id, **factor_style)
        # result.edge(node_id, factor_id)
        # result.edge(factor_id, "x")
    result.edge("t0", "x")
    result.edge("t0", "y0")

    for i in range(1, length):
        factor_id = "t" + str(i)
        ys.node(factor_id, **factor_style)
        result.edge("x", factor_id, constraint="false")
        ys.edge("y" + str(i), factor_id, constraint="true")
        ys.edge("y" + str(i - 1), factor_id, constraint="true")

    result.subgraph(ys)
    return result


def to_classifier_x(data, f):
    return [f(x, i) for x, _ in data for i in range(0, len(x))]


def to_classifier_y(data):
    return [y[i] for _, y in data for i in range(0, len(y))]


class LocalSequenceLabeler:
    def __init__(self, feat, train_data, **lr_params):
        self.train_data = train_data
        self.feat = feat
        self.vectorizer = DictVectorizer()
        self.label_encoder = LabelEncoder()
        train_classifier_x = self.vectorizer.fit_transform(to_classifier_x(train_data, feat))
        train_classifier_y = self.label_encoder.fit_transform(to_classifier_y(train_data))
        self.lr = LogisticRegression(**lr_params)
        self.lr.fit_transform(train_classifier_x, train_classifier_y)

    def plot_lr_weights(self, label, how_many=20, reverse=True):
        v_index = self.label_encoder.transform(label)
        v_weights = self.vectorizer.inverse_transform(self.lr.coef_)[v_index]
        sorted_weights = sorted(v_weights.items(), key=lambda t: t[1], reverse=reverse)
        return util.plot_bar_graph([w for _, w in sorted_weights[:how_many]],
                                   [f for f, _ in sorted_weights[:how_many]], rotation=45)

    def predict(self, data):
        # dev_classifier_Y = label_encoder.transform(to_classifier_y(dev))
        dev_classifier_output = self.label_encoder.inverse_transform(
            self.lr.predict(self.vectorizer.transform(to_classifier_x(data, self.feat))))
        dev_output = to_xy(data, dev_classifier_output)
        return dev_output

    def show_errors(self, data, filter_gold=lambda y: True, filter_guess=lambda y: True):
        guess = self.predict(data)
        for (x, y), y_guess in zip(data, guess):
            for i in range(0, len(y)):
                if y[i] != y_guess[i] and filter_gold(y[i]) and filter_guess(y_guess[i]):
                    print("---------")
                    print("Gold:  {}".format(y[i]))
                    print("Guess: {}".format(y_guess[i]))
                    print(" ".join(x[max(0, i - 5):i]) + " [" + x[i] + "] " + " ".join(x[i + 1:min(i + 5, len(x))]))
                    print(self.feat(x, i))


def padded_history(seq, index, length):
    history = tuple(seq[max(0, index - length):index])
    padding = (length - len(history)) * ("PAD",)
    return padding + history


class MEMMSequenceLabeler:
    def transform_input(self, data):
        return [self.feat(x, i, padded_history(y, i, self.order)) for x, y in data for i in range(0, len(x))]

    def __init__(self, feat, train_data, order=1, **lr_params):
        self.order = order
        self.train_data = train_data
        self.feat = feat
        self.vectorizer = DictVectorizer()
        self.label_encoder = LabelEncoder()

        train_classifier_x = self.vectorizer.fit_transform(self.transform_input(train_data))
        train_classifier_y = self.label_encoder.fit_transform(to_classifier_y(train_data))
        self.lr = LogisticRegression(**lr_params)
        self.lr.fit_transform(train_classifier_x, train_classifier_y)

    def plot_lr_weights(self, label, how_many=20, reverse=True, feat_filter=lambda s: True):
        v_index = self.label_encoder.transform(label)
        v_weights = self.vectorizer.inverse_transform(self.lr.coef_)[v_index]
        # print(type(v_weights.items()))
        filtered = [(k, v) for k, v in v_weights.items() if feat_filter(k)]
        sorted_weights = sorted(filtered, key=lambda t: t[1],
                                reverse=reverse)
        return util.plot_bar_graph([w for _, w in sorted_weights[:how_many]],
                                   [f for f, _ in sorted_weights[:how_many]], rotation=45)

    def input_repr(self, x, i, y):
        return self.feat(x, i, padded_history(y, i, self.order))

    def predict_next(self, x, i, y):
        scikit_x = self.vectorizer.transform([self.input_repr(x, i, y)])
        return self.label_encoder.inverse_transform(self.lr.predict(scikit_x))[0]

    def predict(self, data):
        result = []
        for x, y in data:
            y_guess = []
            for i in range(0, len(x)):
                prediction = self.predict_next(x, i, y_guess)
                y_guess += prediction
            result.append(y_guess)
        return result
        # dev_classifier_Y = label_encoder.transform(to_classifier_y(dev))
        # dev_classifier_output = self.label_encoder.inverse_transform(
        #     self.lr.predict(self.vectorizer.transform(to_classifier_x(data, self.feat))))
        # dev_output = to_xy(data, dev_classifier_output)
        # return dev_output

        # def show_errors(self, data, filter_gold=lambda y: True, filter_guess=lambda y: True):
        #     guess = self.predict(data)
        #     for (x, y), y_guess in zip(data, guess):
        #         for i in range(0, len(y)):
        #             if y[i] != y_guess[i] and filter_gold(y[i]) and filter_guess(y_guess[i]):
        #                 print("---------")
        #                 print("Gold:  {}".format(y[i]))
        #                 print("Guess: {}".format(y_guess[i]))
        #                 print(" ".join(x[max(0, i - 5):i]) + " [" + x[i] + "] " + " ".join(x[i + 1:min(i + 5, len(x))]))
        #                 print(self.feat(x, i))


def memm_greedy_predict(memm: MEMMSequenceLabeler, data):
    result = []
    for x, y in data:
        y_guess = []
        for i in range(0, len(x)):
            prediction = memm.predict_next(x, i, y_guess)
            y_guess += prediction
        result.append(y_guess)
    return result


def accuracy(gold, guess):
    total = 0
    correct = 0
    for (x, y), y_guess in zip(gold, guess):
        for i in range(0, len(x)):
            correct += 1 if y[i] == y_guess[i] else 0
            total += 1
    return correct / total


def confusion_matrix_dict(gold, guess, normalise=False):
    counts = defaultdict(float)
    for (x, y), y_guess in zip(gold, guess):
        for i in range(0, len(x)):
            counts[y[i], y_guess[i]] += 1.0
    if normalise:
        max_count = max(counts.values())
        old = counts
        counts = defaultdict(float, [(k, v / max_count) for (k, v) in old.items()])
    return counts


def plot_confusion_matrix_dict(matrix_dict):
    labels = set([y for y, _ in matrix_dict.keys()] + [y for _, y in matrix_dict.keys()])
    sorted_labels = sorted(labels)
    matrix = np.zeros((len(sorted_labels), len(sorted_labels)))
    for i1, y1 in enumerate(sorted_labels):
        for i2, y2 in enumerate(sorted_labels):
            matrix[i1, i2] = matrix_dict[y1, y2]
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(sorted_labels))
    plt.xticks(tick_marks, sorted_labels, rotation=45)
    plt.yticks(tick_marks, sorted_labels)
    plt.tight_layout()
    # return matrix


def plot_confusion_matrix(gold, guess, normalise=False):
    plot_confusion_matrix_dict(confusion_matrix_dict(gold, guess, normalise))


# dev_classifier_output[2],to_classifier_y(dev)[2]
def to_xy(data, all_y):
    """
    Convert a flat sequence of labels to a sequence of label sequences based on
    an input sequence of sentences.
    """
    index = 0
    result = []
    for x, y in data:
        new_y = []
        for x_i, y_i in zip(x, y):
            new_y.append(all_y[index])
            index += 1
        result.append(tuple(new_y))
    return result
