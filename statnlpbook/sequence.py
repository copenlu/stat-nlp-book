import functools
from collections import defaultdict

import graphviz as gv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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
        node_id = "y" + str(i+1)
        factor_id = "p(y{index} | x, {index})".format(index=i+1)
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

    obs_style = {'shape': 'circle', 'style': 'filled', 'fillcolor': 'lightgrey'}
    ys = graph()
    result = graph()
    result.node("x", **obs_style)
    ys.node("y0", label="P", **obs_style)
    for i in range(1, length):
        node_id = "y" + str(i)
        factor_id = "l" + str(i)
        ys.node(node_id, **var_style)
        # result.node(factor_id, **factor_style)
        # result.edge(node_id, factor_id)
        # result.edge(factor_id, "x")

    for i in range(1, length):
        factor_id = "t" + str(i)
        ys.node(factor_id, **factor_style)
        ys.edge("y" + str(i), factor_id, constraint="false")
        ys.edge("y" + str(i - 1), factor_id, constraint="false")
        ys.edge(factor_id, "x")

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
        if "C" not in lr_params:
            self.lr = LogisticRegression(C=10, **lr_params)
        else:
            self.lr = LogisticRegression(**lr_params)
        self.lr.fit_transform(train_classifier_x, train_classifier_y)

        self.v_weights = self.vectorizer.inverse_transform(self.lr.coef_)

    def weights(self, label):
        v_index = self.label_encoder.transform(label)
        v_weights = self.v_weights[v_index]
        return v_weights

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

    def errors(self, data, filter_gold=lambda y: True, filter_guess=lambda y: True):
        guess = self.predict(data)
        model = self
        errors = []
        for (x, y), y_guess in zip(data, guess):
            for i in range(0, len(y)):
                if filter_gold(y[i]) and filter_guess(y_guess[i]):
                    errors.append(SingleError(i, x, y, y_guess, model))
        return errors


class SingleError:
    def __init__(self, i, x, y, y_guess, model):
        self.model = model
        self.y_guess = y_guess
        self.y = y
        self.x = x
        self.i = i

    def _repr_html_(self):
        from_index = max(0, self.i - 5)
        to_index = min(self.i + 5, len(self.x))
        words = to_td_seq(self.x, from_index, to_index, self.i)
        gold = to_td_seq(self.y, from_index, to_index, self.i)
        guess = to_td_seq(self.y_guess, from_index, to_index, self.i)
        cell = """<table style=""><tr>{words}</tr><tr>{gold}</tr><tr>{guess}</tr></table>""".format(
            words=words,
            gold=gold,
            guess=guess)
        feats = self.model.input_repr(self.x, self.i, self.y_guess) if isinstance(self.model,
                                                                                  MEMMSequenceLabeler) else self.model.feat(
            self.x, self.i)
        sorted_feat_keys = sorted(feats.keys())
        gold_weights = defaultdict(float, self.model.weights(self.y[self.i]))
        guess_weights = defaultdict(float, self.model.weights(self.y_guess[self.i]))
        feat_names = "<td>" + "</td><td>".join(sorted_feat_keys) + "</td>"
        feat_values = "<td>" + "</td><td>".join([str(feats[k]) for k in sorted_feat_keys]) + "</td>"

        def to_feat_key(key, value):
            if isinstance(value, bool):
                return key
            elif isinstance(value, float):
                return key
            else:
                return "{}:{}".format(key, value)

        gold_weights_row = "<td>" + "</td><td>".join(
            ["{:.2f}".format(gold_weights[to_feat_key(key, feats[key])]) for key in sorted_feat_keys]) + "</td>"
        guess_weights_row = "<td>" + "</td><td>".join(
            ["{:.2f}".format(guess_weights[to_feat_key(key, feats[key])]) for key in sorted_feat_keys]) + "</td>"

        feat_table = """
        <table>
          <tr>{names}</tr>
          <tr>{values}</tr>
          <tr>{gold_weights}</tr>
          <tr>{guess_weights}</tr>
        </table>""".format(names=feat_names, values=feat_values, gold_weights=gold_weights_row,
                           guess_weights=guess_weights_row)

        return cell + feat_table


def to_td_seq(sequence, from_index, to_index, focus=None):
    return "<td>" + "</td><td>".join(["{}".format(sequence[j]) if j != focus else "<b>{}</b>".format(sequence[j])
                                      for j in range(from_index, to_index)]) + "</td>"


def padded_history(seq, index, length):
    history = tuple(seq[max(0, index - length):index])
    padding = (length - len(history)) * ("PAD",)
    return padding + history


def errors(data, y_guesses, gold_label=None, guess_label=None, model=None):
    def test_gold(l):
        return gold_label is None or l == gold_label

    def test_guess(l):
        return guess_label is None or l == guess_label

    errors = []
    for (x, y), y_guess in zip(data, y_guesses):
        for i in range(0, len(x)):
            if test_gold(y[i]) and test_guess(y_guess[i]) and y[i] != y_guess[i]:
                errors.append(SingleError(i, x, y, y_guess, model))
    return errors


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
        self.v_weights = self.vectorizer.inverse_transform(self.lr.coef_)

    def weights(self, label):
        v_index = self.label_encoder.transform(label)
        v_weights = self.v_weights[v_index]
        return v_weights

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

    def sklearn_repr(self, x, i, y):
        return self.vectorizer.transform([self.input_repr(x, i, y)])

    def predict_next(self, x, i, y):
        scikit_x = self.vectorizer.transform([self.input_repr(x, i, y)])
        return self.label_encoder.inverse_transform(self.lr.predict(scikit_x))[0]

    def predict_next_hist(self, x, i, hist):
        scikit_x = self.vectorizer.transform([self.feat(x, i, padded_history(hist, 1, self.order))])
        return self.label_encoder.inverse_transform(self.lr.predict(scikit_x))[0]

    def labels(self):
        return self.label_encoder.classes_

    def predict_scores(self, x, i, y):
        return self.lr.predict_log_proba(self.sklearn_repr(x, i, y))[0]

    def predict_scores_hist(self, x, i, hist):
        scikit_x = self.vectorizer.transform([self.feat(x, i, padded_history(hist, 1, self.order))])
        return self.lr.predict_log_proba(scikit_x)[0]

    def predict(self, data):
        result = []
        for x, y in data:
            y_guess = []
            for i in range(0, len(x)):
                prediction = self.predict_next(x, i, y_guess)
                y_guess += prediction
            result.append(y_guess)
        return result


import pycrfsuite as pycrf


def to_item_sequence(x, feat):
    return pycrf.ItemSequence([feat(x, i) for i in range(0, len(x))])


def add_to_trainer(trainer, data, feat):
    for x, y in data:
        trainer.append(to_item_sequence(x, feat), y)


class CRFSequenceLabeler:
    def __init__(self, feat, data, **crf_params):
        self.feat = feat  # lambda x, i: {**feat(x, i), 'b_bias': 'True'}
        self.trainer = pycrf.Trainer(verbose=False)
        self.trainer.set_params({
            **crf_params,
            #     'c1': 0.0,   # coefficient for L1 penalty
            'c2': 0.1,  # coefficient for L2 penalty
            #     'max_iterations': 50,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        add_to_trainer(self.trainer, data, self.feat)
        import tempfile
        f = tempfile.NamedTemporaryFile(delete=True)
        self.trainer.train(f.name)
        self.tagger = pycrf.Tagger()
        self.tagger.open(f.name)

    def predict(self, data):
        return [self.tagger.tag(to_item_sequence(x, self.feat)) for x, _ in data]


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
    for pair, y_guess in zip(gold, guess):
        y = pair[1] if isinstance(pair, tuple) else pair
        for i in range(0, len(y)):
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


def plot_confusion_matrix(gold, guess, normalise=False):
    util.plot_confusion_matrix_dict(confusion_matrix_dict(gold, guess, normalise))


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


def render_beam_history(history, instance, end=None, begin=None):
    x, y = instance
    end = len(x) if end is None else end
    begin = 0 if begin is None else begin
    elements = []
    for beam in history[begin:end]:
        words = to_td_seq(x, begin, end)
        gold = to_td_seq(y, begin, end)
        hypotheses = "<tr>" + "</tr><tr>".join(
            [to_td_seq(cand + ["{:.2f}".format(score)], begin, len(cand) + 1) for cand, score in beam]) + "</tr>"
        element = """
        <table>
          <tr>{words}</tr>
          <tr>{gold}</tr>
          {hypotheses}
        </table>
        """.format(words=words, gold=gold, hypotheses=hypotheses)
        elements.append(element)
    return util.Carousel(elements)


def top_keys(dictionary, width=2):
    sorted_keys_to_keep = sorted(dictionary.keys(), key=lambda k: -dictionary[k])[:width]
    return sorted_keys_to_keep


def drop_keys(dictionary, keys):
    for k in keys:
        del dictionary[k]


def keep_keys(dictionary, keys):
    for k in set(dictionary.keys()) - set(keys):
        del dictionary[k]


def convert_alpha_beta_to_history(x, alpha, beta):
    # construct the sequences corresponding to the alpha and beta maps
    beams = [defaultdict(lambda: [])]
    for i in range(0, len(x)):
        beam = {}
        for l, s in alpha[i].items():
            beam[l] = beams[-1][beta[i][l]] + [l]
        beams.append(beam)
    del beams[0]
    history = [[([], 0.0)]]
    for i in range(0, len(x)):
        beam = []
        for l, s in sorted(alpha[i].items(), key=lambda t: -t[1]):
            beam.append((beams[i][l], s))
        history.append(beam)
    return history


def prune_alpha_beta(alpha, beta, width):
    top = top_keys(alpha, width)
    keep_keys(alpha, top)
    keep_keys(beta, top)
