import graphviz as gv
import functools
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
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
        node_id = "y" + str(i)
        factor_id = "f" + str(i)
        ys.node(node_id, **var_style)
        result.node(factor_id, **factor_style)
        result.edge(node_id, factor_id)
        result.edge(factor_id, "x")

    result.subgraph(ys)
    return result


def draw_mm_fg(length=3):
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
        train_classifier_X = self.vectorizer.fit_transform(to_classifier_x(train_data, feat))
        train_classifier_Y = self.label_encoder.fit_transform(to_classifier_y(train_data))
        self.lr = LogisticRegression(**lr_params)
        self.lr.fit_transform(train_classifier_X, train_classifier_Y)

    def plot_lr_weights(self, label, how_many=20):
        v_index = self.label_encoder.transform(label)
        v_weights = self.vectorizer.inverse_transform(self.lr.coef_)[v_index]
        sorted_weights = sorted(v_weights.items(), key=lambda t: t[1], reverse=True)
        return util.plot_bar_graph([w for _, w in sorted_weights[:how_many]],
                                   [f for f, _ in sorted_weights[:how_many]], rotation=45)

    def predict(self, data):
        # dev_classifier_Y = label_encoder.transform(to_classifier_y(dev))
        dev_classifier_output = self.label_encoder.inverse_transform(
            self.lr.predict(self.vectorizer.transform(to_classifier_x(data, self.feat))))
        dev_output = to_xy(data, dev_classifier_output)
        return dev_output

    def show_errors(self, data, predicate=lambda x, y, y_guess, i: True):
        guess = self.predict(data)
        for (x, y), y_guess in zip(data, guess):
            for i in range(0, len(y)):
                if y[i] != y_guess[i] and predicate(x, y, y_guess, i):
                    print("---------")
                    print("Gold:  {}".format(y[i]))
                    print("Guess: {}".format(y_guess[i]))
                    print(" ".join(x[max(0, i - 5):i]) + " [" + x[i] + "] " + " ".join(x[i + 1:min(i + 5, len(x))]))
                    print(self.feat(x, i))


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
