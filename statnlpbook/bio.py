from collections import defaultdict
from IPython.core.display import HTML
import statnlpbook.transition as transition
from os import listdir
from os.path import isfile, join
import json
import os
import pandas as pd


class Sentence:
    """
    A representation of a sentence.
    """

    def __init__(self, sent, events):
        """
        Construct a new sentence.
        Args:
            sent: a dictionary loaded from the list of elements in the json files of `data/bionlp/train`.
            See for example: `data/bionlp/train/PMC-1310901-00-TIAB.json`
            events: a list of `Event` objects.
        """
        self.events = events
        self.tokens = sent['tokens']
        self.dependencies = sent['deps']
        self.mentions = sent['mentions']
        self.is_protein = defaultdict(bool)
        for m in self.mentions:
            for i in range(m['begin'], m['end']):
                self.is_protein[i] = True

        self.children = defaultdict(list)
        self.parents = defaultdict(list)
        for dep in self.dependencies:
            self.children[dep['head']].append((dep['mod'], dep['label']))
            self.parents[dep['mod']].append((dep['head'], dep['label']))

    def _repr_html_(self):
        return " ".join([t['word'] for t in self.tokens])


class EventCandidate:
    """
    An EventCandidate specifies a trigger word candidate within a given sentence, together with a set of
    candidate argument spans.
    """

    def __init__(self, sent: Sentence, trigger_index: int, argument_candidate_spans):
        """
        Constructs a new EventCandidate.
        Args:
            sent: a `Sentence` object.
            trigger_index: the index of the event trigger candidate within the sentence.
            argument_candidate_spans: a list of (begin,end) pairs that denote the spans in the sentence
            which are argument candidates.
        """
        self.sent = sent
        self.argument_candidate_spans = argument_candidate_spans
        self.trigger_index = trigger_index

    def __str__(self):
        return str((self.sent, self.argument_candidate_spans, self.trigger_index))

    def _repr_html_(self):
        return render_event(self).data


def load_assignment2_training_data(path_to_dataset_folder):
    """
    Load event candidates and they event labels from the specified folder.
    Args:
        path_to_dataset_folder: the path to the folder that stores the dataset you want to load.

    Returns:
        a list of (event_candidate,label) pairs.
    """
    training_corpus = load_corpus(path_to_dataset_folder)
    event_corpus, _ = unzip_events_and_labels(training_corpus)
    return event_corpus


class EventLabels:
    """
    A label for an event and labels for all its argument candidates. **NOT NEEDED IN ASSIGNMENT**, but
    feel free to look at.
    """

    def __init__(self, event_label, arg_labels):
        self.event_label = event_label
        self.arg_labels = arg_labels

    def __repr__(self):
        return self.event_label + ": " + ", ".join(self.arg_labels)


class SentenceLabels:
    """
    A container class for all structured labels in a sentence.
    """

    def __init__(self, event_labels):
        self.event_labels = event_labels

    def __repr__(self):
        return "\n".join([str(e) for e in self.event_labels])


def render_event(event, label=None):
    """
    Produces an HTML rendering of an event, including sentence and argument candidates and proteins in sentence.
    Args:
        event: the event candidate to render.
        label: an optional event label to render as well. Set to `None` if no label should be shown.

    Returns:
        an HTML representation of the event.
    """
    tokens = event.sent.tokens
    prefixes = defaultdict(list)
    postfixes = defaultdict(list)
    prefixes[event.trigger_index].append("<font color='green'>")
    postfixes[event.trigger_index].append("</font>")
    for b, e in event.argument_candidate_spans:
        prefixes[b].append("<font color='red'>[</font>")
        postfixes[e - 1].append("<font color='red'>]</font>")
    if label is not None:
        postfixes[event.trigger_index].insert(0, ":<b>" + label + "</b>")

    for mention in event.sent.mentions:
        prefixes[mention['begin']].append("<font color='blue'>[")
        postfixes[mention['end'] - 1].insert(0, "]</font>")

    def render_token(i):
        word = tokens[i]['word']
        return "".join(prefixes[i]) + word + "".join(postfixes[i])

    words = " ".join([render_token(i) for i in range(0, len(tokens))])
    return HTML(words)


def find_errors(gold_label, guess_label, data, predictions):
    """
    Searches for cases where a `gold_label` was labelled as `guess_label`, using gold `data`, and guess `predictions`.
    Args:
        gold_label: the label that the error should have been given.
        guess_label: the label that model guessed instead (could be identical if you are looking for correct instances).
        data: the gold data, a list of (event_candidate,label) pairs.
        predictions: a list of predicted labels, in order corresponding to `data`

    Returns:
        all triples `(x,y_gold,y_guess)` where `x` is an event candidate for which the gold label `y_gold` was `gold_label`,
        and the guess label `y_guess' was 'guess_label.

    """
    result = []
    for (x, y_gold), y_guess in zip(data, predictions):
        if y_gold == gold_label and y_guess == guess_label:
            result.append((x, y_gold, y_guess))
    return result


def show_event_error(x, y_gold, y_guess):
    """
    Renders an error by rendering the event candidate, the sentence and the dependency parse, together with
    gold and guess labels.
    Args:
        x: the event candidate.
        y_gold: the correct label.
        y_guess: the predicted label.

    Returns:
        An HTML representation of the error.
    """
    label_info = pd.DataFrame([(y_gold, y_guess)], columns=("Gold", "Guess")).to_html(header=True, index=False)
    words = [{'text': t['word'], 'tag': t['pos']} for t in x.sent.tokens]
    arcs = [{'dir': 'right', 'start': a['head'], 'label': a['label'], 'end': a['mod']} if a['head'] < a['mod'] else
            {'dir': 'left', 'start': a['mod'], 'label': a['label'], 'end': a['head']} for a in x.sent.dependencies
            ]
    tree = transition.DependencyTree(arcs, words)
    return HTML(label_info + x._repr_html_() + tree._repr_html_())


def render_dependencies(sent):
    """
    Renders dependency graph of the sentence `sent`.
    Args:
        sent: the sentence to render the dependency parse for.

    Returns:
        HTML representation of the parse, using `displaCy`.
    """
    words = [{'text': t['word'], 'tag': t['pos']} for t in sent.tokens]
    arcs = [{'dir': 'right', 'start': a['head'], 'label': a['label'], 'end': a['mod']} if a['head'] < a['mod'] else
            {'dir': 'left', 'start': a['mod'], 'label': a['label'], 'end': a['head']} for a in sent.dependencies
            ]
    return HTML(transition.DependencyTree(arcs, words)._repr_html_())


def load_document(doc_dict, event_filter=lambda e: True, arg_filter=lambda a: True):
    """
    Loads a json event document.
    Args:
        doc_dict: the json document as dictionary.
        event_filter: a function that decides whether a particular event should be loaded or not.
        arg_filter: a function that decideds whether a particular event argument candidate should be loaded.

    Returns:
        A list of pairs of `Sentence` and `SentenceLabels`.
    """
    result = []
    for sent_dict in doc_dict['sentences']:
        events = []
        event_labels = []
        for event_dict in sent_dict['eventCandidates']:
            arg_labels = []
            arg_spans = []
            trigger_index = event_dict['begin']
            event_label = event_dict['gold']
            for arg in event_dict['arguments']:
                arg_label = arg['gold']
                if arg_filter(arg_label):
                    arg_labels.append(arg_label)
                    arg_spans.append((arg['begin'], arg['end']))
            event = EventCandidate(None, trigger_index, arg_spans)
            if event_filter(event) and len(arg_labels) > 0 and arg_spans[0][0] != trigger_index:
                events.append(event)
                event_labels.append(EventLabels(event_label, arg_labels))
        sentence = Sentence(sent_dict, events)
        sentence_labels = SentenceLabels(event_labels)
        for e in events:
            e.sent = sentence
        result.append((sentence, sentence_labels))
    return result


def load_corpus(directory, filter_events=lambda e: True, filter_arguments=lambda a: True):
    """
    Loads a corpus of events and their structured labels.
    Args:
        directory: The directory to load the events from.
        filter_events: Function to decide whether an event should be loaded.
        filter_arguments: Function to decide whether an event argument candidate should be loaded.

    Returns:
        List of pairs of `Sentence` and `SentenceLabels` object.
    """
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    result = []
    for file_name in sorted(files):
        with open(directory + os.sep + file_name) as file:
            data = json.load(file)
            result += load_document(data, filter_events, filter_arguments)
    return result


def unzip_events_and_labels(corpus):
    """
    Unzips the structured sentence labels into two lists: of (event_candidate,label) pairs,
    and nested ((event_candidate,argument_index),argument_label) pairs.
    Args:
        corpus: a list of `Sentence`,`SentenceLabel` pairs.

    Returns:
        a pair `event_corpus, argument_corpus`, where `event_corpus` is a list
        of pairs of `event_candidate` and `label`, and `argument_corpus` is a list of
        `(event_candidate,argument_index),argument_label)` pairs that indicate the labels of the
        `argument_index`-th argument in event `event_candidate`.
    """
    event_data = [(event, e.event_label)
                  for x, y in corpus
                  for event, e in zip(x.events, y.event_labels)]
    arg_data = [((event, arg_index), arg_label)
                for x, y in corpus
                for event, e in zip(x.events, y.event_labels)
                for arg_index, arg_label in enumerate(e.arg_labels)]
    return event_data, arg_data


def create_confusion_matrix(data, predictions):
    """
    Creates a confusion matrix that counts for each gold label how often it was labelled by what label
    in the predictions.
    Args:
        data: a list of gold (x,y) pairs.
        predictions: a list of y labels, same length and with matching order.

    Returns:
        a `defaultdict` that maps `(gold_label,guess_label)` pairs to their prediction counts.
    """
    confusion = defaultdict(int)
    for (x, y_gold), y_guess in zip(data, predictions):
        confusion[(y_gold, y_guess)] += 1
    return confusion


def evaluate(conf_matrix, label_filter=None):
    """
    Evaluate Precision, Recall and F1 based on a confusion matrix as produced by `create_confusion_matrix`.
    Args:
        conf_matrix: a confusion matrix in form of a dictionary from `(gold_label,guess_label)` pairs to counts.
        label_filter: a set of gold labels to consider. If set to `None` all labels are considered.

    Returns:
        Precision, Recall, F1 triple.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for (gold, guess), count in conf_matrix.items():
        if label_filter is None or gold in label_filter or guess in label_filter:
            if gold == 'None' and guess != gold:
                fp += count
            elif gold == 'None' and guess == gold:
                tn += count
            elif gold != 'None' and guess == gold:
                tp += count
            elif gold != 'None' and guess == 'None':
                fn += count
            else:  # both gold and guess are not-None, but different
                fp += count if label_filter is None or guess in label_filter else 0
                fn += count if label_filter is None or gold in label_filter else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec * recall > 0 else 0.0
    return prec, recall, f1


def full_evaluation_table(confusion_matrix):
    """
    Produce a pandas data-frame with Precision, F1 and Recall for all labels.
    Args:
        confusion_matrix: the confusion matrix to calculate metrics from.

    Returns:
        a pandas Dataframe with one row per gold label, and one more row for the aggregate of all labels.
    """
    labels = sorted(list({l for l, _ in confusion_matrix.keys()} | {l for _, l in confusion_matrix.keys()}))
    gold_counts = defaultdict(int)
    guess_counts = defaultdict(int)
    for (gold_label, guess_label), count in confusion_matrix.items():
        if gold_label != "None":
            gold_counts[gold_label] += count
            gold_counts["[All]"] += count
        if guess_label != "None":
            guess_counts[guess_label] += count
            guess_counts["[All]"] += count

    result_table = []
    for label in labels:
        if label != "None":
            result_table.append((label, gold_counts[label], guess_counts[label], *evaluate(confusion_matrix, {label})))

    result_table.append(("[All]", gold_counts["[All]"], guess_counts["[All]"], *evaluate(confusion_matrix)))
    return pd.DataFrame(result_table, columns=('Label', 'Gold', 'Guess', 'Precision', 'Recall', 'F1'))
