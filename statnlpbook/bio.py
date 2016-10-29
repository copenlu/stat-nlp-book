from collections import defaultdict
from IPython.core.display import HTML


class EventCandidate:
    def __init__(self, sent, trigger_index, argument_candidate_indices):
        self.sent = sent
        self.argument_candidate_indices = argument_candidate_indices
        self.trigger_index = trigger_index

    #         if trigger_index == 79:
    #             print(sent)

    def __str__(self):
        return str((self.sent, self.argument_candidate_indices, self.trigger_index))

    def _repr_html_(self):
        return render_event(self).data


class Sentence:
    def __init__(self, sent, events):
        self.events = events
        self.tokens = sent['tokens']
        self.dependencies = sent['deps']
        self.mentions = sent['mentions']

    def _repr_html_(self):
        return " ".join([t['word'] for t in self.tokens])


class EventLabels:
    def __init__(self, event_label, arg_labels):
        self.event_label = event_label
        self.arg_labels = arg_labels

    def __repr__(self):
        return self.event_label + ": " + ", ".join(self.arg_labels)


class SentenceLabels:
    def __init__(self, event_labels):
        self.event_labels = event_labels

    def __repr__(self):
        return "\n".join([str(e) for e in self.event_labels])


# TODO:
# * Extract Sentence objects with event candidates
# * load events and labels jointly,
# * support filter on events and arguments that produces consistent sentence/label pairs
# * label function works on sentences, returns [(label,[arg_labels])] list
# * SentenceLabelStructure -> EventLabelStructure -> (string, [string])
# * def wrap(sentences, event_labels:[string], argument_labels:[string]) -> [SentenceLabelStructure]
# * def unwrap(sentences, sentence_label_structures) -> ([string],[string])

def render_event(event, labels=None):
    tokens = event.sent.tokens
    prefixes = defaultdict(list)
    postfixes = defaultdict(list)
    prefixes[event.trigger_index].append("<font color='green'>")
    postfixes[event.trigger_index].append("</font>")
    for b, e in event.argument_candidate_indices:
        prefixes[b].append("<font color='red'>[</font>")
        postfixes[e - 1].append("<font color='red'>]</font>")
    if labels is not None:
        postfixes[event.trigger_index].insert(0, ":<b>" + labels.event_label + "</b>")
        for arg_index in range(len(labels.arg_labels)):
            if labels.arg_labels[arg_index] != 'None':
                e = event.argument_candidate_indices[arg_index][1]
                postfixes[e - 1].insert(0, ":<b>" + labels.arg_labels[arg_index] + "</b>")

    for mention in event.sent.mentions:
        prefixes[mention['begin']].append("<font color='blue'>[")
        postfixes[mention['end'] - 1].insert(0, "]</font>")

    def render_token(i):
        word = tokens[i]['word']
        return "".join(prefixes[i]) + word + "".join(postfixes[i])

    words = " ".join([render_token(i) for i in range(0, len(tokens))])
    return HTML(words)


def find_errors(gold_label, guess_label, data, predictions):
    result = []
    for (x, y_gold), y_guess in zip(data, predictions):
        if y_gold == gold_label and y_guess == guess_label:
            result.append((x, y_gold, y_guess))
    return result


from IPython.core.display import HTML
import statnlpbook.transition as transition


def show_event_error(x, y_gold, y_guess):
    label_info = pd.DataFrame([(y_gold, y_guess)], columns=("Gold", "Guess")).to_html(header=True, index=False)
    words = [{'text': t['word'], 'tag': t['pos']} for t in x.sent.tokens]
    arcs = [{'dir': 'right', 'start': a['head'], 'label': a['label'], 'end': a['mod']} if a['head'] < a['mod'] else
            {'dir': 'left', 'start': a['mod'], 'label': a['label'], 'end': a['head']} for a in x.sent.dependencies
            ]
    tree = transition.DependencyTree(arcs, words)
    return HTML(label_info + x._repr_html_() + tree._repr_html_())


def load_document(doc_dict, event_filter=lambda e: True, arg_filter=lambda a: True):
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


from os import listdir
from os.path import isfile, join
import json
import os


def load_corpus(directory, filter_events=lambda e: True, filter_arguments=lambda a: True):
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    result = []
    for file_name in files:
        with open(directory + os.sep + file_name) as file:
            data = json.load(file)
            result += load_document(data, filter_events, filter_arguments)
    return result


def unzip_events_and_labels(corpus):
    event_data = [(event, e.event_label)
                  for x, y in corpus
                  for event, e in zip(x.events, y.event_labels)]
    arg_data = [((event, arg_index), arg_label)
                for x, y in corpus
                for event, e in zip(x.events, y.event_labels)
                for arg_index, arg_label in enumerate(e.arg_labels)]
    return event_data, arg_data


def unzip_events(corpus):
    event_data = [event
                  for x in corpus
                  for event in x.events]
    arg_data = [(event, arg_index)
                for x in corpus
                for event in x.events
                for arg_index in range(0, len(event.argument_candidate_indices))]
    return event_data, arg_data


def zip_events(corpus, event_labels, arg_labels):
    result = []
    event_label_index = 0
    arg_label_index = 0

    def get_label(l):
        return l[1] if isinstance(l, tuple) else l

    for instance in corpus:
        if isinstance(instance, tuple):
            sent = instance[0]
        else:
            sent = instance
        print(type(sent))
        print(isinstance(sent, Sentence))
        assert (isinstance(sent, Sentence))
        sentence_event_labels = []
        for event in sent.events:
            event_label = get_label(event_labels[event_label_index])
            event_label_index += 1
            event_arg_labels = []
            for arg in event.argument_candidate_indices:
                event_arg_labels.append(get_label(arg_labels[arg_label_index]))
                arg_label_index += 1
            sentence_event_labels.append(EventLabels(event_label, event_arg_labels))
        result.append((sent, SentenceLabels(sentence_event_labels)))
    return result


def create_confusion_matrix(data, predictions):
    confusion = defaultdict(int)
    for (x, y_gold), y_guess in zip(data, predictions):
        confusion[(y_gold, y_guess)] += 1
    return confusion


def evaluate(conf_matrix, label_filter=None):
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
            elif gold != 'None' and guess != gold:
                fn += count
            elif gold != 'None' and guess == gold:
                tp += count
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec * recall > 0 else 0.0
    return prec, recall, f1


import pandas as pd


def full_evaluation_table(confusion_matrix):
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


import random

random_subsample = random.Random(0)


def subsample_nones(pairs, accept_none_probability):
    result = []
    for pair in pairs:
        if pair[1] != 'None' or random_subsample.random() <= accept_none_probability:
            result.append(pair)
    return result
