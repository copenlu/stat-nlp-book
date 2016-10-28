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
        prefixes[b].append("<font color='blue'>[")
        postfixes[e - 1].append("]</font>")
    if labels is not None:
        postfixes[event.trigger_index].insert(0, ":<b>" + labels.event_label + "</b>")
        for arg_index in range(len(labels.arg_labels)):
            if labels.arg_labels[arg_index] != 'None':
                e = event.argument_candidate_indices[arg_index][1]
                postfixes[e - 1].insert(0, ":<b>" + labels.arg_labels[arg_index] + "</b>")

    def render_token(i):
        word = tokens[i]['word']
        return "".join(prefixes[i]) + word + "".join(postfixes[i])

    words = " ".join([render_token(i) for i in range(0, len(tokens))])
    return HTML(words)


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
