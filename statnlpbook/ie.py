#!/usr/bin/env python3

__author__ = 'Isabelle Augenstein'

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import collections
from tfutil import *
import random

def readLabelledPatternData(filepath="../data/ie/ie_bootstrap_patterns.txt"):
    f = open(filepath, "r")
    patterns = []
    entpairs = []
    for l in f:
        label, pattern, entpair = l.strip().replace("    ", "\t").split("\t")
        patterns.append(pattern)
        entpair = entpair.strip("['").strip("']").split("', '")
        entpairs.append(entpair)
    return patterns, entpairs


def readPatternData(filepath="../data/ie/ie_patterns.txt"):
    f = open(filepath, "r")
    patterns = []
    entpairs = []
    for l in f:
        pattern, entpair = l.strip().replace("    ", "\t").split("\t")
        patterns.append(pattern)
        entpair = entpair.strip("['").strip("']").split("', '")
        entpairs.append(entpair)
    return patterns, entpairs


def readLabelledData(filepath="../data/ie/ie_training_data.txt"):
    f = open(filepath, "r")
    sents = []
    entpairs = []
    labels = []
    for l in f:
        label, pattern, entpair = l.strip().replace("    ", "\t").split("\t")
        labels.append(label)
        sents.append(pattern)
        entpair = entpair.strip("['").strip("']").split("', '")
        entpairs.append(entpair)
    return sents, entpairs, labels


def readDataForDistantSupervision(filepath="../data/ie/ie_training_data.txt"):
    f = open(filepath, "r")
    unlab_sents = []
    unlab_entpairs = []
    kb_entpairs = []
    for l in f:
        label, sent, entpair = l.strip().replace("    ", "\t").split("\t")
        entpair = entpair.strip("['").strip("']").split("', '")
        # Define the positively labelled entity pairs as the KB ones, which are all for the same relation.
        # Normally these would come from an actual KB.
        if label != "NONE":
            kb_entpairs.append(entpair)
        unlab_sents.append(sent)
        unlab_entpairs.append(entpair)
    return kb_entpairs, unlab_sents, unlab_entpairs


def sentenceToShortPath(sent):
    """
    Returns the path between two arguments in a sentence, where the arguments have been masked
    Args:
        sent: the sentence
    Returns:
        the path between to arguments
    """
    sent_toks = sent.split(" ")
    indeces = [i for i, ltr in enumerate(sent_toks) if ltr == "XXXXX"]
    pattern = " ".join(sent_toks[indeces[0]+1:indeces[1]])
    return pattern


def patternExtraction(training_sentences, testing_sentences):
    """
    Given a set of patterns for a relation, searches for those patterns in other sentences
    Args:
        sent: training sentences with arguments masked, testing sentences with arguments masked
    Returns:
        the testing sentences which the training patterns appeared in
    """
    # convert training and testing sentences to short paths to obtain patterns
    training_patterns = set([sentenceToShortPath(test_sent) for test_sent in training_sentences])
    testing_patterns = [sentenceToShortPath(test_sent) for test_sent in testing_sentences]
    # look for training patterns in testing patterns
    testing_extractions = []
    for i, testing_pattern in enumerate(testing_patterns):
        if testing_pattern in training_patterns:
            testing_extractions.append(testing_sentences[i])
    return testing_extractions


def searchForPatternsAndEntpairsByPatterns(training_patterns, testing_patterns, testing_entpairs, testing_sentences):
    testing_extractions = []
    appearing_testing_patterns = []
    appearing_testing_entpairs = []
    for i, testing_pattern in enumerate(testing_patterns):
        if testing_pattern in training_patterns:
            testing_extractions.append(testing_sentences[i])
            appearing_testing_patterns.append(testing_pattern)
            appearing_testing_entpairs.append(testing_entpairs[i])
    return testing_extractions, appearing_testing_patterns, appearing_testing_entpairs


def searchForPatternsAndEntpairsByEntpairs(training_entpairs, testing_patterns, testing_entpairs, testing_sentences):
    testing_extractions = []
    appearing_testing_patterns = []
    appearing_testing_entpairs = []
    for i, testing_entpair in enumerate(testing_entpairs):
        if testing_entpair in training_entpairs:
            testing_extractions.append(testing_sentences[i])
            appearing_testing_entpairs.append(testing_entpair)
            appearing_testing_patterns.append(testing_patterns[i])
    return testing_extractions, appearing_testing_patterns, appearing_testing_entpairs


def distantlySupervisedLabelling(kb_entpairs, unlab_sents, unlab_entpairs):
    """
    Label instances using distant supervision assumption
    Args:
        kb_entpairs: entity pairs for a specific relation
        unlab_sents: unlabelled sentences with entity pairs anonymised
        unlab_entpairs: entity pairs which were anonymised in unlab_sents

    Returns: pos_train_sents, pos_train_enpairs, neg_train_sents, neg_train_entpairs

    """
    train_sents, train_entpairs, train_labels = [], [], []
    for i, unlab_entpair in enumerate(unlab_entpairs):
        if unlab_entpair in kb_entpairs:  # if the entity pair is a KB tuple, it is a positive example for that relation
            train_entpairs.append(unlab_entpair)
            train_sents.append(unlab_sents[i])
            train_labels.append("method used for task")
        else: # else, it is a negative example for that relation
            train_entpairs.append(unlab_entpair)
            train_sents.append(unlab_sents[i])
            train_labels.append("NONE")

    return train_sents, train_entpairs, train_labels


def bootstrappingExtraction(train_sents, train_entpairs, test_sents, test_entpairs):
    """
    Given a set of patterns and entity pairs for a relation, extracts more patterns and entity pairs iteratively
    Args:
        train_sents: training sentences with arguments masked
        train_entpairs: training entity pairs
        test_sents: testing sentences with arguments masked
        test_entpairs: testing entity pairs
    Returns:
        the testing sentences which the training patterns or any of the inferred patterns appeared in
    """

    # convert training and testing sentences to short paths to obtain patterns
    train_patterns = set([sentenceToShortPath(test_sent) for test_sent in train_sents])
    test_patterns = [sentenceToShortPath(test_sent) for test_sent in test_sents]
    test_extracts = []

    # iteratively get more patterns and entity pairs
    for i in range(0, 5):
        print("Number extractions at iteration", str(i), ":", str(len(test_extracts)))
        print("Number patterns at iteration", str(i), ":", str(len(train_patterns)))
        print("Number entpairs at iteration", str(i), ":", str(len(train_entpairs)))
        # get more patterns and entity pairs
        test_extracts_p, ext_test_patterns_p, ext_test_entpairs_p = searchForPatternsAndEntpairsByPatterns(train_patterns, test_patterns, test_entpairs, test_sents)
        test_extracts_e, ext_test_patterns_e, ext_test_entpairs_e = searchForPatternsAndEntpairsByEntpairs(train_entpairs, test_patterns, test_entpairs, test_sents)
        # add them to the existing entity pairs for the next iteration
        train_patterns.update(ext_test_patterns_p)
        train_patterns.update(ext_test_patterns_e)
        train_entpairs.extend(ext_test_entpairs_p)
        train_entpairs.extend(ext_test_entpairs_e)
        test_extracts.extend(test_extracts_p)
        test_extracts.extend(test_extracts_e)

    return test_extracts



def featTransform(sents_train, sents_test):
    cv = CountVectorizer()
    cv.fit(sents_train)
    print(cv.get_params())
    features_train = cv.transform(sents_train)
    features_test = cv.transform(sents_test)
    return features_train, features_test, cv


def model_train(feats_train, labels):
    # s(f(x), g(x)) + loss function handled by this model
    model = LogisticRegression(penalty='l2')
    model.fit(feats_train, labels)
    return model


def predict(model, features_test):
    """Find the most compatible output class given the input `x` and parameter `theta`"""
    preds = model.predict(features_test)
    #preds_prob = model.predict_proba(features_test)  # probabilities instead of classes
    return preds


def supervisedExtraction(train_sents, train_entpairs, train_labels, test_sents, test_entpairs):
    """
    Given pos/neg training instances, train a logistic regression model with simple BOW features and predict labels on unseen test instances
    Args:
        train_sents: training sentences with arguments masked
        train_entpairs: training entity pairs
        train_labels: labels of training instances
        test_sents: testing sentences with arguments masked
        test_entpairs: testing entity pairs
    Returns:
        predictions for the testing sentences
    """

    # convert training and testing sentences to short paths to obtain patterns
    train_patterns = [sentenceToShortPath(test_sent) for test_sent in train_sents]
    test_patterns = [sentenceToShortPath(test_sent) for test_sent in test_sents]

    # extract features
    features_train, features_test, cv = featTransform(train_patterns, test_patterns)

    # train model
    model = model_train(features_train, train_labels)

    # show most common features
    show_most_informative_features(cv, model)

    # get predictions
    predictions = predict(model, features_test)

    # show the predictions
    for tup in zip(predictions, test_sents, test_entpairs):
        print(tup)

    return predictions


def distantlySupervisedExtraction(kb_entpairs, unlab_sents, unlab_entpairs, test_sents, test_entpairs):
    train_sents, train_entpairs, train_labels = distantlySupervisedLabelling(kb_entpairs, unlab_sents, unlab_entpairs)
    supervisedExtraction(train_sents, train_entpairs, train_labels, test_sents, test_entpairs)


def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))


def build_dataset(words, vocabulary_size=5000000, min_count=1):
    """
    Build vocabulary, code based on tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    :param words: list of words in corpus
    :param vocabulary_size: max vocabulary size
    :param min_count: min count for words to be considered
    :return: counts, dictionary mapping words to indeces, reverse dictionary
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        if _ >= min_count:# or _ == -1:  # that's UNK only
            dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print("Final vocab size:", len(dictionary))
    return count, dictionary, reverse_dictionary


def transform_dict(dictionary, words, maxlen):
    """
    Transform list of tokens, add padding to maxlen
    :param dictionary: dict which maps tokens to integer indices
    :param words: list of tokens
    :param maxlen: maximum length
    :return: transformed tweet, as numpy array
    """
    data = list()
    for i in range(0, maxlen-1):  #range(0, len(words)-1):
        if i < len(words):
            word = words[i]
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
        else:
            index = 0
        data.append(index)
    return np.asarray(data)


def transf_labels(labels):
    # transforming labels
    labels_t = []
    for lab in training_labels:
        v = np.zeros(2)
        if lab == 'NONE':
            ix = 1
        else:
            ix = 0
        v[ix] = 1
        labels_t.append(v)

    return labels_t


def vectorise_data(training_sents, training_entpairs, training_labels, testing_sents, testing_entpairs):

    labels = transf_labels(training_labels)

    #training_toks = [sentenceToShortPath(t).split(" ") for t in training_sents]  # this version doesn't work so well as shortest path is very short - show that
    #testing_toks = [sentenceToShortPath(t).split(" ") for t in testing_sents] # this version doesn't work so well as shortest path is very short - show that
    training_toks = [t.split(" ") for t in training_sents]
    testing_toks = [t.split(" ") for t in testing_sents]

    training_ent_toks = [" ".join(t).split(" ") for t in training_entpairs]
    testing_ent_toks = [" ".join(t).split(" ") for t in testing_entpairs]

    lens_rel = [len(s) for s in training_toks]
    lens_ents = [len(s) for s in training_ent_toks]
    print("Max sentence length:", max(lens_rel))
    print("Max entity length:", max(lens_ents))

    count_rels, dictionary_rels, reverse_dictionary_rels = build_dataset(
        [token for senttoks in training_ent_toks for token in senttoks])

    count_ents, dictionary_ents, reverse_dictionary_ents = build_dataset(
        [token for senttoks in training_toks for token in senttoks])

    transf_rels_train = [transform_dict(dictionary_rels, senttoks, max(lens_rel)) for senttoks in training_toks]
    transf_ents_train = [transform_dict(dictionary_ents, senttoks, max(lens_ents)) for senttoks in training_ent_toks]  # this needs to have one dimensionality more for the neg data

    transf_rels_test = [transform_dict(dictionary_rels, senttoks, max(lens_rel)) for senttoks in testing_toks]
    transf_ents_test = [transform_dict(dictionary_ents, senttoks, max(lens_ents)) for senttoks in testing_ent_toks]

    vocab_size_rels = len(dictionary_rels)
    vocab_size_ents = len(dictionary_ents)

    neg_ents_train = random.sample(transf_ents_train, len(transf_ents_train)) # negatively sample some for training
    train_ents = [(pos, neg) for pos, neg in zip(transf_ents_train, neg_ents_train)]

    test_ents = [transf_ents_test] * len(transf_ents_test) # for testing, the targets are all the targets


    #train_target_values = [(pos, neg) for pos, neg in zip(labels, [np.array([1.0, 0.0]) for _ in range(0, len(labels))])]

    train_target_values = labels

    #return transf_rels_train, transf_ents_train, transf_rels_test, transf_ents_test, labels, testing_toks, testing_ent_toks, vocab_size_rels, vocab_size_ents
    return transf_rels_train, train_ents, transf_rels_test, test_ents, train_target_values, testing_toks, testing_ent_toks, vocab_size_rels, vocab_size_ents



def create_dense_embedding(ids, repr_dim, num_symbols, name):
    """
    :param ids: tensor [d1, ... ,dn] of int32 symbols
    :param repr_dim: dimension of embeddings
    :param num_symbols: number of symbols
    :return: [d1, ... ,dn,repr_dim] tensor representation of symbols.
    """
    embeddings = tf.Variable(tf.random_normal((num_symbols, repr_dim)), name=name)
    encodings = tf.gather(embeddings, ids)  # [batch_size, repr_dim]
    return encodings


def create_dot_product_scorer(rel_encodings, cand_encodings):
    """

    :param rel_encodings: [batch_size, enc_dim] tensor of relation representations
    :param cand_encodings: [batch_size, num_candidates, enc_dim] tensor of candidate encodings
    :return: a [batch_size, num_candidate] tensor of scores for each candidate
    """
    return tf.reduce_sum(tf.expand_dims(rel_encodings, 1) * cand_encodings, 2)


def create_softmax_loss(scores, target_values):
    """

    :param scores: [batch_size, num_candidates] logit scores
    :param target_values: [batch_size, num_candidates] vector of 0/1 target values.
    :return: [batch_size] vector of losses (or single number of total loss).
    """
    return tf.nn.softmax_cross_entropy_with_logits(scores, target_values)


def create_model_f_reader(batch_size, max_cand_seq_length, max_rel_seq_length, repr_dim, vocab_size_rels, vocab_size_cands):
    """
    Create a ModelF reader.
    :param options: 'repr_dim', dimension of representation .
    :param reference_data: the data to determine the question / answer candidate symbols.
    :return: ModelF
    """
    relations = tf.placeholder(tf.int32, [batch_size, max_rel_seq_length], name='relations')
    candidates = tf.placeholder(tf.int32, [batch_size, None, max_cand_seq_length], name="candidates")
    rel_encoding = create_dense_embedding(relations, repr_dim, vocab_size_rels, 'rel_emb')
    cand_encoding = create_dense_embedding(candidates, repr_dim, vocab_size_cands, 'cand_embed')
    rel_encoding = tf.reduce_sum(rel_encoding, 1)  # [batch_size, num_rels, repr_dim]
    cand_encoding = tf.reduce_sum(cand_encoding, 2)  # [batch_size, num_candidates, repr_dim]
    scores = create_dot_product_scorer(rel_encoding, cand_encoding)
    return scores, [relations, candidates]


def universalSchemaExtraction(training_rels, training_entpairs, testing_rels, testing_entpairs, train_target_values, vocab_size_rels, vocab_size_ents):

    batch_size = 5
    repr_dim = 10
    learning_rate = 0.001
    max_epochs = 21
    target_size = 2 # binary relation classification
    max_rel_seq_length = len(training_rels[0])
    max_cand_seq_length = len(training_entpairs[0][0])

    target_values = tf.placeholder(tf.float32, [batch_size, target_size], name="target_values")

    scores, placeholders = create_model_f_reader(batch_size, max_cand_seq_length, max_rel_seq_length, repr_dim, vocab_size_rels,
                          vocab_size_ents)

    loss = create_softmax_loss(scores, target_values)

    data = [np.asarray(training_rels), np.asarray(training_entpairs), np.asarray(train_target_values)]

    optimizer = tf.train.AdamOptimizer(learning_rate)
    batcher = BatchBucketSampler(data, batch_size)

    placeholders += [target_values]

    with tf.Session() as sess:
        trainer = Trainer(optimizer, max_epochs)

        trainer(batcher=batcher, placeholders=placeholders, loss=loss, model=scores, session=sess)

    # todo: show results on test


if __name__ == '__main__':
    training_patterns, training_entpairs = readLabelledPatternData()
    testing_patterns, testing_entpairs = readPatternData()

    # for relation extraction with patterns
    #patternExtraction(training_patterns, testing_patterns)

    # for relation extraction with bootstrapping
    #bootstrappingExtraction(training_patterns, training_entpairs, testing_patterns, testing_entpairs)

    training_sents, training_entpairs, training_labels = readLabelledData()
    #supervisedExtraction(training_sents, training_entpairs, training_labels, testing_patterns, testing_entpairs)

    #kb_entpairs, unlab_sents, unlab_entpairs = readDataForDistantSupervision()
    #distantlySupervisedExtraction(kb_entpairs, unlab_sents, unlab_entpairs, testing_patterns, testing_entpairs)

    np.random.seed(1337)
    tf.set_random_seed(1337)

    transf_rels_train, transf_ents_train, transf_rels_test, transf_ents_test, labels, testing_toks, testing_ent_toks, vocab_size_rels, vocab_size_ents = vectorise_data(training_sents, training_entpairs, training_labels, testing_patterns, testing_entpairs)

    universalSchemaExtraction(transf_rels_train, transf_ents_train, transf_rels_test, transf_ents_test, labels, vocab_size_rels, vocab_size_ents)