#!/usr/bin/env python3

__author__ = 'Isabelle Augenstein'

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import collections
from tfutil import *
import random
import copy
import json

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


def read_few_rel_data(filepath="../data/ie/fewrel_val_wiki.json"):
    with open(filepath) as f:
        return json.load(f)


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


def bootstrappingExtraction(train_sents, train_entpairs, test_sents, test_entpairs, num_iter=6):
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
    train_patterns = set([sentenceToShortPath(s) for s in train_sents])
    #train_patterns.remove("in") # too general
    test_patterns = [sentenceToShortPath(s) for s in test_sents]
    test_extracts = []

    # iteratively get more patterns and entity pairs
    for i in range(0, num_iter):
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

    return test_extracts, test_entpairs



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
    count = [['UNK', 100000]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        if _ >= min_count:
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
    for i in range(0, maxlen):  #range(0, len(words)-1):
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


def reverse_dict_lookup(dictionary, indeces):
    res = []
    for i in indeces:
        word = dictionary[i]
        res.append(word)
    res.reverse()
    for w in copy.deepcopy(res):
        if w == 'UNK':
            res.remove(w)
        else:
            break
    res.reverse()
    if res[-1] == '||':
        res.append('UNK')
    return res




def split_labels_pos_neg(labels):
    neg_train_ids = []
    pos_train_ids = []
    for i, lab in enumerate(labels):
        if lab == "NONE":
            neg_train_ids.append(i)
        else:
            pos_train_ids.append(i)
    return pos_train_ids, neg_train_ids


def vectorise_data(training_sents, training_entpairs, training_kb_rels, testing_sents, testing_entpairs):

    pos_train_ids, neg_train_ids = split_labels_pos_neg(training_kb_rels + training_kb_rels)

    training_toks_pos = [t.split(" ") for i, t in enumerate(training_sents + training_kb_rels) if i in pos_train_ids]
    training_toks_neg = [t.split(" ") for i, t in enumerate(training_sents + training_kb_rels) if i in neg_train_ids]

    training_ent_toks_pos = [" || ".join(t).split(" ") for i, t in enumerate(training_entpairs + training_entpairs) if i in pos_train_ids]
    training_ent_toks_neg = [" || ".join(t).split(" ") for i, t in enumerate(training_entpairs + training_entpairs) if i in neg_train_ids]
    testing_ent_toks = [" || ".join(t).split(" ") for t in testing_entpairs]

    lens_rel = [len(s) for s in training_toks_pos + training_toks_neg]
    lens_ents = [len(s) for s in training_ent_toks_pos + training_ent_toks_neg + testing_ent_toks]
    print("Max relation length:", max(lens_rel))
    print("Max entity pair length:", max(lens_ents))

    count_rels, dictionary_rels, reverse_dictionary_rels = build_dataset(
        [token for senttoks in training_toks_pos + training_toks_neg for token in senttoks])

    count_ents, dictionary_ents, reverse_dictionary_ents = build_dataset(
        [token for senttoks in training_ent_toks_pos + training_ent_toks_neg for token in senttoks])

    rels_train_pos = [transform_dict(dictionary_rels, senttoks, max(lens_rel)) for senttoks in training_toks_pos]
    rels_train_neg = [transform_dict(dictionary_rels, senttoks, max(lens_rel)) for senttoks in training_toks_neg]
    ents_train_pos = [transform_dict(dictionary_ents, senttoks, max(lens_ents)) for senttoks in training_ent_toks_pos]
    ents_train_neg = [transform_dict(dictionary_ents, senttoks, max(lens_ents)) for senttoks in training_ent_toks_neg]

    ents_test_pos = [transform_dict(dictionary_ents, senttoks, max(lens_ents)) for senttoks in testing_ent_toks]
    ents_test_neg_samp = [random.choice(ents_train_neg) for _ in ents_test_pos]  # sample those from train as for that we have neg annotations

    vocab_size_rels = len(dictionary_rels)
    vocab_size_ents = len(dictionary_ents)

    ents_train_neg_samp = [random.choice(ents_train_neg) for _ in rels_train_neg] # Negatively sample some for training. Here we have some manually labelled neg ones, so we can sample from them.

    rels_test_pos = [transform_dict(dictionary_rels, training_toks_pos[-1], max(lens_rel)) for _ in testing_sents]  # for testing, we want to check if each unlabelled instance expresses the given relation "method for tas"
    rels_test_neg_samp = [random.choice(rels_train_neg) for _ in rels_test_pos]

    return rels_train_pos, rels_train_neg, ents_train_pos, ents_train_neg_samp, rels_test_pos, rels_test_neg_samp, \
           ents_test_pos, ents_test_neg_samp, vocab_size_rels, vocab_size_ents, max(lens_rel), max(lens_ents), \
           reverse_dictionary_rels, reverse_dictionary_ents



def create_model_f_reader(max_lens_rel, max_lens_ents, repr_dim, vocab_size_rels, vocab_size_ents):
    """
    Create a Model F Universal Schema reader (Tensorflow graph).
    Args:
        max_rel_seq_length: maximum sentence sequence length
        max_cand_seq_length: maximum candidate sequence length
        repr_dim: dimensionality of vectors
        vocab_size_rels: size of relation vocabulary
        vocab_size_cands: size of candidate vocabulary
    Returns:
        dotprod_pos: dot product between positive entity pairs and relations
        dotprod_neg: dot product between negative entity pairs and relations
        diff_dotprod: difference in dot product of positive and negative instances, used for BPR loss (optional)
        [relations_pos, relations_neg, ents_pos, ents_neg]: placeholders, fed in during training for each batch
    """

    # Placeholders (empty Tensorflow variables) for positive and negative relations and entity pairs
    # In each training epoch, for each batch, those will be set through mini batching

    relations_pos = tf.placeholder(tf.int32, [None, max_lens_rel],
                                   name='relations_pos')  # [batch_size, max_rel_seq_len]
    relations_neg = tf.placeholder(tf.int32, [None, max_lens_rel],
                                   name='relations_neg')  # [batch_size, max_rel_seq_len]

    ents_pos = tf.placeholder(tf.int32, [None, max_lens_ents], name="ents_pos")  # [batch_size, max_ent_seq_len]
    ents_neg = tf.placeholder(tf.int32, [None, max_lens_ents], name="ents_neg")  # [batch_size, max_ent_seq_len]

    # Creating latent representations of relations and entity pairs
    # latent feature representation of all relations, which are initialised randomly
    relation_embeddings = tf.Variable(tf.random_uniform([vocab_size_rels, repr_dim], -0.1, 0.1, dtype=tf.float32),
                                      name='rel_emb', trainable=True)

    # latent feature representation of all entity pairs, which are initialised randomly
    ent_embeddings = tf.Variable(tf.random_uniform([vocab_size_ents, repr_dim], -0.1, 0.1, dtype=tf.float32),
                                 name='cand_emb', trainable=True)

    # look up latent feature representation for relations and entities in current batch
    rel_encodings_pos = tf.nn.embedding_lookup(relation_embeddings, relations_pos)
    rel_encodings_neg = tf.nn.embedding_lookup(relation_embeddings, relations_neg)

    ent_encodings_pos = tf.nn.embedding_lookup(ent_embeddings, ents_pos)
    ent_encodings_neg = tf.nn.embedding_lookup(ent_embeddings, ents_neg)

    # our feature representation here is a vector for each word in a relation or entity
    # because our training data is so small
    # we therefore take the sum of those vectors to get a representation of each relation or entity pair
    rel_encodings_pos = tf.reduce_sum(rel_encodings_pos, 1)  # [batch_size, num_rel_toks, repr_dim]
    rel_encodings_neg = tf.reduce_sum(rel_encodings_neg, 1)  # [batch_size, num_rel_toks, repr_dim]

    ent_encodings_pos = tf.reduce_sum(ent_encodings_pos, 1)  # [batch_size, num_ent_toks, repr_dim]
    ent_encodings_neg = tf.reduce_sum(ent_encodings_neg, 1)  # [batch_size, num_ent_toks, repr_dim]

    # measuring compatibility between positive entity pairs and relations
    # used for ranking test data
    dotprod_pos = tf.reduce_sum(tf.multiply(ent_encodings_pos, rel_encodings_pos), 1)

    # measuring compatibility between negative entity pairs and relations
    dotprod_neg = tf.reduce_sum(tf.multiply(ent_encodings_neg, rel_encodings_neg), 1)

    # difference in dot product of positive and negative instances
    # used for BPR loss (ranking loss)
    diff_dotprod = tf.reduce_sum(
        tf.multiply(ent_encodings_pos, rel_encodings_pos) - tf.multiply(ent_encodings_neg, rel_encodings_neg), 1)

    return dotprod_pos, dotprod_neg, diff_dotprod, [relations_pos, relations_neg, ents_pos, ents_neg]


def universalSchemaExtraction(data):
    rels_train_pos, rels_train_neg, ents_train_pos, ents_train_neg_samp, rels_test_pos, rels_test_neg_samp, \
    ents_test_pos, ents_test_neg_samp, vocab_size_rels, vocab_size_ents, max_lens_rel, max_lens_ents, \
    dictionary_rels_rev, dictionary_ents_rev = data

    batch_size = 4
    repr_dim = 30
    learning_rate = 0.001
    max_epochs = 21

    dotprod_pos, dotprod_neg, diff_dotprod, placeholders = create_model_f_reader(max_lens_rel, max_lens_ents, repr_dim, vocab_size_rels,
                          vocab_size_ents)

    # logistic loss
    loss = tf.reduce_sum(tf.nn.softplus(-dotprod_pos)+tf.nn.softplus(dotprod_neg))

    # alternative: BPR loss
    #loss = tf.reduce_sum(tf.nn.softplus(diff_dotprod))

    data = [np.asarray(rels_train_pos), np.asarray(rels_train_neg), np.asarray(ents_train_pos), np.asarray(ents_train_neg_samp)]
    data_test = [np.asarray(rels_test_pos), np.asarray(rels_test_neg_samp), np.asarray(ents_test_pos), np.asarray(ents_test_neg_samp)]

    optimizer = tf.train.AdamOptimizer(learning_rate)
    batcher = BatchBucketSampler(data, batch_size)
    batcher_test = BatchBucketSampler(data_test, 1, test=True)

    with tf.Session() as sess:
        trainer = Trainer(optimizer, max_epochs)

        trainer(batcher=batcher, placeholders=placeholders, loss=loss, session=sess)

        test_scores = trainer.test(batcher=batcher_test, placeholders=placeholders, model=tf.nn.sigmoid(dotprod_pos), session=sess)

    # show predictions
    ents_test = [reverse_dict_lookup(dictionary_ents_rev, e) for e in ents_test_pos]
    rels_test = [reverse_dict_lookup(dictionary_rels_rev, r) for r in rels_test_pos]
    testresults = sorted(zip(test_scores, ents_test, rels_test), key=lambda t: t[0], reverse=True)  # sort for decreasing score

    print("Test predictions by decreasing probability:")
    for score, tup, rel in testresults:
        print('%f\t%s\tREL\t%s' % (score, " ".join(tup), " ".join(rel)))


if __name__ == '__main__':
    training_patterns, training_entpairs = readLabelledPatternData()
    testing_patterns, testing_entpairs = readPatternData()

    # for relation extraction with patterns
    #patternExtraction(training_patterns, testing_patterns)

    # for relation extraction with bootstrapping
    bootstrappingExtraction(training_patterns, training_entpairs, testing_patterns, testing_entpairs)

    #training_sents, training_entpairs, training_labels = readLabelledData()
    #supervisedExtraction(training_sents, training_entpairs, training_labels, testing_patterns, testing_entpairs)

    #kb_entpairs, unlab_sents, unlab_entpairs = readDataForDistantSupervision()
    #distantlySupervisedExtraction(kb_entpairs, unlab_sents, unlab_entpairs, testing_patterns, testing_entpairs)

    #np.random.seed(1337)
    #tf.set_random_seed(1337)

    #data = vectorise_data(training_sents, training_entpairs, training_labels, testing_patterns, testing_entpairs)
    #universalSchemaExtraction(data)
