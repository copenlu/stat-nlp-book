#!/usr/bin/env python3

__author__ = 'Isabelle Augenstein'

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


def readLabelledPatternData(filepath="data/ie/ie_bootstrap_patterns.txt"):
    f = open(filepath, "r")
    patterns = []
    entpairs = []
    for l in f:
        label, pattern, entpair = l.strip().replace("    ", "\t").split("\t")
        patterns.append(pattern)
        entpair = entpair.strip("['").strip("']").split("', '")
        entpairs.append(entpair)
    return patterns, entpairs


def readPatternData(filepath="data/ie/ie_patterns.txt"):
    f = open(filepath, "r")
    patterns = []
    entpairs = []
    for l in f:
        pattern, entpair = l.strip().replace("    ", "\t").split("\t")
        patterns.append(pattern)
        entpair = entpair.strip("['").strip("']").split("', '")
        entpairs.append(entpair)
    return patterns, entpairs


def readLabelledData(filepath="data/ie/ie_training_data.txt"):
    f = open(filepath, "r")
    patterns = []
    entpairs = []
    labels = []
    for l in f:
        label, pattern, entpair = l.strip().replace("    ", "\t").split("\t")
        labels.append(label)
        patterns.append(pattern)
        entpair = entpair.strip("['").strip("']").split("', '")
        entpairs.append(entpair)
    return patterns, entpairs, labels


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


def supervisedExtraction(train_sents, train_labels, test_sents):
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
    for pair in zip(predictions, test_sents):
        print(pair)

    return predictions



def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))




if __name__ == '__main__':
    training_patterns, training_entpairs = readLabelledPatternData()
    testing_patterns, testing_entpairs = readPatternData()

    # for relation extraction with patterns
    #patternExtraction(training_patterns, testing_patterns)

    # for relation extraction with bootstrapping
    #bootstrappingExtraction(training_patterns, training_entpairs, testing_patterns, testing_entpairs)

    training_sents, training_entpairs, training_labels = readLabelledData()
    supervisedExtraction(training_sents, training_labels, testing_patterns)