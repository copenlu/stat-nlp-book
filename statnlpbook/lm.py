import abc

import numpy as np
import math
import collections


class LanguageModel(metaclass=abc.ABCMeta):
    """
    Args:
        vocab: the vocabulary underlying this language model. Should be a set of words.
        order: history length (-1).
    """

    def __init__(self, vocab, order):
        self.vocab = vocab
        self.order = order

    @abc.abstractmethod
    def probability(self, word, *history):
        """
        Args:
            word: the word we need the probability of
            history: words to condition on.
        Returns:
            the probability p(w|history)
        """
        pass


class UniformLM(LanguageModel):
    """
    A uniform language model that assigns the same probability to each word in the vocabulary.
    """

    def __init__(self, vocab):
        super().__init__(vocab, 1)

    def probability(self, word, *history):
        return 1.0 / len(self.vocab) if word in self.vocab else 0.0


def sample(lm, init, amount):
    """
    Sample from a language model.
    Args:
        lm: the language model
        init: the initial sequence of words to condition on
        amount: how long should the sampled sequence be
    """
    words = list(lm.vocab)
    result = []
    result += init
    for _ in range(0, amount):
        history = result[-(lm.order - 1):]
        probs = [lm.probability(word, *history) for word in words]
        sampled = np.random.choice(words, p=probs)
        result.append(sampled)
    return result


def perplexity(lm, data):
    """
    Calculate the perplexity of the language model given the provided data.
    Args:
        lm: a language model.
        data: the data to calculate perplexity on.

    Returns:
        the perplexity of `lm` on `data`.

    """
    log_prob = 0.0
    history_order = lm.order - 1
    for i in range(history_order, len(data)):
        history = data[i - history_order: i]
        word = data[i]
        p = lm.probability(word, *history)
        log_prob += math.log(p) if p > 0.0 else float("-inf")
    return math.exp(-log_prob / (len(data) - history_order))


OOV = '[OOV]'


def replace_OOVs(vocab, data):
    """
    Replace every word not within the vocabulary with the `OOV` symbol.
    Args:
        vocab: the reference vocabulary.
        data: the sequence of tokens to replace words within

    Returns:
        a version of `data` where each word not in `vocab` is replaced by the `OOV` symbol.
    """
    return [word if word in vocab else OOV for word in data]


def inject_OOVs(data):
    """
    Uses a heuristic to inject OOV symbols into a dataset.
    Args:
        data: the sequence of words to inject OOVs into.

    Returns: the new sequence with OOV symbols injected.
    """
    seen = set()
    result = []
    for word in data:
        if word in seen:
            result.append(word)
        else:
            result.append(OOV)
            seen.add(word)
    return result


class OOVAwareLM(LanguageModel):
    """
    This LM converts out of vocabulary tokens to a special OOV token before their probability is calculated.
    """

    def __init__(self, base_lm, oov_count, oov=OOV):
        super().__init__(base_lm.vocab, base_lm.order)
        self.base_lm = base_lm
        self.oov = oov
        self.oov_count = oov_count

    def probability(self, word, *history):
        actual_word, norm = (word, 1) if word in self.base_lm.vocab else (self.oov, self.oov_count)
        return self.base_lm.probability(actual_word, *history) / norm


class CountLM(LanguageModel):
    """
    A Language Model that uses counts of events and histories to calculate probabilities of words in context.
    """

    @abc.abstractmethod
    def counts(self, word_and_history):
        pass

    @abc.abstractmethod
    def norm(self, history):
        pass

    def probability(self, word, *history):
        sub_history = tuple(history[-(self.order - 1):]) if self.order > 1 else ()
        return self.counts((word,) + sub_history) / self.norm(sub_history)


class NGramLM(CountLM):
    def __init__(self, train, order):
        """
        Create an NGram language model.
        Args:
            train: list of training tokens.
            order: order of the LM.
        """
        super().__init__(set(train), order)
        self._counts = collections.defaultdict(float)
        self._norm = collections.defaultdict(float)
        for i in range(self.order, len(train)):
            history = tuple(train[i - self.order + 1: i])
            word = train[i]
            self._counts[(word,) + history] += 1.0
            self._norm[history] += 1.0

    def counts(self, word_and_history):
        return self._counts[word_and_history]

    def norm(self, history):
        return self._norm[history]


class LaplaceLM(CountLM):
    def __init__(self, base_lm, alpha):
        super().__init__(base_lm.vocab, base_lm.order)
        self.base_lm = base_lm
        self.alpha = alpha

    def counts(self, word_and_history):
        return self.base_lm.counts(word_and_history) + self.alpha

    def norm(self, history):
        return self.base_lm.norm(history) + self.alpha * len(self.base_lm.vocab)


class InterpolatedLM(LanguageModel):
    def __init__(self, main, backoff, alpha):
        super().__init__(main.vocab, main.order)
        self.main = main
        self.backoff = backoff
        self.alpha = alpha

    def probability(self, word, *history):
        return self.alpha * self.main.probability(word, *history) + \
               (1.0 - self.alpha) * self.backoff.probability(word, *history)


class StupidBackoff(LanguageModel):
    def __init__(self, main, backoff, alpha):
        super().__init__(main.vocab, main.order)
        self.main = main
        self.backoff = backoff
        self.alpha = alpha

    def probability(self, word, *history):
        return self.main.probability(word, *history) \
            if self.main.counts((word,) + tuple(history)) > 0 \
            else self.alpha * self.backoff.probability(word, *history)
