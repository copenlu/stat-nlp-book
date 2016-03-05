import re


def sentence_segment(match_regex, tokens):
    """
    Splits a sequence of tokens into sentences, splitting wherever the given matching regular expression
    matches.

    Parameters
    ----------
    match_regex the regular expression that defines at which token to split.
    tokens the input sequence of string tokens.

    Returns
    -------
    a list of token lists, where each inner list represents a sentence.

    >>> tokens = ['the','man','eats','.','She', 'sleeps', '.']
    >>> sentence_segment(re.compile('\.'), tokens)
    [['the', 'man', 'eats', '.'], ['She', 'sleeps', '.']]
    """
    current = []
    sentences = [current]
    for tok in tokens:
        current.append(tok)
        if match_regex.match(tok):
            current = []
            sentences.append(current)
    if not sentences[-1]:
        sentences.pop(-1)
    return sentences
