def create_translation_table(source_vocab, target_vocab, fixed=None):
    if fixed is None:
        fixed = {}
    alpha = {}
    for s in source_vocab:
        for t in target_vocab:
            if t in fixed:
                alpha[s, t] = 1.0 if fixed[t] == s else 0.0
            else:
                alpha[s, t] = 1.0 / len(source_vocab)
    return alpha


def create_distortion_table(max_length, fixed=None):
    if fixed is None:
        fixed = {}
    beta = {}
    for ti in range(0, max_length):
        for si in range(0, max_length):
            for lt in range(1, max_length + 1):
                for ls in range(1, max_length + 1):
                    if ti in fixed:
                        beta[ti, si, lt, ls] = 1.0 if fixed[ti] == si else 0.0
                    else:
                        beta[ti, si, lt, ls] = 1.0 / lt
    return beta
