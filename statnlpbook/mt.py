from statnlpbook import util


def create_translation_table(source_vocab, target_vocab, fixed=None):
    if fixed is None:
        fixed = {}
    alpha = {}
    for s in source_vocab:
        for t in target_vocab:
            if t in fixed:
                alpha[s, t] = fixed[t].get(s, 0.0)
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
                    if si in fixed:
                        beta[ti, si, lt, ls] = 1.0 if fixed[si] == ti else 0.0
                    else:
                        beta[ti, si, lt, ls] = 1.0 / lt
    return beta


def render_history(history):
    rows = []
    for beam in history:
        for j in range(len(beam), 0, -1):
            hyp = beam[j - 1]
            remaining_str = [("_" if i not in hyp.remaining else hyp.source[i])
                             for i in range(0, len(hyp.source))]
            rows.append((" ".join(hyp.target), " ".join(remaining_str), len(hyp.remaining), hyp.score))
    return util.Table(rows, column_names=["Target", "Remaining", "Len", "Score"])

    # class Test:
    #     def _repr_html_(self):
    #         rows = []
    #         for beam in history:
    #             for j in range(len(beam), 0, -1):
    #                 hyp = beam[j - 1]
    #                 remaining_str = [("_" if i not in hyp.remaining else hyp.source[i])
    #                                  for i in range(0, len(hyp.source))]
    #                 rows.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
    #                     " ".join(hyp.target), " ".join(remaining_str), len(hyp.remaining), hyp.score))
    #         return "<table>" + "\n".join(rows) + "</table>"
    #
    # return Test()
