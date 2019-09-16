from collections import Counter
import itertools

def flatten(x):
    for e in x:
        if isinstance(e, (list, tuple)):
            yield from flatten(e)
        else:
            yield e

class Vocab:
    def __init__(self, with_unk=False):
        self._id = itertools.count()
        self._idx_to_label = {}
        self._label_to_idx = {}
        self._pad = self.add("<PAD>")
        if with_unk:
            self._unk = self.add("<UNK>")
        else:
            self._unk = None

    def add(self, label):
        if label in self._label_to_idx:
            return self._label_to_idx[label]
        idx = next(self._id)
        self._idx_to_label[idx] = label
        self._label_to_idx[label] = idx
        return idx

    def get_index(self, label):
        return self._label_to_idx.get(label, self._unk)

    def get_label(self, index):
        return self._idx_to_label[index]

    def __len__(self):
        return len(self._idx_to_label)

    def map_to_index(self, items):
        return list(self.get_index(i) for i in items)

    def map_to_label(self, indices):
        return list(self.get_label(i) for i in indices if i != self._pad)

    @classmethod
    def from_iterable(cls, x, max_size=None):
        if max_size is None:
            labels = set(flatten(x))
            vocab = cls(with_unk=False)
        else:
            labels = set(e[0] for e in Counter(flatten(x)).most_common(max_size - 2))
            vocab = cls(with_unk=True)
        for l in labels:
            vocab.add(l)
        return vocab
