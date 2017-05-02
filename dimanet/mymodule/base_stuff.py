# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import codecs
import re
import numpy as np

from collections import defaultdict


class Vocab:
    UNK = '_UNK_'
    BOS = '_BOS_'
    EOS = '_EOS_'
    PAD = '_PAD_'

    def __init__(self, tokens):
        assert isinstance(tokens, list)
        tokens.extend([self.UNK, self.BOS, self.EOS, self.PAD])

        self.tokens = tokens
        self.n_tokens = len(tokens)

        self.UNK_ix, self.BOS_ix, self.EOS_ix, self.PAD_ix = map(self.tokens.index,
                                                                 [self.UNK, self.BOS, self.EOS, self.PAD])
        self.token2idx = defaultdict(lambda: self.UNK_ix, {t: i for i, t in enumerate(tokens)})

    @classmethod
    def read_from_file(cls, filename):
        with codecs.open(filename, encoding='utf8') as fin:
            tokens = list(filter(len, fin.read().split('\n')))
        tokens.append("_BOS_")
        tokens.append("_PAD_")

        return cls(tokens)


def normalize_line(line):
    line = line.strip().lower()

    # pad all punctuation with spaces
    line = re.sub("([.,!?()~`'])", r' \1 ', line)
    # collapse two+ spaces into one.
    line = re.sub('\s{2,}', ' ', line)
    return line


def preprocess(lines, vocab, normalize=False, speaker=None, add_eos=True):
    if type(lines) is str:
        lines = [lines]

    context = []
    for line in lines:
        if normalize:
            line = normalize_line(line)
        else:
            line = line.strip()

        line_ix = list(map(vocab.token2idx.__getitem__, filter(len, line.split())))
        if add_eos:
            line_ix.append(vocab.EOS_ix)
        context += line_ix

    if speaker is not None:
        context.append(speaker)

    return context


def idx2matrix(phrases_ix, vocab, max_len=None):
    max_len = max_len or max(map(len, phrases_ix))

    matrix = np.zeros((len(phrases_ix), max_len), dtype='int32') + vocab.PAD_ix

    for i, phrase_ix in enumerate(phrases_ix):
        matrix[i, :min(len(phrase_ix), max_len)] = phrase_ix[:max_len]

    return matrix


def phrase2matrix(contexts, vocab, max_len=None, **kwargs):
    return idx2matrix([preprocess(phrases, vocab, **kwargs) for phrases in contexts], vocab, max_len=max_len)


