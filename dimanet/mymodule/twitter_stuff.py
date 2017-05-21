# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import codecs
import numpy as np
import copy

from collections import defaultdict

from background_generator import background
from base_stuff import phrase2matrix


def parse_id_and_msgs(raw_line):
    raw_line = raw_line.strip()
    id_, msgs = raw_line.split('\t\t')
    msgs = json.loads(msgs)
    return id_, msgs


@background
def iterate_minibatches_twitter(filename, vocab, batch_size, context_size=2, convert_to_matrix=True,
                                weight_fn=None):
    """File at 'filename' has the following format:
    2134812350123\t\t["привет ахахаха", "дарова брат", "че как сам ? ? ?"]

    weight_fn = lambda ans: dssm_model.similarity(uid, ans)
    """
    with codecs.open(filename, encoding='utf8') as fin:
        batch_context = []
        batch_answer = []
        if weight_fn:
            batch_weights = []
            
        for line in fin:
            id_, msgs = parse_id_and_msgs(line)
            if len(msgs) <= 1:
                continue
            elif len(msgs) <= context_size:
                context, answer = msgs[:-1], msgs[-1]
                batch_context.append(context)
                batch_answer.append(answer)
                if weight_fn:
                    batch_weights.append(weight_fn(answer))
            else:
                for k in xrange(context_size, len(msgs) - 1):
                    context, answer = msgs[k - context_size:k], msgs[k + 1]

                    batch_context.append(context)
                    batch_answer.append(answer)
                    if weight_fn:
                        batch_weights.append(weight_fn(answer))

            assert len(batch_context) == len(batch_answer), '******* MAAAN, YOU FUCKED UP!!!! *******'

            if len(batch_context) >= batch_size:
                next_context = batch_context[batch_size:]
                next_answer = batch_answer[batch_size:]
                if weight_fn:
                    next_weights = batch_weights[batch_size:]

                batch_context = phrase2matrix(batch_context[:batch_size], vocab, normalize=False) \
                    if convert_to_matrix else batch_context[:batch_size]
                batch_answer = phrase2matrix(batch_answer[:batch_size], vocab, normalize=False) \
                    if convert_to_matrix else batch_answer[:batch_size]
                if weight_fn:
                    yield batch_context, batch_answer, batch_weights[:batch_size]
                else:
                    yield batch_context, batch_answer
                    batch_weights = next_weights

                batch_context = next_context
                batch_answer = next_answer


def _sample_from_but(arr, exclude=()):
    res = np.random.choice(arr)
    while res in exclude:
        res = np.random.choice(arr)
    return res


def _sample_utt(arr, delete=False):
    idx = np.random.choice(np.arange(len(arr)))
    res = arr[idx]
    if delete:
        del arr[idx]
    return res


# @background
def iterate_minibatches_twitter_dssm(filename, vocab, batch_size, convert_to_matrix=True):
    """File at 'filename' has the following format:
    534\tахаххаха приветосикиииа
    """
    uid2msgs = defaultdict(list)
    with codecs.open(filename, encoding='utf8') as fin:
        for line in fin:
            chunks = line.strip().split('\t')
            id_, msg = int(chunks[0]), ' '.join(chunks[1:])
            uid2msgs[id_].append(msg)
    uid2msgs_copy = copy.deepcopy(uid2msgs)

    while len(uid2msgs) > 0:
        batch_uid = []
        batch_good_utt = []
        batch_bad_utt = []
        for i in xrange(batch_size):
            uid1 = _sample_from_but(uid2msgs.keys())
            uid2 = _sample_from_but(uid2msgs_copy.keys(), exclude=[uid1])

            good_utt = _sample_utt(uid2msgs[uid1], delete=True)
            if len(uid2msgs[uid1]) == 0:
                uid2msgs.pop(uid1)
            bad_utt = _sample_utt(uid2msgs_copy[uid2])

            batch_uid.append(uid1), batch_good_utt.append(good_utt), batch_bad_utt.append(bad_utt)

            if len(uid2msgs) == 0:
                break
        batch_good_utt = phrase2matrix(batch_good_utt, vocab, normalize=True) if convert_to_matrix else batch_good_utt
        batch_bad_utt = phrase2matrix(batch_bad_utt, vocab, normalize=True) if convert_to_matrix else batch_bad_utt
        yield np.array(batch_uid), batch_good_utt, batch_bad_utt


def iterate_minibatches_twitter_user_chains(filename, vocab, batch_size, convert_to_matrix=True):
    """File at 'filename' has the following format:
    ахаххаха приветосикиииа\\tну привяу\\tкак делы?
    """
    with codecs.open(filename, encoding='utf8') as fin:
        batch_context = []
        batch_answer = []
        for line in fin:
            msgs = line.split('\\t')
            assert len(msgs) == 3

            batch_context.append(msgs[:2])
            batch_answer.append(msgs[2])

            if len(batch_context) == batch_size:
                batch_context = phrase2matrix(batch_context, vocab,
                                              normalize=True) if convert_to_matrix else batch_context
                batch_answer = phrase2matrix(batch_answer, vocab, normalize=True) if convert_to_matrix else batch_answer
                yield batch_context, batch_answer

                batch_context = []
                batch_answer = []


def iterate_minibatches_twitter_selected_users_chains(filename, vocab, batch_size, convert_to_matrix=True):
    """File at 'filename' has the following format:
    534\t\tахаххаха приветосикиииа\\tну дарова\\tкак делыыы??
    """
    with codecs.open(filename, encoding='utf8') as fin:
        batch_context = []
        batch_answer = []
        batch_uid = []
        for line in fin:
            uid, chain = line.split('\t\t')
            msgs = chain.split('\\t')
            assert len(msgs) == 3

            batch_context.append(msgs[:2])
            batch_answer.append(msgs[2])
            batch_uid.append(int(batch_uid))

            if len(batch_context) == batch_size:
                batch_context = phrase2matrix(batch_context, vocab,
                                              normalize=True) if convert_to_matrix else batch_context
                batch_answer = phrase2matrix(batch_answer, vocab, normalize=True) if convert_to_matrix else batch_answer

                yield np.array(batch_uid), batch_context, batch_answer

                batch_context = []
                batch_answer = []
                batch_uid = []


_registry = {}


def register_iterator(iterator, name):
    _registry[name] = iterator


def get_iterator(name):
    return _registry[name]


register_iterator(iterate_minibatches_twitter, 'lm-training')
register_iterator(iterate_minibatches_twitter_dssm, 'dssm-training')
register_iterator(iterate_minibatches_twitter_user_chains, 'llh-on-user')
register_iterator(iterate_minibatches_twitter_selected_users_chains, 'llh-on-user-dssm-weighted')
