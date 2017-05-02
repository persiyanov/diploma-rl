# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import codecs

from background_generator import background
from base_stuff import phrase2matrix


def parse_id_and_msgs(raw_line):
    raw_line = raw_line.strip()
    id_, msgs = raw_line.split('\t\t')
    msgs = json.loads(msgs)
    return id_, msgs


@background
def iterate_minibatches_twitter(filename, vocab, batch_size, context_size=2):
    """File at 'filename' has the following format:
    2134812350123\t\t["привет ахахаха", "дарова брат", "че как сам ? ? ?"]
    """
    with codecs.open(filename, encoding='utf8') as fin:
        batch_id = 0
        batch_context = []
        batch_answer = []
        for line in fin:
            id_, msgs = parse_id_and_msgs(line)
            if len(msgs) <= 1:
                continue
            elif len(msgs) <= context_size:
                context, answer = msgs[:-1], msgs[-1]

                batch_context.append(context)
                batch_answer.append(answer)
            else:
                for k in xrange(context_size, len(msgs)-1):
                    context, answer = msgs[k-context_size:k], msgs[k+1]

                    batch_context.append(context)
                    batch_answer.append(answer)

            assert len(batch_context) == len(batch_answer), 'MAAAN, YOU FUCKED UP!!!!'

            if len(batch_context) >= batch_size:
                next_context = batch_context[batch_size:]
                next_answer = batch_answer[batch_size:]

                yield (phrase2matrix(batch_context[:batch_size], vocab, normalize=False),
                       phrase2matrix(batch_answer[:batch_size], vocab, normalize=False))

                batch_context = next_context
                batch_answer = next_answer

                batch_id += 1
