from __future__ import unicode_literals

import codecs
from collections import deque

from base_stuff import phrase2matrix


def iterate_minibatches_opensub(contexts, vocab, batch_size):
    total_batches = len(contexts) // batch_size
    for b in range(total_batches):
        excerpt = contexts[b * batch_size:(b + 1) * batch_size]
        excerpt_context = [item['context'] for item in excerpt]
        excerpt_answer = [item['answer'] for item in excerpt]
        yield (phrase2matrix(excerpt_context, vocab, normalize=True),
               phrase2matrix(excerpt_answer, vocab, normalize=True))


def read_opensub_contexts(data_path, context_size=2, verbose=100000):
    with codecs.open(data_path, encoding='utf8') as fin:
        lines = []
        for l in fin:
            lines.append(l.strip())

    contexts = []
    curr_context = deque(lines[:context_size], context_size)
    curr_answer = lines[context_size]

    t = 0
    for line in lines[context_size + 1:]:
        contexts.append({'context': list(curr_context), 'answer': curr_answer})

        if t % verbose == 0:
            print(t)
        curr_context.append(curr_answer)
        curr_answer = line.strip()

        t += 1
    return contexts
