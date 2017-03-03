import re
import numpy as np
from collections import defaultdict

with open("./tokens.txt") as fin:
    tokens = list(filter(len,fin.read().split('\n')))
    
tokens.append("_BOS_") #beginning of sentence. Omitted in danet
tokens.append("_PAD_") #padding. Omitted in danet

UNK_ix, BOS_ix, EOS_ix, PAD_ix = map(tokens.index,["_UNK_","_BOS_","_EOS_","_PAD_"])
N_TOKENS = len(tokens)

token2idx = defaultdict(lambda:UNK_ix, {t:i for i,t in enumerate(tokens)})

def preprocess(lines, speaker=None, add_eos=True):
    if type(lines) is str:
        lines = [lines]
    
    context = []
    for line in lines:
        line = line.lower()
        
        # pad all punctuation with spaces
        line = re.sub("([.,!?()~`'])", r' \1 ', line)
        # collapse two+ spaces into one.
        line = re.sub('\s{2,}', ' ', line)
        
        line_ix = list(map(token2idx.__getitem__, filter(len, line.split())))
        if add_eos:
            line_ix.append(EOS_ix)
        context += line_ix
            
    if speaker is not None:
        context.append(speaker)
        
    return context

def idx2matrix(phrases_ix, max_len=None):
    max_len = max_len or max(map(len, phrases_ix))
    
    matrix = np.zeros((len(phrases_ix), max_len), dtype='int32') + PAD_ix
    
    for i, phrase_ix in enumerate(phrases_ix):
        matrix[i, :min(len(phrase_ix), max_len)] = phrase_ix[:max_len]
        
    return matrix

def phrase2matrix(contexts, max_len=None, **kwargs):
    return idx2matrix([preprocess(phrases, **kwargs) for phrases in contexts], max_len=max_len)


def iterate_minibatches(contexts, batch_size=64):
    total_batches = len(contexts)//batch_size
    for b in range(total_batches):
        excerpt = contexts[b*batch_size:(b+1)*batch_size]
        excerpt_context = [item['context'] for item in excerpt]
        excerpt_answer = [item['answer'] for item in excerpt]
        yield phrase2matrix(excerpt_context), phrase2matrix(excerpt_answer)

def threaded_generator(generator, num_cached=50):
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()