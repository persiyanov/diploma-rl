with open("./tokens.txt") as fin:
    tokens = list(filter(len,fin.read().split('\n')))
    
tokens.append("_BOS_") #beginning of sentence. Omitted in danet
tokens.append("_PAD_") #padding. Omitted in danet

UNK_ix, BOS_ix, EOS_ix, PAD_ix = map(tokens.index,["_UNK_","_BOS_","_EOS_","_PAD_"])
N_TOKENS = len(tokens)

from collections import defaultdict
token2idx = defaultdict(lambda:UNK_ix, {t:i for i,t in enumerate(tokens)})

import re
import numpy as np
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


from collections import deque
import pickle

def read_all_contexts(context_size=2, verbose=100000):
    with open('./open_subtitles_en_raw') as fin:
        cnt = 0
        lines = []
        for l in fin:
            lines.append(l.strip())
            cnt += 1
            if cnt >= 1000:
                break

    contexts = []
    curr_context = deque(lines[:context_size], context_size)
    curr_answer = lines[context_size]
    
    t = 0
    for line in lines[context_size+1:]:
        contexts.append({'context':list(curr_context), 'answer': curr_answer})

        if t % verbose == 0:
            print(t)
        curr_context.append(curr_answer)
        curr_answer = line.strip()

        t += 1
    return contexts


with open('contexts.pkl', 'rb') as fin:
    contexts = pickle.load(fin)



from warnings import warn
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
import lasagne


GRAD_CLIP = 5
N_LSTM_UNITS = 1024
EMB_SIZE = 512
BOTTLENECK_UNITS = 256

TEMPERATURE = theano.shared(np.float32(1.), name='temperature')


from lasagne.layers import *

print("Enc")

class Enc:
    ### THEANO GRAPH INPUT ###
    input_phrase = T.imatrix("encoder phrase tokens")
    ##########################
    
    l_in = InputLayer((None, None), input_phrase, name='context input')
    l_mask = InputLayer((None, None), T.neq(input_phrase, PAD_ix), name='context mask')
    
    l_emb = EmbeddingLayer(l_in, N_TOKENS, EMB_SIZE, name="context embedding")
    
    
    ####LSTMLayer with CORRECT outputgate####
    
    l_lstm = LSTMLayer(l_emb,
                       N_LSTM_UNITS,
                       name='encoder_lstm',
                       grad_clipping=GRAD_CLIP,
                       mask_input=l_mask,
                       only_return_final=True,
                       peepholes=False)
    
    output = l_lstm


from agentnet import Recurrence
from agentnet.resolver import  ProbabilisticResolver
from agentnet.memory import LSTMCell

print("Dec")

class Dec:
    # Define inputs of decoder at each time step.
    prev_cell = InputLayer((None, N_LSTM_UNITS), name='cell')
    prev_hid = InputLayer((None, N_LSTM_UNITS), name='hid')
    input_word = InputLayer((None,))
    encoder_lstm = InputLayer((None, N_LSTM_UNITS), name='encoder')

    
    # Embed input word and use the same embeddings as in the encoder.
    word_embedding = EmbeddingLayer(input_word, N_TOKENS, EMB_SIZE,
                                    W=Enc.l_emb.W, name='emb')
    
    
    # This is not WrongLSTMLayer! *Cell is used for one-tick networks.
    new_cell, new_hid = LSTMCell(prev_cell, prev_hid,
                                 input_or_inputs=[word_embedding, encoder_lstm],
                                 name='decoder_lstm',
                                 peepholes=False)
    
    # Define parts for new word prediction. Bottleneck is a hack for reducing time complexity.
    bottleneck = DenseLayer(new_hid, BOTTLENECK_UNITS, nonlinearity=T.tanh, name='decoder intermediate')

    
    next_word_probs = DenseLayer(bottleneck, N_TOKENS,
                                 nonlinearity=lambda probs: T.nnet.softmax(probs/TEMPERATURE),
                                 name='decoder next word probas')

    next_words = ProbabilisticResolver(next_word_probs, assume_normalized=True)

print("GenTest")

class GenTest:
    n_steps = theano.shared(25)
    # This theano tensor is used as first input word for decoder.
    bos_input_var = T.zeros((Enc.input_phrase.shape[0],), 'int32')+BOS_ix
    
    bos_input_layer = InputLayer((None,), bos_input_var, name="first input")

    recurrence = Recurrence(
        # This means that encoder.output passed to decoder.encoder_lstm input at each tick.
        input_nonsequences={Dec.encoder_lstm: Enc.output},
        
        # This defines how outputs moves to inputs at each tick in decoder. 
        # These corresponds to outputs in theano scan function.
        state_variables=OrderedDict([(Dec.new_cell, Dec.prev_cell),
                                     (Dec.new_hid, Dec.prev_hid),
                                     (Dec.next_words, Dec.input_word)]),        
        state_init={Dec.next_words: bos_input_layer},
        n_steps=n_steps,
        unroll_scan=False)
    
    weights = get_all_params(recurrence, trainable=True)
    
    recurrence_outputs = get_output(recurrence)
        
    ##### DECODER UNROLLED #####
    # Theano tensor which represents sequence of generated words.
    words_seq = recurrence_outputs[Dec.next_words]
    
    # Theano tensor which represents decoder hidden states.
    dec_cell_seq = recurrence_outputs[Dec.new_cell]
    ############################
                                                 
    generate = theano.function([Enc.input_phrase], [words_seq, dec_cell_seq],
                               updates=recurrence.get_automatic_updates())
    
    
    @staticmethod
    def reply(phrase, max_len=25, **kwargs):
        old_value = GenTest.n_steps.get_value()
        
        GenTest.n_steps.set_value(max_len)
        phrase_ix = phrase2matrix([phrase],**kwargs)
        answer_ix = GenTest.generate(phrase_ix)[0][0]
        if EOS_ix in answer_ix:
            answer_ix = answer_ix[:list(answer_ix).index(EOS_ix)]
            
        GenTest.n_steps.set_value(old_value)
        return ' '.join(map(tokens.__getitem__, answer_ix))

print("GenTrain")

class GenTrain:
    """contains a recurrent loop where network is fed with reference answers instead of her own outputs.
    Also contains some functions that train network in that mode."""
    
    ### THEANO GRAPH INPUT. ###
    reference_answers = T.imatrix("decoder reference answers") # shape [batch_size, max_len]
    ###########################
    
    bos_column = T.zeros((reference_answers.shape[0], 1), 'int32')+BOS_ix
    reference_answers_bos = T.concatenate((bos_column, reference_answers), axis=1)  #prepend BOS
    
    l_ref_answers = InputLayer((None, None), reference_answers_bos, name='context input')
    l_ref_mask = InputLayer((None, None), T.neq(reference_answers_bos, PAD_ix), name='context mask')
    
    recurrence = Recurrence(
        input_nonsequences=OrderedDict([(Dec.encoder_lstm, Enc.output)]),
        input_sequences=OrderedDict([(Dec.input_word, l_ref_answers)]),
        state_variables=OrderedDict([(Dec.new_cell, Dec.prev_cell),
                                     (Dec.new_hid, Dec.prev_hid)]),
        tracked_outputs=[Dec.next_word_probs, Dec.next_words],
        mask_input=l_ref_mask,
        unroll_scan=False)
    
    recurrence_outputs = get_output(recurrence)
    
    P_seq = recurrence_outputs[Dec.next_word_probs]
    
    
    ############################
    ###loglikelihood training###
    ############################
    predicted_probas = P_seq[:, :-1].reshape((-1, N_TOKENS))+1e-6
    target_labels = reference_answers.ravel()
    
    llh_loss = lasagne.objectives.categorical_crossentropy(predicted_probas, target_labels).mean()
    
    llh_updates = lasagne.updates.adam(llh_loss, GenTest.weights, 0.001)
    
    train_step = theano.function([Enc.input_phrase, reference_answers], llh_loss,
                                 updates=llh_updates+recurrence.get_automatic_updates())
    get_llh = theano.function([Enc.input_phrase, reference_answers], llh_loss, no_default_updates=True)



from agentnet.utils.persistence import save,load


def iterate_minibatches(batch_size=64):
    total_batches = len(contexts)//batch_size
    for b in range(total_batches):
        excerpt = contexts[b*batch_size:(b+1)*batch_size]
        excerpt_context = [item['context'] for item in excerpt]
        excerpt_answer = [item['answer'] for item in excerpt]
        yield phrase2matrix(excerpt_context), phrase2matrix(excerpt_answer)
        
from IPython.display import clear_output
import matplotlib.pyplot as plt
%matplotlib inline

test_phrases = [['Hello! How are you?'],
                ['How old are you?'],
                ['Are you fucking kidding me?'],
                ['Suck. What are you doing?'], 
                ['You are piece of shit!!!'], 
                ['holy fucking crap. you are motherfucker']]


BATCH_SIZE = 128
N_EPOCHS = 100
VERBOSITY = 10 # number of batches before printing
NUM_BATCHES = len(contexts)//BATCH_SIZE

SAVE_EVERY = 500

f_log = open('log.txt', 'w')
WEIGHTS_FILE = 'weights/LM2.pkl'

loss_history = []
print("Start training.")


for n_epoch in range(N_EPOCHS):
    for nb,batch in enumerate(iterate_minibatches(BATCH_SIZE)):
        ## Saving stuff.
        if (n_epoch*NUM_BATCHES + nb + 1) % SAVE_EVERY == 0:
            save(GenTest.recurrence, WEIGHTS_FILE)
            f_log.write("\nSAVED WEIGHTS!!!\n")
        
        ## Printing stuff.
        if (n_epoch*NUM_BATCHES + nb + 1) % VERBOSITY == 0:
            clear_output(wait=True)
            f_log.write("Processed {}/{} epochs and {}/{} batches in current epoch\n".format(n_epoch, N_EPOCHS,
                                                                                     nb+1, NUM_BATCHES))
            f_log.write("Loss (averaged with last 10 batches): {0:.5f}\n".format(np.mean(loss_history[-10:])))
            print("Loss:\t{:.4f}".format(np.mean(loss_history[-10:])))
            
            f_log.write("Answers on test phrases:\n")

            for i in range(len(test_phrases)):
                f_log.write("Phrase:\t{}\n".format(test_phrases[i]))
                f_log.write("Answer:\t{}\n".format(GenTest.reply(test_phrases[i])))
                f_log.write('---'*5+'\n')
            f_log.write('****'*10+'\n')
            
            f_log.flush()
            
            with open('loss_history.pkl', 'wb') as fout:
                pickle.dump(loss_history, fout)
        
        ## Training stuff.
        batch_loss = GenTrain.train_step(batch[0], batch[1])
    
        loss_history.append(batch_loss)