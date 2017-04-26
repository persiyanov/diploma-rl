from __future__ import unicode_literals

from collections import OrderedDict
import os

import numpy as np
import theano
import theano.tensor as T
import lasagne

from lasagne.layers import InputLayer, EmbeddingLayer, LSTMLayer, DenseLayer, get_all_params, get_output
from agentnet import Recurrence
from agentnet.resolver import  ProbabilisticResolver
from agentnet.memory import LSTMCell

import data_stuff as ds


class Config:
    LSTM_LAYER_GRAD_CLIP = os.environ.get('LSTM_LAYER_GRAD_CLIP') or 5.
    TOTAL_NORM_GRAD_CLIP = os.environ.get('TOTAL_NORM_GRAD_CLIP') or 10.
    N_LSTM_UNITS = os.environ.get('N_LSTM_UNITS') or 1024
    EMB_SIZE = os.environ.get('EMB_SIZE') or 512
    BOTTLENECK_UNITS = os.environ.get('BOTTLENECK_UNITS') or 256
    TEMPERATURE = theano.shared(np.float32(1.), name='temperature')

    @classmethod
    def to_dict(cls):
        return {
            'LSTM_LAYER_GRAD_CLIP': cls.LSTM_LAYER_GRAD_CLIP,
            'TOTAL_NORM_GRAD_CLIP': cls.TOTAL_NORM_GRAD_CLIP,
            'N_LSTM_UNITS': cls.N_LSTM_UNITS,
            'EMB_SIZE': cls.EMB_SIZE,
            'BOTTLENECK_UNITS': cls.BOTTLENECK_UNITS,
            'TEMPERATURE': cls.TEMPERATURE
        }


class Enc:
    ### THEANO GRAPH INPUT ###
    input_phrase = T.imatrix("encoder phrase tokens")
    ##########################

    l_in = InputLayer((None, None), input_phrase, name='context input')
    l_mask = InputLayer((None, None), T.neq(input_phrase, ds.PAD_ix), name='context mask')

    l_emb = EmbeddingLayer(l_in, ds.N_TOKENS, Config.EMB_SIZE, name="context embedding")

    l_lstm = LSTMLayer(l_emb,
                       Config.N_LSTM_UNITS,
                       name='encoder_lstm',
                       grad_clipping=Config.LSTM_LAYER_GRAD_CLIP,
                       mask_input=l_mask,
                       only_return_final=True,
                       peepholes=False)

    output = l_lstm


class Dec:
    # Define inputs of decoder at each time step.
    prev_cell = InputLayer((None, Config.N_LSTM_UNITS), name='cell')
    prev_hid = InputLayer((None, Config.N_LSTM_UNITS), name='hid')
    input_word = InputLayer((None,))
    encoder_lstm = InputLayer((None, Config.N_LSTM_UNITS), name='encoder')

    # Embed input word and use the same embeddings as in the encoder.
    word_embedding = EmbeddingLayer(input_word, ds.N_TOKENS, Config.EMB_SIZE,
                                    W=Enc.l_emb.W, name='emb')

    # This is not WrongLSTMLayer! *Cell is used for one-tick networks.
    new_cell, new_hid = LSTMCell(prev_cell, prev_hid,
                                 input_or_inputs=[word_embedding, encoder_lstm],
                                 name='decoder_lstm',
                                 peepholes=False)

    # Define parts for new word prediction. Bottleneck is a hack for reducing time complexity.
    bottleneck = DenseLayer(new_hid, Config.BOTTLENECK_UNITS, nonlinearity=T.tanh, name='decoder intermediate')

    next_word_probs = DenseLayer(bottleneck, ds.N_TOKENS,
                                 nonlinearity=lambda probs: T.nnet.softmax(probs / Config.TEMPERATURE),
                                 name='decoder next word probas')

    next_words = ProbabilisticResolver(next_word_probs, assume_normalized=True)


class GenTest:
    n_steps = theano.shared(25)
    # This theano tensor is used as first input word for decoder.
    bos_input_var = T.zeros((Enc.input_phrase.shape[0],), 'int32') + ds.BOS_ix

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
        phrase_ix = ds.phrase2matrix([phrase], **kwargs)
        answer_ix = GenTest.generate(phrase_ix)[0][0]
        if ds.EOS_ix in answer_ix:
            answer_ix = answer_ix[:list(answer_ix).index(ds.EOS_ix)]

        GenTest.n_steps.set_value(old_value)
        return ' '.join(map(ds.tokens.__getitem__, answer_ix))


class GenTrain:
    """contains a recurrent loop where network is fed with reference answers instead of her own outputs.
    Also contains some functions that train network in that mode."""

    ### THEANO GRAPH INPUT. ###
    reference_answers = T.imatrix("decoder reference answers")  # shape [batch_size, max_len]
    ###########################

    bos_column = T.zeros((reference_answers.shape[0], 1), 'int32') + ds.BOS_ix
    reference_answers_bos = T.concatenate((bos_column, reference_answers), axis=1)  # prepend BOS

    l_ref_answers = InputLayer((None, None), reference_answers_bos, name='context input')
    l_ref_mask = InputLayer((None, None), T.neq(reference_answers_bos, ds.PAD_ix), name='context mask')

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
    predicted_probas = P_seq[:, :-1].reshape((-1, ds.N_TOKENS)) + 1e-6
    target_labels = reference_answers.ravel()

    llh_loss = lasagne.objectives.categorical_crossentropy(predicted_probas, target_labels).mean()
    llh_grads = lasagne.updates.total_norm_constraint(T.grad(llh_loss, GenTest.weights), Config.TOTAL_NORM_GRAD_CLIP)

    llh_updates = lasagne.updates.rmsprop(llh_grads, GenTest.weights, learning_rate=0.005, rho=0.9)

    train_step = theano.function([Enc.input_phrase, reference_answers], llh_loss,
                                 updates=llh_updates + recurrence.get_automatic_updates())
    get_llh = theano.function([Enc.input_phrase, reference_answers], llh_loss, no_default_updates=True)

