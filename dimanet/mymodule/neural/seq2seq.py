from __future__ import unicode_literals

from collections import OrderedDict
import os

import numpy as np
import theano
import theano.tensor as T
import lasagne

from lasagne.layers import InputLayer, EmbeddingLayer, LSTMLayer, DenseLayer, get_all_params, get_output
from agentnet import Recurrence
from agentnet.resolver import ProbabilisticResolver
from agentnet.memory import LSTMCell

from mymodule import base_stuff


class Config:
    LSTM_LAYER_GRAD_CLIP = os.environ.get('LSTM_LAYER_GRAD_CLIP', 5.)
    TOTAL_NORM_GRAD_CLIP = os.environ.get('TOTAL_NORM_GRAD_CLIP', 10.)
    N_LSTM_UNITS = os.environ.get('N_LSTM_UNITS', 1024)
    EMB_SIZE = os.environ.get('EMB_SIZE', 512)
    BOTTLENECK_UNITS = os.environ.get('BOTTLENECK_UNITS', 256)
    TEMPERATURE = theano.shared(np.float32(1.), name='temperature')

    @classmethod
    def to_dict(cls):
        return {
            'LSTM_LAYER_GRAD_CLIP': cls.LSTM_LAYER_GRAD_CLIP,
            'TOTAL_NORM_GRAD_CLIP': cls.TOTAL_NORM_GRAD_CLIP,
            'N_LSTM_UNITS': cls.N_LSTM_UNITS,
            'EMB_SIZE': cls.EMB_SIZE,
            'BOTTLENECK_UNITS': cls.BOTTLENECK_UNITS,
            'TEMPERATURE': cls.TEMPERATURE.get_value()
        }

    @classmethod
    def print_dict(cls):
        from pprint import pprint
        pprint(cls.to_dict())


class Enc:
    def __init__(self, vocab):
        ### THEANO GRAPH INPUT ###
        self.input_phrase = T.imatrix("encoder phrase tokens")
        ##########################

        self.l_in = InputLayer((None, None), self.input_phrase, name='context input')
        self.l_mask = InputLayer((None, None), T.neq(self.input_phrase, vocab.PAD_ix), name='context mask')

        self.l_emb = EmbeddingLayer(self.l_in, vocab.n_tokens, Config.EMB_SIZE, name="context embedding")

        self.l_lstm = LSTMLayer(self.l_emb,
                                Config.N_LSTM_UNITS,
                                name='encoder_lstm',
                                grad_clipping=Config.LSTM_LAYER_GRAD_CLIP,
                                mask_input=self.l_mask,
                                only_return_final=True,
                                peepholes=False)

        self.output = self.l_lstm


class Dec:
    def __init__(self, vocab, enc):
        # Define inputs of decoder at each time step.
        self.prev_cell = InputLayer((None, Config.N_LSTM_UNITS), name='cell')
        self.prev_hid = InputLayer((None, Config.N_LSTM_UNITS), name='hid')
        self.input_word = InputLayer((None,))
        self.encoder_lstm = InputLayer((None, Config.N_LSTM_UNITS), name='encoder')

        # Embed input word and use the same embeddings as in the encoder.
        self.word_embedding = EmbeddingLayer(self.input_word, vocab.n_tokens, Config.EMB_SIZE,
                                             W=enc.l_emb.W, name='emb')

        # This is not WrongLSTMLayer! *Cell is used for one-tick networks.
        self.new_cell, self.new_hid = LSTMCell(self.prev_cell, self.prev_hid,
                                               input_or_inputs=[self.word_embedding, self.encoder_lstm],
                                               name='decoder_lstm',
                                               peepholes=False)

        # Define parts for new word prediction. Bottleneck is a hack for reducing time complexity.
        self.bottleneck = DenseLayer(self.new_hid, Config.BOTTLENECK_UNITS, nonlinearity=T.tanh,
                                     name='decoder intermediate')

        self.next_word_probs = DenseLayer(self.bottleneck, vocab.n_tokens,
                                          nonlinearity=lambda probs: T.nnet.softmax(probs / Config.TEMPERATURE),
                                          name='decoder next word probas')

        self.next_words = ProbabilisticResolver(self.next_word_probs, assume_normalized=True)


class GenTest:
    def __init__(self, vocab, enc, dec):
        self.vocab = vocab

        self.n_steps = theano.shared(25)
        # This theano tensor is used as first input word for decoder.
        self.bos_input_var = T.zeros((enc.input_phrase.shape[0],), 'int32') + vocab.BOS_ix

        self.bos_input_layer = InputLayer((None,), self.bos_input_var, name="first input")

        self.recurrence = Recurrence(
            # This means that encoder.output passed to decoder.encoder_lstm input at each tick.
            input_nonsequences={dec.encoder_lstm: enc.output},

            # This defines how outputs moves to inputs at each tick in decoder.
            # These corresponds to outputs in theano scan function.
            state_variables=OrderedDict([(dec.new_cell, dec.prev_cell),
                                         (dec.new_hid, dec.prev_hid),
                                         (dec.next_words, dec.input_word)]),
            state_init={dec.next_words: self.bos_input_layer},
            n_steps=self.n_steps,
            unroll_scan=False)

        self.weights = get_all_params(self.recurrence, trainable=True)

        self.recurrence_outputs = get_output(self.recurrence)
        self.recurrence_updates = self.recurrence.get_automatic_updates()

        ##### DECODER UNROLLED #####
        # Theano tensor which represents sequence of generated words.
        self.words_seq = self.recurrence_outputs[dec.next_words]
        self.words_seq_greedy = get_output(self.recurrence[dec.next_words], recurrence_flags={'greedy': True})
        self.recurrence_greedy_updates = self.recurrence.get_automatic_updates()

        self.generate = theano.function([enc.input_phrase], self.words_seq,
                                        updates=self.recurrence_updates+self.recurrence_greedy_updates)

    def reply(self, phrase, max_len=25, **kwargs):
        old_value = self.n_steps.get_value()

        self.n_steps.set_value(max_len)
        phrase_ix = base_stuff.phrase2matrix([phrase], self.vocab, **kwargs)
        answer_ix = self.generate(phrase_ix)[0]
        if self.vocab.EOS_ix in answer_ix:
            answer_ix = answer_ix[:list(answer_ix).index(self.vocab.EOS_ix)]

        self.n_steps.set_value(old_value)
        return ' '.join(map(self.vocab.tokens.__getitem__, answer_ix))


class GenTrain:
    """contains a recurrent loop where network is fed with reference answers instead of her own outputs.
    Also contains some functions that train network in that mode."""

    def __init__(self, vocab, enc, dec, gentest):
        self.vocab = vocab

        ### THEANO GRAPH INPUT. ###
        self.reference_answers = T.imatrix("decoder reference answers")  # shape [batch_size, max_len]
        self.sample_weights = T.vector("samples weights for different users")  # shape [batch_size,]
        ###########################

        self.bos_column = T.zeros((self.reference_answers.shape[0], 1), 'int32') + vocab.BOS_ix
        self.reference_answers_bos = T.concatenate((self.bos_column, self.reference_answers), axis=1)  # prepend BOS

        self.l_ref_answers = InputLayer((None, None), self.reference_answers_bos, name='context input')
        self.l_ref_mask = InputLayer((None, None), T.neq(self.reference_answers_bos, vocab.PAD_ix), name='context mask')

        self.recurrence = Recurrence(
            input_nonsequences=OrderedDict([(dec.encoder_lstm, enc.output)]),
            input_sequences=OrderedDict([(dec.input_word, self.l_ref_answers)]),
            state_variables=OrderedDict([(dec.new_cell, dec.prev_cell),
                                         (dec.new_hid, dec.prev_hid)]),
            tracked_outputs=[dec.next_word_probs, dec.next_words],
            mask_input=self.l_ref_mask,
            unroll_scan=False)

        self.recurrence_outputs = get_output(self.recurrence)
        self.words_seq = self.recurrence_outputs[dec.next_words]
        self.words_seq_greedy = get_output(self.recurrence[dec.next_words], recurrence_flags={'greedy': True})
        self.recurrence_greedy_updates = self.recurrence.get_automatic_updates()

        self.P_seq = self.recurrence_outputs[dec.next_word_probs]

        ############################
        ###loglikelihood training###
        ############################
        self.predicted_probas = self.P_seq[:, :-1].reshape((-1, vocab.n_tokens)) + 1e-6
        self.target_labels = self.reference_answers.ravel()

        self._raw_llh_loss = lasagne.objectives.categorical_crossentropy(self.predicted_probas, self.target_labels)
        self.llh_loss = self._raw_llh_loss.mean()
        self.llh_loss_weighted = (self.sample_weights[:, None] * self._raw_llh_loss).mean()

        self.llh_grads = lasagne.updates.total_norm_constraint(T.grad(self.llh_loss, gentest.weights),
                                                               Config.TOTAL_NORM_GRAD_CLIP)

        self.llh_updates = lasagne.updates.rmsprop(self.llh_grads, gentest.weights, learning_rate=0.005, rho=0.9)

        self.train_step = theano.function([enc.input_phrase, self.reference_answers], self.llh_loss,
                                          updates=self.llh_updates + self.recurrence.get_automatic_updates())
        self.train_step_weighted = theano.function([enc.input_phrase, self.reference_answers, self.sample_weights],
                                                   self.llh_loss_weighted, updates=self.llh_updates + self.recurrence.get_automatic_updates())

        self.get_llh = theano.function([enc.input_phrase, self.reference_answers], self.llh_loss,
                                       no_default_updates=True)
        self.get_llh_weighted = theano.function([enc.input_phrase, self.reference_answers, self.sample_weights],
                                                self.llh_loss_weighted,
                                                no_default_updates=True)


class Seq2Seq:
    def __init__(self, vocab):
        self.vocab = vocab
        self.enc = Enc(self.vocab)
        self.dec = Dec(self.vocab, self.enc)
        self.gentest = GenTest(self.vocab, self.enc, self.dec)
        self.gentrain = GenTrain(self.vocab, self.enc, self.dec, self.gentest)
