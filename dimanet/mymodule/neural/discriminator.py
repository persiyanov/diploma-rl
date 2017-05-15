# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os

import theano
import lasagne
import theano.tensor as T
from lasagne.layers import (
    InputLayer,
    DenseLayer,
    EmbeddingLayer,
    ExpressionLayer,
    LSTMLayer,
    get_output,
    get_all_params,
    dropout
)

from seq2seq import Config


class DssmConfig:
    SEMANTIC_SPACE_SIZE = os.environ.get('SEMANTIC_SPACE_SIZE', 512)
    USER_EMB_SIZE = os.environ.get('USER_EMB_SIZE', 128)
    DROPOUT_RATE = os.environ.get('DROPOUT_RATE', 0.3)

    @classmethod
    def to_dict(cls):
        return {
            'SEMANTIC_SPACE_SIZE': cls.SEMANTIC_SPACE_SIZE,
            'USER_EMB_SIZE': cls.USER_EMB_SIZE,
            'DROPOUT_RATE': cls.DROPOUT_RATE
        }

    @classmethod
    def print_dict(cls):
        from pprint import pprint
        pprint(cls.to_dict())


class Enc:
    def __init__(self, vocab, input_var=None):
        ### THEANO GRAPH INPUT ###
        # self.input_phrase = T.imatrix("encoder phrase tokens")
        ##########################

        self.l_in = InputLayer((None, None), input_var=input_var, name='utt input')
        self.l_mask = ExpressionLayer(self.l_in, lambda x: T.neq(x, vocab.PAD_ix), name='utt mask')

        self.l_emb = EmbeddingLayer(self.l_in, vocab.n_tokens, Config.EMB_SIZE, name="utt embedding")

        self.l_lstm = LSTMLayer(self.l_emb,
                                Config.N_LSTM_UNITS,
                                name='encoder_lstm',
                                grad_clipping=Config.LSTM_LAYER_GRAD_CLIP,
                                mask_input=self.l_mask,
                                only_return_final=True,
                                peepholes=False)

        self.output = self.l_lstm


class DssmModel:
    """
    One lstm for utterance + embedding layer for user.
    """
    def __init__(self, vocab, num_users):
        self.vocab = vocab

        self._user_id = T.ivector('user ids')
        self._good_utterance = T.imatrix('utterance from user')
        self._bad_utterance = T.imatrix('utterance not from user')

        self.l_utt_enc = Enc(vocab)

        self._user_inp = InputLayer((None,), input_var=self._user_id, name='user ids layer')
        self.l_user_emb = EmbeddingLayer(self._user_inp, num_users, DssmConfig.USER_EMB_SIZE, name='user embedding')
        self.l_user_semantic = DenseLayer(self.l_user_emb, DssmConfig.SEMANTIC_SPACE_SIZE, name='user representation')
        self.l_user_semantic = dropout(self.l_user_semantic, p=DssmConfig.DROPOUT_RATE)

        self.l_utt_semantic = DenseLayer(self.l_utt_enc.output, DssmConfig.SEMANTIC_SPACE_SIZE, name='utterance representation')
        self.l_utt_semantic = dropout(self.l_utt_semantic, p=DssmConfig.DROPOUT_RATE)

        self.user_semantic = get_output(self.l_user_semantic)
        self.user_semantic_d = get_output(self.l_user_semantic, deterministic=True)

        self.good_utt_semantic = get_output(self.l_utt_semantic, inputs={self.l_utt_enc.l_in: self._good_utterance})
        self.good_utt_semantic_d = get_output(self.l_utt_semantic, inputs={self.l_utt_enc.l_in: self._good_utterance},
                                              deterministic=True)

        self.bad_utt_semantic = get_output(self.l_utt_semantic, inputs={self.l_utt_enc.l_in: self._bad_utterance})
        self.bad_utt_semantic_d = get_output(self.l_utt_semantic, inputs={self.l_utt_enc.l_in: self._bad_utterance},
                                             deterministic=True)

    @classmethod
    def _get_norm(cls, v):
        return (v**2).sum(axis=-1)**.5

    @classmethod
    def _get_cosine(cls, v1, v2):
        return (v1*v2).sum(axis=-1) / (cls._get_norm(v1)*cls._get_norm(v2))

    def _build_loss_and_ops(self):
        self.good_similarity = self._get_cosine(self.user_semantic, self.good_utt_semantic)
        self.good_similarity_d = self._get_cosine(self.user_semantic_d, self.good_utt_semantic_d)
        self.bad_similarity = self._get_cosine(self.user_semantic, self.bad_utt_semantic)
        self.bad_similarity_d = self._get_cosine(self.user_semantic_d, self.bad_utt_semantic_d)

        self.loss = T.nnet.relu(self.bad_similarity - self.good_similarity + 1).mean()
        self.loss_d = T.nnet.relu(self.bad_similarity_d - self.good_similarity_d + 1).mean()

        self.weights = get_all_params(self.l_utt_semantic)+get_all_params(self.l_user_semantic)
        grads = lasagne.updates.total_norm_constraint(T.grad(self.loss, self.weights), Config.TOTAL_NORM_GRAD_CLIP)

        self.updates = lasagne.updates.adam(grads, self.weights)

        self.train_op = theano.function([self._user_id, self._good_utterance, self._bad_utterance], self.loss,
                                        updates=self.updates, allow_input_downcast=True)

        self.val_op = theano.function([self._user_id, self._good_utterance, self._bad_utterance], self.loss_d,
                                      allow_input_downcast=True)

        self.predict_d_op = theano.function([self._user_id, self._good_utterance], self.good_similarity_d,
                                            allow_input_downcast=True)
