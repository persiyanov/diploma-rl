from __future__ import unicode_literals

import numpy as np
import theano
import theano.tensor as T
from theano.gradient import disconnected_grad
import codecs


class AnswerRewards(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def get(self, answer):
        """
        answer is a theano tensor of shape int32[batch_size, n_steps] with token ids.
        """
        raise NotImplementedError

    @property
    def input_vars(self):
        raise NotImplementedError


class BePoliteRewards(AnswerRewards):
    def __init__(self, vocab, target_words_filepath):
        super(BePoliteRewards, self).__init__(vocab)

        with codecs.open(target_words_filepath, encoding='utf8') as fin:
            target_words = set()
            for line in fin:
                words_in_line = line.strip().split()
                if len(words_in_line) == 1:
                    target_words.update(words_in_line)

        self.target_words = target_words

        target_idxs = set(filter(lambda x: x != vocab.UNK_ix, [vocab.token2idx[w] for w in target_words]))
        self.target_idxs_shared = theano.shared(np.array(list(target_idxs)))

        target_idxs_mask = np.ones((vocab.n_tokens,))
        target_idxs_mask[np.array(list(target_idxs))] = 0
        target_idxs_mask = theano.shared(target_idxs_mask)
        self._target_idxs_mask = target_idxs_mask

    def _calc_rewards(self, symbolic_batch):
        assert symbolic_batch.ndim == 2
        rewards = T.eq(self.target_idxs_shared[None, None, :], symbolic_batch[:, :, None]).any(-1)
        rewards = T.cast(rewards, 'int32')
        assert rewards.ndim == 2

        # Find EOS_ix in batch
        done_mask = T.eq(symbolic_batch, self.vocab.EOS_ix)
        # Set done==True for all words after EOS_ix
        done_mask = T.concatenate([T.zeros_like(done_mask[:, :1]), done_mask[:, :-1]], axis=1)

        is_alive = T.eq(T.cumsum(done_mask, axis=1), 0).astype('uint8')
        return -rewards, is_alive

    def get(self, answer):
        return self._calc_rewards(answer)

    @property
    def input_vars(self):
        return []


class BeLikeXRewards(AnswerRewards):
    def __init__(self, vocab, dssm_model):
        super(BeLikeXRewards, self).__init__(vocab)

        self.dssm_model = dssm_model

    def get(self, answer):
        return self.dssm_model.get_similarity_tensor_for(answer)

    @property
    def input_vars(self):
        return [self.dssm_model._user_id]


class SCTrainer(object):
    """
    Self-critical trainer [https://arxiv.org/abs/1612.00563]
    """
    def __init__(self, rewards_getter, seq2seq):
        """
        Args:
            rewards_getter (BeLikeXRewards):
            seq2seq (seq2seq.Seq2Seq):
        """
        self.rewards_getter = rewards_getter
        self.s2s = seq2seq

        self.rewards = rewards_getter.get(self.s2s.gentest.words_seq)
        self.baseline = rewards_getter.get(self.s2s.gentest.words_seq_greedy)

        self.advantage = disconnected_grad(self.rewards-self.baseline)
