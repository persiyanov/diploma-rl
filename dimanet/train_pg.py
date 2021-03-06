# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import pickle
import argparse
import os
import numpy as np
from functools import partial


def get_parser():
    parser = argparse.ArgumentParser(description='Language model baseline train script.'
                                                 'You can specify network acrhitecture by setting environment variables.'
                                                 'See mymodule.neural.seq2seq.Config.')
    parser.add_argument('--train-data', type=str, help='Path to your training data.')
    parser.add_argument('--val-data', type=str, help='Path to you validation data.')
    parser.add_argument('--vocab-path', type=str, help='Path to your vocab tokens.')
    parser.add_argument('--bsize', type=int, default=64, help='Batchsize. Default to 64.')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs. Default to 100.')
    parser.add_argument('--verbosity', type=int, default=10, help='Number of batches before printing current loss. '
                                                                  'Default to 10.')
    parser.add_argument('--save-every', type=int, default=500, help='Number of batches loop should pass before saving '
                                                                    'weights of networks. Default to 500.')
    parser.add_argument('--eval-every', type=int, default=500, help='Number of batches before evaluating on validation'
                                                                    'set. Default to 500.')
    parser.add_argument('--weights-path', type=str, help='Path to your model weights. Set this argument if '
                                                         'you want to continue training.')
    parser.add_argument('--name', type=str, help='Name of your model. Will be used in dumping additional files.')
    parser.add_argument('--dssm-model-utt', type=str, help='Which dssm model to use for sample weighting.')
    parser.add_argument('--dssm-model-user', type=str, help='Which dssm model to use for sample weighting.')
    parser.add_argument('--user-id', type=int, help='Which user to use for finetune by dssm scores.')
    parser.add_argument('--iterator-name', type=str)

    return parser


def main():
    args = get_parser().parse_args()
    assert all([args.iterator_name, args.train_data, args.val_data, args.vocab_path, args.name,
                args.dssm_model_user, args.dssm_model_utt, args.user_id is not None]),\
        "Not all required arguments were provided."

    print "Running training with config:"
    for key, value in vars(args).iteritems():
        print "--{}={}".format(key, value)

    train(args)


def train(args):
    from agentnet.utils import persistence
    from mymodule.twitter_stuff import get_iterator
    from mymodule.base_stuff import Vocab
    from mymodule.neural import seq2seq, discriminator, a2c

    print "Network architecture config:"
    seq2seq.Config.print_dict()

    train_data_path = os.path.expanduser(args.train_data)
    val_data_path = os.path.expanduser(args.val_data)

    print "Reading vocab..."
    vocab = Vocab.read_from_file(args.vocab_path)

    print "Loading dssm model..."
    assert args.dssm_model_user, "Provide both arguments for initializing dssm!"
    assert args.user_id is not None, "If you want to use dssm, provide user-id to finetuning."
    dssm_model = discriminator.DssmModel(vocab, 1000)
    dssm_model.load(args.dssm_model_utt, args.dssm_model_user)

    print "Set up BeLikeXRewards..."
    rewards_getter = a2c.BeLikeXRewards(vocab, dssm_model)
    print "Set up Seq2Seq model..."
    seq2seq_model = seq2seq.Seq2Seq(vocab)
    print "Set up SCST trainer..."
    scst_trainer = a2c.SCTrainer(rewards_getter, seq2seq_model)

    iterator = get_iterator(args.iterator_name)
    iterate_minibatches_train = partial(iterator, train_data_path, vocab)
    iterate_minibatches_val = partial(iterator, val_data_path, vocab)

    f_log = open("{}_log.txt".format(args.name), 'w')
    model_weights_filename = args.weights_path or 'weights/{}_seq2seq.pkl'.format(args.name)
    try:
        with open('{}_loss_history.pkl'.format(args.name), 'rb') as fin:
            loss_history = pickle.load(fin)
            print("Loaded old loss history")
    except:
        loss_history = []

    try:
        persistence.load(seq2seq_model.gentest.recurrence, model_weights_filename)
        print("Loaded old weights!")
    except:
        pass

    print("Start training.")
    val_loss_history = []
    avg_reward_history = []

    for n_epoch in range(args.num_epochs):
        for nb, batch in enumerate(iterate_minibatches_train(args.bsize)):
            # Saving stuff.
            if (nb + 1) % args.save_every == 0:
                persistence.save(seq2seq_model.gentest.recurrence, model_weights_filename)
                f_log.write("\nSAVED WEIGHTS to {}!!!\n".format(model_weights_filename))

            # Printing stuff.
            if (nb + 1) % args.verbosity == 0:
                f_log.write("Processed {}/{} epochs and {} batches in current epoch\n".format(n_epoch,
                                                                                              args.num_epochs,
                                                                                              nb + 1))
                f_log.write("Loss (averaged with last 10 batches): {0:.5f}\n".format(np.mean(loss_history[-10:])))
                print("Loss:\t{:.4f}\tAvg. Reward:\t{:.4f}".format(np.mean(loss_history[-10:]), np.mean(avg_reward_history[-10:])))

                f_log.write('****'*10+'\n')

                f_log.flush()

                with open('{}_loss_history.pkl'.format(args.name), 'wb') as fout:
                    pickle.dump(loss_history, fout)

            # Training stuff.
            batch_loss, batch_reward = scst_trainer.train_step(np.ones(batch[0].shape[0], dtype=np.int32)*args.user_id, batch[0], batch[1])
            loss_history.append(batch_loss)
            avg_reward_history.append(batch_reward)

            if (nb + 1) % args.eval_every == 0:
                val_loss = 0.0
                num_batches = 0
                for nb, batch in enumerate(iterate_minibatches_val(args.bsize)):
                    val_loss += seq2seq_model.gentrain.get_llh(batch[0], batch[1])
                    num_batches += 1
                val_loss /= num_batches
                if len(val_loss_history) == 0:
                    val_loss_history.append(val_loss)
                else:
                    val_loss = val_loss_history[-1]*0.1 + val_loss*0.9
                    val_loss_history.append(val_loss)
                print('******** VALIDATION TIME *************')
                print('Loss:\t{:.4f}'.format(val_loss_history[-1]))
                print('**************************************')
                print

if __name__ == '__main__':
    main()
