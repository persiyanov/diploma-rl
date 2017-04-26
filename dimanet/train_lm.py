from __future__ import unicode_literals
from mymodule.data_stuff import *
from mymodule.seq2seq import *
from agentnet.utils import persistence

import pickle


data_path = './open_subtitles_en_raw'

#contexts = read_all_contexts(data_path)
#print("Len contexts = {}".format(len(contexts)))
#with open('contexts.pkl', 'wb') as fout:
#    pickle.dump(contexts, fout)
with open('contexts.pkl', 'rb') as fin:
    contexts = pickle.load(fin)

train_contexts = contexts[:-10000]
val_contexts = contexts[-10000:]


def get_generator(contexts, batch_size):
    raw_generator = iterate_minibatches(contexts, batch_size)
    return raw_generator

test_phrases = [['Hello! How are you?'],
                ['How old are you?'],
                ['Are you fucking kidding me?'],
                ['Suck. What are you doing?'], 
                ['You are piece of shit!!!'], 
                ['holy fucking crap. you are motherfucker']]


BATCH_SIZE = 64
N_EPOCHS = 100
VERBOSITY = 10 # number of batches before printing
NUM_BATCHES = len(contexts)//BATCH_SIZE

SAVE_EVERY = 500
EVAL_EVERY = 500

f_log = open('log.txt', 'w')
WEIGHTS_FILE = 'weights/LM3.pkl'

try:
    with open('loss_history.pkl', 'rb') as fin:
        loss_history = pickle.load(fin)
        print("Loaded old loss history")
except:
    loss_history = []
print("Start training.")

try:
    persistence.load(GenTest.recurrence, WEIGHTS_FILE)
    print("Loaded old weights!")
except:
    pass

val_loss_history = []

for n_epoch in range(N_EPOCHS):
    for nb,batch in enumerate(get_generator(train_contexts, BATCH_SIZE)):
        # Saving stuff.
        if (n_epoch*NUM_BATCHES + nb + 1) % SAVE_EVERY == 0:
            persistence.save(GenTest.recurrence, WEIGHTS_FILE)
            f_log.write("\nSAVED WEIGHTS!!!\n")
        
        # Printing stuff.
        if (n_epoch*NUM_BATCHES + nb + 1) % VERBOSITY == 0:
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
        
        # Training stuff.
        batch_loss = GenTrain.train_step(batch[0], batch[1])
    
        loss_history.append(batch_loss)

        if (n_epoch*NUM_BATCHES + nb + 1) % EVAL_EVERY == 0:
            val_loss = 0.0
            for nb, batch in enumerate(iterate_minibatches(val_contexts, BATCH_SIZE)):
                val_loss += GenTrain.get_llh(batch[0], batch[1])
            val_loss /= len(val_contexts) // BATCH_SIZE
            if len(val_loss_history) == 0:
                val_loss_history.append(val_loss)
            else:
                val_loss = val_loss_history[-1]*0.1 + val_loss*0.9
                val_loss_history.append(val_loss)
            print('******** VALIDATION TIME *************')
            print('Loss:\t{:.4f}'.format(val_loss_history[-1]))
            print('**************************************')
            print


