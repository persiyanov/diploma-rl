import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from agentnet.memory import GRUCell
from agentnet.resolver import ProbabilisticResolver
from agentnet.agent import Recurrence
from collections import OrderedDict
from warnings import warn

from twoline_model import ConversationModel

class ConversationModelEmphEOS(ConversationModel):
    def build_decoder_objective(self,encoder_output,reference_answer,mask,eos_emphasis_coeff=100.):
    
        #get prev_reference_answers by shifting to the right by 1 tick (e.g. at t=5, prev is t=4)
        padding = T.repeat(T.constant(self.preprocessor.token_to_ix["EOS"],dtype='int32'),
                           reference_answer.shape[0])[:,None]
        prev_reference_answer = T.concatenate([padding,reference_answer[:,:-1]],axis=1)
        
        l_prev_reference_answer = InputLayer((None,None),prev_reference_answer)
        
        if not hasattr(self,'decoder_step'):
            raise ValueError("Please build_decoder_step first")

        dec_prev_word,dec_memory,dec_probas,dec_next_word = self.decoder_step
        
        self.decoder_training_rec = Recurrence(state_variables=dec_memory,
                                               input_sequences={dec_prev_word:l_prev_reference_answer},
                                               tracked_outputs=(dec_probas,dec_next_word),
                                               unroll_scan = False,
                                               state_init = {dec_memory.keys()[0]:encoder_output}
                                        )
        
        
        state_seqs, (probas_seq,output_tokens_seq) = self.decoder_training_rec.get_sequence_layers()
        
        next_token_probas = get_output(probas_seq)
        self.training_auto_updates = self.decoder_training_rec.get_automatic_updates()#must be called right after get_output
        
        elementwise_ce = lasagne.objectives.categorical_crossentropy(next_token_probas.reshape([-1,self.n_tokens]),
                                                                     reference_answer.ravel()).reshape(reference_answer.shape)
        
        #multiply EOS tick by the coeff, keep the rest as is
        elementwise_ce *= 1.+ (eos_emphasis_coeff-1)*T.eq(reference_answer,self.preprocessor.token_to_ix["EOS"])
        
        ce_loss = (elementwise_ce * mask).sum() / mask.sum()
        
        return ce_loss

    