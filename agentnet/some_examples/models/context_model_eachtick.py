import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from agentnet.memory import GRUCell
from agentnet.resolver import ProbabilisticResolver
from agentnet.agent import Recurrence
from collections import OrderedDict
from warnings import warn
import numpy as np

from twoline_model import ConversationModel

class ConversationModelET(ConversationModel):
    def __init__(self,preprocessor,context_shape,bottleneck_size=1024,emb_size=128,context_size=64,optimizer=lasagne.updates.adam):
        self.preprocessor = preprocessor       
        self.n_tokens = len(preprocessor.tokens)
        self.context_shape = context_shape
        
        self.prev_phrase = T.imatrix('user input line[batch,token_ix]')
        self.batch_size = self.prev_phrase.shape[0]
        self.prev_phrase_mask = T.neq(self.prev_phrase,preprocessor.token_to_ix["PAD"])
        
        self.prev_context = T.itensor3('user input line[batch,phrase,token_ix]')

        self.reference_answer = T.imatrix('reference answer[batch,token_ix]')
        self.reference_mask = T.neq(self.reference_answer,preprocessor.token_to_ix["PAD"])
        
        #default embedding that will be used in both network parts through self.apply_shared_embedding
        self.word_emb = EmbeddingLayer(InputLayer([None]),
                                             self.n_tokens,
                                             emb_size)
        
        self.encoder_output = self.build_encoder(self.prev_phrase,self.prev_phrase_mask,gru0_units=bottleneck_size)
        self.context_output = self.build_context_encoder(self.prev_context,self.context_shape,num_units=context_size)
        
        self.decoder_step = self.build_decoder_step(gru0_units=bottleneck_size,context_size=context_size)
        
        self.loss = self.build_decoder_objective(self.encoder_output,self.context_output,
                                                 self.reference_answer,self.reference_mask)
        
        self.params = get_all_params(self.decoder_training_rec,trainable=True)
        
        updates = optimizer(self.loss,self.params)
        updates = theano.OrderedUpdates(updates)
        
        self.train_fun = theano.function([self.prev_context,self.prev_phrase,self.reference_answer],self.loss,
                                         updates=updates+self.training_auto_updates)
        
        self.reply_tokens = self.build_applier(self.encoder_output,self.context_output)
        
        self.apply_fun = theano.function([self.prev_context,self.prev_phrase],self.reply_tokens,
                                         updates=self.generative_auto_updates)
        
    def build_context_encoder(self,prev_context,context_shape,num_units=64):
        
        ctx_input = InputLayer(context_shape,prev_context,name='context input')
        
        #new shape = [batch, time*phrases ravel]
        ctx_flat = flatten(ctx_input,outdim=2)
        
        ctx_emb = self.apply_shared_embedding(ctx_flat,name="flat context embedding")
        
        ctx_emb = dimshuffle(ctx_emb,[0,2,1])
        
        cnn0 = Conv1DLayer(ctx_emb,num_filters = num_units, filter_size=3,name='ctx conv0')
        
        cnn0_max = GlobalPoolLayer(cnn0,pool_function=T.max)
        
        return cnn0_max
    
    def build_decoder_step(self,gru0_units=512,grad_clip=5,context_size=64):
        
        context_vec = InputLayer((None,context_size),name="context vector inp")
        prev_phrase_vec = InputLayer((None,gru0_units),name="input phrase vector inp")
        
        prev_output_word = InputLayer((None,),name='decoder prev output inp')
        prev_output_word.output_dtype='int32'
        
        prev_output_emb = self.apply_shared_embedding(prev_output_word,
                                                      name='decoder prev output emb')
        
        l_prev_gru0 = InputLayer([None,gru0_units],name='decoder prev gru0')
        l_gru0 = GRUCell(l_prev_gru0,[prev_output_emb,context_vec,prev_phrase_vec],
                         name='decoder gru0',grad_clipping=grad_clip)
        
        self.greed = theano.shared(np.float32(1.),name='decoder greed')
        
        next_token_probas = DenseLayer(l_gru0,self.n_tokens,
                                       nonlinearity=lambda x: lasagne.nonlinearities.softmax(x*self.greed),
                                      name='decoder next letter probas')
        
        
        
        next_token = ProbabilisticResolver(next_token_probas,assume_normalized=True,
                                           name='decoder next letter picker')
        
        
        return [prev_output_word,prev_phrase_vec,context_vec],{l_gru0:l_prev_gru0},next_token_probas,next_token

    def build_decoder_objective(self,encoder_output,context_output,reference_answer,mask):
    
        #get prev_reference_answers by shifting to the right by 1 tick (e.g. at t=5, prev is t=4)
        padding = T.repeat(T.constant(self.preprocessor.token_to_ix["EOS"],dtype='int32'),
                           reference_answer.shape[0])[:,None]
        prev_reference_answer = T.concatenate([padding,reference_answer[:,:-1]],axis=1)
        
        l_prev_reference_answer = InputLayer((None,None),prev_reference_answer)
        
        if not hasattr(self,'decoder_step'):
            raise ValueError("Please build_decoder_step first")

        (dec_prev_word,dec_prev_phrase,dec_context),dec_memory,dec_probas,dec_next_word = self.decoder_step
        
        self.decoder_training_rec = Recurrence(state_variables=dec_memory,
                                               input_sequences={dec_prev_word:l_prev_reference_answer},
                                               input_nonsequences={dec_context:context_output,
                                                                   dec_prev_phrase:encoder_output},
                                               tracked_outputs=(dec_probas,dec_next_word),
                                               unroll_scan = False,
                                        )
        
        
        state_seqs, (probas_seq,output_tokens_seq) = self.decoder_training_rec.get_sequence_layers()
        
        next_token_probas = get_output(probas_seq)
        self.training_auto_updates = self.decoder_training_rec.get_automatic_updates()#must be called right after get_output
        
        self.offset = theano.shared(np.float32(0.0001))
        next_token_probas = self.offset+next_token_probas*(1-2*self.offset)
        
        elementwise_ce = lasagne.objectives.categorical_crossentropy(next_token_probas.reshape([-1,self.n_tokens]),
                                                                     reference_answer.ravel()).reshape(reference_answer.shape)
        
        ce_loss = (elementwise_ce * mask).sum() / mask.sum()
        
        return ce_loss

    def build_applier(self,encoder_output,context_output,max_steps=100):
        
        if not hasattr(self,'decoder_step'):
            raise ValueError("Please build_decoder_step first")
        
        (dec_prev_word,dec_prev_prhase,dec_context),dec_memory,dec_probas,dec_next_word = self.decoder_step
        
        recurrent_states = OrderedDict(dec_memory)
        recurrent_states[dec_next_word] = dec_prev_word
        
        self.decoder_generative_rec = Recurrence(state_variables=recurrent_states,
                                                 input_nonsequences={dec_context:context_output,
                                                                     dec_prev_prhase:encoder_output
                                                                     },
                                                 tracked_outputs=(dec_probas,dec_next_word),
                                                 batch_size=self.batch_size,
                                                 n_steps = max_steps,
                                                 unroll_scan=False,)
        
        
        state_seqs, (probas_seq,output_tokens_seq) = self.decoder_generative_rec.get_sequence_layers()
        
        out = get_output(output_tokens_seq)
        self.generative_auto_updates = self.decoder_generative_rec.get_automatic_updates()#must be called right after get_output
        return out

        
    def reply(self,input_phrase,context_lines=()):
                
        input_ix = self.preprocessor.phrase_to_ix(input_phrase)
        
        context_lines = list(context_lines)
        
        #crop/pad context_lines
        _,context_window,ctx_line_length = self.context_shape
        context_lines = context_lines[-context_window:]
        while len(context_lines) < context_window:
            context_lines.append("")
            
        context_ix = np.array(map(lambda ln: self.preprocessor.phrase_to_ix(ln,max_len=ctx_line_length),
                                     context_lines),dtype='int32')

        
        
        reply_ix = self.apply_fun([context_ix],[input_ix])[0]
        
        new_context = tuple(context_lines[-context_window:]) + (self.preprocessor.ix_to_phrase(reply_ix),)
        
        return self.preprocessor.ix_to_phrase(reply_ix),new_context