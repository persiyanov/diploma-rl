from operator import add
import numpy as np
def get_phrase_pairs(conversations,
                     speaker_filter = lambda speaker1,speaker2: True):
    
    #get all pairs
    pairs = reduce(add, map(lambda conv: zip(conv[:-1],conv[1:]),conversations))
    
    #filter phrases of target speaker
    pairs = filter(lambda ((sp1,ph1),(sp2,ph2)): speaker_filter(sp1,sp2),pairs)
    
    #remove speakers from pairs
    pairs = map(lambda ((sp1,ph1),(sp2,ph2)): (ph1,ph2),pairs)
    
    pairs = np.array(pairs)

    return pairs
import sys
def get_samples_with_context(conversations,
                             context_window_size=3,
                             padder = -1,
                             speaker_filter = lambda speaker1,speaker2: True):
    
    #get all tuples
    def get_tuples(conv):
        
        prev_phrase = conv[:-1]
        ref_answer = conv[1:]
        
        conv_lines = np.array([line for speaker,line in conv])
        ctx_selector = np.arange(len(conv))[:-1,None] + np.arange(context_window_size)[None,:] - context_window_size
        ctx_selector = np.maximum(0,ctx_selector)
        context = conv_lines[ctx_selector]
        for i in range(min(context_window_size-1,len(context))):
            context[i,:context_window_size-i] = np.zeros_like(context[i,:context_window_size-i],dtype='int32')+padder
            
                
        return zip(context,prev_phrase,ref_answer)
    
    pair_chunks = filter(len,map(get_tuples,conversations))
    
    pair_chunks = map(lambda ch: np.array(ch,dtype=object),pair_chunks)
        
    pairs = np.concatenate(pair_chunks,axis=0)
    
    
    #filter phrases of target speaker
    pairs = filter(lambda (ctx,(sp1,ph1),(sp2,ph2)): speaker_filter(sp1,sp2),pairs)
    
    #remove speakers from pairs
    pairs = map(lambda (ctx,(sp1,ph1),(sp2,ph2)): (ctx,ph1,ph2),pairs)
    pairs = np.array(pairs)

    return pairs



def iterate_minibatches(arrays, batchsize, shuffle=False):
    assert len(set(map(len,arrays)))==1
    
    indices = np.arange(len(arrays[0]))
    if shuffle:np.random.shuffle(indices)
    
    for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield [array[excerpt] for array in arrays]


        
