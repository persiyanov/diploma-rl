"""
an all-in-one tool to convert raw conversations into tokens into NN-readable data and back
"""
from collections import defaultdict,Counter
from nltk.tokenize import RegexpTokenizer
import regex

default_tokenizer = RegexpTokenizer(r"[A-Z|a-z|0-9|']+|[^ \n]")

class Preprocessor:
    def __init__(self,
                 tokens,
                 tokenizer=default_tokenizer):
        self.tokenizer = tokenizer
        
        special_tokens=["UNK","EOS","PAD"]
        for token in special_tokens:
            if token not in tokens:
                tokens.insert(0,token)
        
        self.tokens=tokens
        
        token_to_ix = {t:i for i,t in enumerate(tokens)}
        token_to_ix = defaultdict(lambda:token_to_ix["UNK"],token_to_ix)
        self.token_to_ix= token_to_ix
    
    @staticmethod
    def from_conversations(conversations,
                           tokenizer = default_tokenizer,
                           max_tokens=None,
                           verbose=False):
        token_freqs = Counter()
        
        if verbose:
            from tqdm import tqdm
            conversations = tqdm(conversations)
        
        for conversation in conversations:
            for (speaker,phrase) in conversation:
                token_freqs.update(tokenizer.tokenize(Preprocessor.preprocess_phrase(phrase)))
        
        tokens = [token for token,freq in token_freqs.most_common(max_tokens)]
        
        if verbose:
            coverage = sum(map(token_freqs.get,tokens))*1./sum(token_freqs.values())
            print "%i out of %i tokens, coverage=%.5f)"%(len(tokens),len(token_freqs),coverage)
        
        return Preprocessor(tokens)
    
    @staticmethod
    def preprocess_phrase(phrase):

        #remove capitalization
        phrase = phrase.lower()

        #remove phrases [in square brackets] since they're not speech
        bracket_depth = [0]
        for c in phrase:
            if c=='[':
                bracket_depth.append(bracket_depth[-1]+1)
            elif c==']':
                bracket_depth.append(bracket_depth[-1]-1)
            else:
                bracket_depth.append(bracket_depth[-1])

        phrase = [phrase[i] for i in range(len(phrase)) if (bracket_depth[i]==0 and bracket_depth[i+1]==0)]
        phrase = ''.join(phrase)

        #strip leading/ending spaces
        phrase = phrase.strip()
        
        phrase = regex.sub(r'(\p{P}|`|~)', r' \1 ', phrase)
        
        return phrase
    
        
    def phrase_to_ix(self,phrase,max_len=None):
        tokens = [self.token_to_ix[t] for t in self.tokenizer.tokenize(self.preprocess_phrase(phrase))]
        #crop
        if max_len is not None:
            tokens = tokens[:max_len-1]
        #add eos
        tokens.append(self.token_to_ix["EOS"])
        #pad
        if max_len is not None:
            tokens += [self.token_to_ix["PAD"]]*(max_len-len(tokens))
        return tokens
    
    
    def speaker_to_tags(self,speaker_str):
        return self.tokenizer.tokenize(self.preprocess_phrase(speaker_str))
     
    def ix_to_phrase(self,tokens_ix,ignore_tokens=["EOS","PAD"],end_token="EOS"):
        tokens = [self.tokens[t_ix] for t_ix in tokens_ix]
        if end_token in tokens:
            tokens = tokens[:tokens.index(end_token)]
        tokens = filter(lambda t: t not in ignore_tokens, tokens)
        return ' '.join(tokens)
    
    def preprocess_conversations(self,conversations,verbose=False,max_len = None):
        if verbose:
            from tqdm import tqdm
            conversations = tqdm(conversations)

        for conversation in conversations:
            yield [(self.speaker_to_tags(speaker),self.phrase_to_ix(phrase,max_len=max_len))
                               for (speaker,phrase) in conversation]
        

