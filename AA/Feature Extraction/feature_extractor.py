import pickle
import nltk
import numpy as np
import pickle
from collections import defaultdict
from collections import Counter
import re
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
def getFeature(text):
    
    fvs = np.zeros(16, np.float64)    
    # note: the nltk.word_tokenize includes punctuation
    tokens = nltk.word_tokenize(text.lower())
    words = word_tokenizer.tokenize(text.lower())
    sentences = sentence_tokenizer.tokenize(text)
    vocab = set(words)
    words_per_sentence = np.array([len(word_tokenizer.tokenize(s)) for s in sentences])
    # average number of words per sentence
    fvs[0] = words_per_sentence.mean()
    # sentence length variation
    fvs[1] = words_per_sentence.std()
    # Lexical diversity
    fvs[2] = len(vocab) / float(len(words)) 
    # Commas per sentence
    fvs[3] = tokens.count(',') / float(len(sentences))
    # Semicolons per sentence
    fvs[4] = tokens.count(';') / float(len(sentences))
    # Colons per sentence
    fvs[5] = tokens.count(':') / float(len(sentences))
    
    #fvs[:6]=fvs[:6]/np.linalg.norm(fvs[:6])
    # Word counts for some common words(Normalized)    
    fdist = nltk.FreqDist(tokens)  
    TOP_WORDS=['the','be','to','of','and','a','in','that','have','i']
    index=6
    for word in TOP_WORDS:
        fvs[index]=fdist[word]#/total_num_tokens
        index=index+1
    #fvs[6:]=fvs[6:]/np.linalg.norm(fvs[6:])
    return fvs
    
    
    


author_text_dict=pickle.load( open( "TrainDataBlogs.p", "rb" ) )
feature_vector=defaultdict(list)
for author in author_text_dict:
    for text in author_text_dict[author]:
        if text=='' or len(word_tokenizer.tokenize(text.lower()))==0:
            continue        
        text=''.join([i if ord(i) < 128 else 'NON_ASCII' for i in text])
        feature_vector[author].append(getFeature(text))

pickle.dump(feature_vector, open( "train_blog_features.p", "wb" ) )    
    
            
        