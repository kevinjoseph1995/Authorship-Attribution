#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pickle
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
author_text_dict=pickle.load( open( "TrainData.p", "rb" ) )
entire_corpus=''
for author in author_text_dict:
    authors_text=' '.join(author_text_dict[author])
    entire_corpus=entire_corpus+authors_text

NUM_TOP_WORDS = 10
all_tokens = nltk.word_tokenize(entire_corpus)
fdist = nltk.FreqDist(all_tokens)
most_common_words=fdist.most_common(NUM_TOP_WORDS)

# note: the nltk.word_tokenize includes punctuation
tokens = nltk.word_tokenize(entire_corpus.lower())
words = word_tokenizer.tokenize(entire_corpus.lower())
sentences = sentence_tokenizer.tokenize(entire_corpus)
vocab = set(words)
words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                               for s in sentences])
print 'Most common words and their frequency:'
print most_common_words
# average number of words per sentence
avg_words_p_sen = words_per_sentence.mean()
print "average number of words per sentence:"+str(avg_words_p_sen)
# sentence length variation
sen_len_var = words_per_sentence.std()
print "average  sentence length variation:"+str(sen_len_var)
# Lexical diversity
average_lexical_diversity = len(vocab) / float(len(words))
print "average  Lexical diversity:"+str(average_lexical_diversity)
# Commas per sentence
avg_comma_p_sen= tokens.count(',') / float(len(sentences))
print "Commas per sentence:"+str(avg_comma_p_sen)
# Semicolons per sentence
avg_semi_col_p_sen = tokens.count(';') / float(len(sentences))
print "Semicolons per sentence:"+str(avg_semi_col_p_sen)
# Colons per sentence
avg_col_per_sen= tokens.count(':') / float(len(sentences))
print "Colons per sentence:"+str(avg_col_per_sen)