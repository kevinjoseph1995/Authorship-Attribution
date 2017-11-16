import pickle
import nltk
import numpy as np
from collections import defaultdict
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
from collections import defaultdict
class FeatureExtractor:
    def __init__(self,path):
        self.author_text_dict=pickle.load( open( path+".p", "rb" ) )
    def extract_lexical_features(self):
        lexical_features=defaultdict(lambda:np.zeros([1,3]))
        for author in self.author_text_dict:
            list_of_texts=self.author_text_dict[author]
            for text in list_of_texts:
                words = word_tokenizer.tokenize(text.lower())
                sentences = sentence_tokenizer.tokenize(text)
                vocab = set(words)
                words_per_sentence = np.array([len(word_tokenizer.tokenize(s))for s in sentences])
                # average number of words per sentence
                avg_words_per_sentence=words_per_sentence.mean()
                lexical_features[author][0,0]=0.5*lexical_features[author][0,0]+0.5*avg_words_per_sentence
                # sentence length variation
                avg_sentence_length_variation=words_per_sentence.std()
                lexical_features[author][0,1]=0.5*lexical_features[author][0,1]+0.5*avg_sentence_length_variation
                # Lexical diversity:a measure of the richness of the author’s vocabulary
                lexical_diversity=len(vocab) / float(len(words))
                lexical_features[author][0,2]=0.5*lexical_features[author][0,2]+0.5*lexical_diversity
        return lexical_features
    def punctuation_feat_extractor(self):
        punct_features=defaultdict(lambda:np.zeros([1,3]))
        for author in self.author_text_dict:
            list_of_texts=self.author_text_dict[author]
            for text in list_of_texts:
                tokens = nltk.word_tokenize(text.lower())
                sentences = sentence_tokenizer.tokenize(text)
                # Commas per sentence
                punct_features[author][0,0]=0.5*punct_features[author][0,0]+0.5*tokens.count(',') / float(len(sentences))
                # Semicolons per sentence
                punct_features[author][0,1]=0.5*punct_features[author][0,1]+0.5*tokens.count(';') / float(len(sentences))
                # Colons per sentence
                punct_features[author][0,2]=0.5*punct_features[author][0,2]+0.5*tokens.count(':') / float(len(sentences))
        return punct_features
    def bag_of_words_feat_extractor(self):
        entire_corpus=''
        for key in self.author_text_dict:
            entire_corpus=entire_corpus+' '.join(self.author_text_dict[key])
        NUM_TOP_WORDS = 500
        all_tokens = nltk.word_tokenize(entire_corpus)
        fdist = nltk.FreqDist(all_tokens)
        vocab = fdist.keys()[:NUM_TOP_WORDS]
        
        print vocab
            
            
        
obj=FeatureExtractor('SmallTrain')
#lexical_features=obj.extract_lexical_features()
#punct_features=obj.punctuation_feat_extractor()
obj.bag_of_words_feat_extractor()
            
        