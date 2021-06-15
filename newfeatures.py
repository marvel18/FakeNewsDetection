from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import numpy as np
import pandas
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
class WordCountExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def word_count(self,data):
        data =data.str.split().str.len()
        data = pandas.DataFrame(data)
        return data

    def transform(self, df, y=None):
        data = self.word_count(df)
        return data
    
    def fit(self, df, y=None):
        return self
class CapitalWordCountExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def title_capital_word_count(self, data):
        cap_count = []
        for row in data:
            count = 0
            for letter in row.split(" "):
                if(letter.isupper()):
                    count+=1
            cap_count.append(count)        
        return  pandas.DataFrame(cap_count)
    def transform(self, df, y=None):
       return self.title_capital_word_count(df)

    def fit(self, df, y=None):
        return self

class WordLematization(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.word_lemmatizer = WordNetLemmatizer()
    def tokenize(self,data):
        data = [entry.lower() for entry in data]
        data= [word_tokenize(entry) for entry in data]
        return data    
    def lemmatizer(self,data):
        stop_words = stopwords.words('english')
        train = []
        for row in data:
            filter_sentence = ''
            sentence = row
            sentence = re.sub(r'[^\w\s]', '', sentence)
            words = nltk.word_tokenize(sentence)
            words = [w for w in words if not w in stop_words]
            for word in words:
                filter_sentence = filter_sentence + ' ' + str(self.word_lemmatizer.lemmatize(word))
            train.append(str(filter_sentence))  
        return train
    def lemmatizerNew(self,data_final):
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        train = []
        for entry in data_final:
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(entry):
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words.append(word_Final)
            train.append(str(Final_words))
        return train            
    def transform(self, df, y=None):
        #df = self.tokenize(df)
        #return  self.lemmatizerNew(df)
        return self.lemmatizer(df)

    def fit(self, df, y=None):
        return self    