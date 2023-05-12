# -*- coding: utf-8 -*-
"""
Created on Mon May  8 23:58:07 2023

@author: ME
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

"""
nltk.download('wordnet')
nltk.download('stopwords')"""
lm = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

class Preprocessing:
  def __init__(self,data):
     self.data = data

  def preprocess_text(self):
        lm = WordNetLemmatizer()
        #initialise corpus to store texts p
        pred_data = [self.data]
        preprocessed_data = []

        for data in pred_data:
            review = re.sub("a-zA-Z0-9"," ",data)
            review = review.lower() #convert to lower case
            review = review.split() #Tokenize text
            review = [lm.lemmatize(x) for x in review if x not in list(stop_words)] #lemmatize and removing stopwords
            review  = " ".join(review) #join as text
            preprocessed_data.append(review)
            
        return preprocessed_data
