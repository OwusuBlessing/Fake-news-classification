# -*- coding: utf-8 -*-
"""
Created on Mon May  8 23:57:50 2023

@author: ME
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessor import Preprocessing
import joblib

# Load the saved TF-IDF preprocessor using joblib
path = r"C:/Users/ME/Desktop/Blessing_AI/Fake_News/Artifacts/tfidf_preprocessor.pkl"
class Prediction:
  def __init__(self,pred_data,model):
      self.pred_data = pred_data
      self.model = model



  def predict(self):
    preprocess_data = Preprocessing(self.pred_data).preprocess_text()

    loaded_tfidf = joblib.load(path)
    data = loaded_tfidf.transform(preprocess_data)
    predicted = self.model.predict(data)
    proba = self.model.predict_proba(data)
   
    if predicted[0] == 0:
      return "The news is fake",proba
    else:
      return "The news is real",proba
