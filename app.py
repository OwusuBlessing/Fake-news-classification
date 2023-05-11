# -*- coding: utf-8 -*-
"""
Created on Mon May  8 23:58:36 2023

@author: ME
"""
import catboost
from src.prediction import Prediction 
from src.preprocessor import Preprocessing
import streamlit as st
import joblib
import pandas as pd
import altair as alt


#load saved model
model = catboost.CatBoostClassifier()
model_path = "Artifacts/cb_fakes_news_model.cbm"
model.load_model(model_path)

def predict_article(text):
    pred_,conf = Prediction(text,model).predict()
    return pred_,conf
    
#create emoji for predictions
fake_emoji = "\U0001F925"
real_emoji = "\U0001F60A"
emoji_dict = {"The news is real":real_emoji,"The news is fake":fake_emoji}
def main():
    st.title("TruthFinder: Detecting Fake News through US Article Titles")
    menu = ["Home","Tracker","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.subheader("Home - Article title In Text")
        with st.form(key="fake_news_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")
            
        if submit_text:
            col1, col2 = st.columns(2)
            
            #predict article title
            pred,proba = predict_article(raw_text)
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emoji_dict[pred]
                st.write("{} {}".format(pred,emoji_icon))
                confidence =  proba.max()
                st.success("Prediction confidence")
                confidence = f"{round(confidence* 100,2)}%"
                st.write(confidence)
                
            with col2:
               st.success("Prediction Probability")
    
               proba_df = pd.DataFrame(proba,columns=["Fake","Real"])
               #st.write(proba_df.T)
               proba_df_clean = proba_df.T.reset_index()
               proba_df_clean.columns = ["Label","Probability"]
               
               fig = alt.Chart(proba_df_clean).mark_bar().encode(x="Label",y="Probability")
               st.altair_chart(fig,use_container_width=True)
        
       
   
    
   
if __name__ == "__main__":
    main()