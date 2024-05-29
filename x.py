
import os
import pandas as pd
import numpy as np
import streamlit as st
import re
import pickle
def remove_tags(text):
    return re.sub(re.compile('<.*?>'),'',text)

def lwr(text):
    return text.lower()

from nltk.corpus import stopwords
sw_list=stopwords.words('english')

def stopword(text):
    return " ".join([word for word in text.split() if word not in sw_list])

import string 
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

import contractions
def remove_contractions(text):
    return contractions.fix(text)

def dec_vector(doc):
    with open("Sentimental_Analysis_WV.pkl", 'rb') as file:  
        model = pickle.load(file)
    doc=[word for word in doc.split() if word in model.wv.index_to_key]
    return np.mean(model.wv[doc],axis=0)

def xvalue(text):
    X=[]
    X.append(dec_vector(text))
    return X

def preprocessed(text):
    
    text=remove_tags(text)
    text=lwr(text)
    text=stopword(text)
    text=remove_punctuation(text)
    text=remove_contractions(text)
    X=xvalue(text)
    X=np.array(X)
    return X

def clear_text():
    st.session_state["text"] = ""


def main():
    
    
    with open("Sentimental_Analysis_Word2Vec.pkl", 'rb') as file:  
        rf = pickle.load(file)
    st.title('Sentiment Analysis')
    
    text = st.text_input(
        "Enter some text 👇", key="text")
    
    if st.button('Classify'):
        z=preprocessed(text)
        if rf.predict(z)[0]==1:
            st.success("Positive")
        else:
            st.success("Negative")
        st.button("Clear", on_click=clear_text)


if __name__=='__main__':
    main()
