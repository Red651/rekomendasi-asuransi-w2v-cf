import streamlit as st
import pickle
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from transformers import BertTokenizer, BertForSequenceClassification



class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.vector_size = word2vec.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec.wv[w] for w in word_tokenize(words) if w in self.word2vec.wv] or [np.zeros(self.vector_size)], axis=0)
            for words in X
        ])




# load model SVM + Word2Vec
with open('Model/model_svm_w2v.pkl', 'rb') as md_svm:
    model_svm = pickle.load(md_svm)

# load model BERT
model_name = 'indobenchmark/indobert-large-p1'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# menampilkan judul
st.title('Review Produk Asuransi')

# menampilkan dataset produk asuransi
data_raw = pd.read_csv('Fix_Dataset/asuransi_clean.csv')

# menampilkan dataset
st.write(data_raw)