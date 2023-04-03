import numpy as np
import pandas as pd
import sklearn
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

os.chdir('./Assignment3/')
train=pd.read_csv(r"./Data/Training Data.csv")
val=pd.read_csv(r"./Data/Validation Data.csv")
test=pd.read_csv(r"./Data/Test Data.csv")
os.chdir('../')

#splitting the datframe into X and y
y_train,X_train=train["Label"],train["Text"]
y_val,X_val=val["Label"],val["Text"]
y_test,X_test=test["Label"],test["Text"]

tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(X_train)

def score(text:str, model, threshold:float) -> tuple:
    # Transform the input text using the same used during training
    emb = tfidf.transform([text])
    # Predict the propensity score for the input text for each class
    prediction = model.predict_proba(emb)[:,1] >= threshold
    propensity = model.predict_proba(emb)[0]
    return int(prediction[0]), float(propensity[1])