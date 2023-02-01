# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VQI7SruhB0EBRbxvJcU4HGqdRnlZUbRL
"""

import pandas as pd
import numpy as np
import regex as re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC
import string
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))

# from google.colab import drive
# drive.mount('/content/drive')

train = pd.read_csv("/content/drive/MyDrive/Spam_folder/train.csv")
val = pd.read_csv("/content/drive/MyDrive/Spam_folder/validation.csv")
test = pd.read_csv("/content/drive/MyDrive/Spam_folder/test.csv")

"""## Support Vector Classifier Model"""

clf = SVC(C=1)
clf.fit(train_mssg_tfidf, train.train_spam)

spam_predicted = clf.predict(val_mssg_tfidf)

print(accuracy_score(val.val_spam, spam_pred), '\n', classification_report(val.val_spam, spam_pred))

for i in range(1,10000,1000):
    clf = SVC(C=i)
    clf.fit(train_mssg_tfidf, train.train_spam)
    spam_predicted = clf.predict(val_mssg_tfidf)
    print(accuracy_score(val.val_spam, spam_pred), '\n', classification_report(val.val_spam, spam_pred))

"""For the different values of regularization parameter model performs similar."""



"""## Multinomial Naive Bayes Model (Count tokenizer)"""

vectorizer = CountVectorizer()
vectorizer.fit(train.train_mssg)

bagofwords=vectorizer.vocabulary_
train_mssg=vectorizer.transform(train.train_mssg)
val_mssg=vectorizer.transform(val.val_mssg)
test_mssg=vectorizer.transform(test.test_mssg)
print(len(bagofwords))
train_mssg.shape,val_mssg.shape

for i in np.arange(0.25, 2.25, 0.25):
    spam_detector = MultinomialNB(alpha=i).fit(train_mssg, train.train_spam)
    spam_pred = spam_detector.predict(val_mssg)
    print(accuracy_score(val.val_spam, spam_pred), '\n', classification_report(val.val_spam, spam_pred))

"""The model performs similar for all values of alpha from 0.5 to 2

## Multinomial Naive Bayes Model (tfidf tokenizer)
"""

vectorizer = CountVectorizer()
vectorizer.fit(train.train_mssg)

"""### Tf-Idf Tokenizer"""

bagofwords=vectorizer.vocabulary_
train_mssg=vectorizer.transform(train.train_mssg)
val_mssg=vectorizer.transform(val.val_mssg)
test_mssg=vectorizer.transform(test.test_mssg)
print(len(bagofwords))
train_mssg.shape,val_mssg.shape

tfidf_transformer = TfidfTransformer().fit(train_mssg)
train_mssg_tfidf=tfidf_transformer.transform(train_mssg)
val_mssg_tfidf=tfidf_transformer.transform(val_mssg)
test_mssg_tfidf=tfidf_transformer.transform(test_mssg)
train_mssg_tfidf.shape,val_mssg_tfidf.shape

"""### Model Fitting and Hyperparameter Tuning"""

for i in np.arange(0.25, 2.25, 0.25):
    spam_detector = MultinomialNB(alpha=i).fit(train_mssg_tfidf, train.train_spam)
    spam_pred = spam_detector.predict(val_mssg_tfidf)
    print(accuracy_score(val.val_spam, spam_pred), '\n', classification_report(val.val_spam, spam_pred))

"""After trial we found that α = 0.5 gives best results. 
It maximises accuracy and recall for Spam mails.

"""

spam_detector = MultinomialNB(alpha=0.5).fit(train_mssg_tfidf, train.train_spam)

spam_pred = spam_detector.predict(val_mssg_tfidf)
accuracy_score(val.val_spam, spam_pred)

print(classification_report(val.val_spam, spam_pred))

spam_pred_test = spam_detector.predict(test_mssg_tfidf)
accuracy_score(test.test_spam, spam_pred_test)

"""### Model Evaluation on Test Data"""

print(classification_report(test.test_spam, spam_pred_test))