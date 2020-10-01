# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

df = pd.read_csv('dataset/train_file.csv')
df_test = pd.read_csv('dataset/test_file.csv')

df.drop(['IDLink','PublishDate','Source'],axis=1,inplace=True)

import re
import nltk
import inflect 
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from string import punctuation
import string

def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    new_string = [] 
    p = inflect.engine() 
    for word in text: 
        # if word is a digit, convert the digit 
        # to numbers and append into the new_string list 
        if word.isdigit(): 
            temp = p.number_to_words(word) 
            new_string.append(temp) 
  
        # append the word as it is 
        else: 
            new_string.append(word)
    text = new_string
    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r"\-", " ", text)
    text = re.sub(r"\=", " ", text)
    text = re.sub(r"'", " ", text)
    # text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(' +', ' ', text) 
    
    # text = re.sub(r"\s{2,}", " ", text)
    # Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    # lemmatizer=WordNetLemmatizer()
    # text=word_tokenize(text)
    # lemma = [lemmatizer.lemmatize(word) for word in text]
    # text = " ".join(lemma)
    return text

df['Headline']=df['Headline'].apply(clean_text)
df['Title']=df['Title'].apply(clean_text)

title_sent = df['Title'].tolist()
head_sent = df['Headline'].tolist()
senti_title = df['SentimentTitle'].tolist()
senti_head = df['SentimentHeadline'].tolist()

df_test['Headline']=df_test['Headline'].apply(clean_text)
df_test['Title']=df_test['Title'].apply(clean_text)

title_sent_test = df_test['Title'].tolist()
head_sent_test = df_test['Headline'].tolist()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

title_texts_train=df['Title'].astype(str)
title_texts_test=df_test['Title'].astype(str)

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=5000)

full_tfidf_word_title=word_vectorizer.fit_transform(df['Title'].values.tolist() + df_test['Title'].values.tolist())
train_tfidf_word_title = word_vectorizer.transform(df['Title'].values.tolist())
test_tfidf_word_title = word_vectorizer.transform(df_test['Title'].values.tolist())

full_tfidf_word_head=word_vectorizer.fit_transform(df['Headline'].values.tolist() + df_test['Headline'].values.tolist())
train_tfidf_word_head = word_vectorizer.transform(df['Headline'].values.tolist())
test_tfidf_word_head = word_vectorizer.transform(df_test['Headline'].values.tolist())

df=df.reset_index()
df_test=df_test.reset_index()

from sklearn.decomposition import TruncatedSVD

n_comp_title = 10
svd_obj_title = TruncatedSVD(n_components=n_comp_title, algorithm='arpack')
svd_obj_title.fit(full_tfidf_word_title)
train_svd_word_title = pd.DataFrame(svd_obj_title.transform(train_tfidf_word_title))
test_svd_word_title = pd.DataFrame(svd_obj_title.transform(test_tfidf_word_title))
    
train_svd_word_title.columns = ['svd_word_title_'+str(i) for i in range(n_comp_title)]
test_svd_word_title.columns = ['svd_word_title_'+str(i) for i in range(n_comp_title)]
df = pd.concat([df, train_svd_word_title], axis=1)
df_test = pd.concat([df_test, test_svd_word_title], axis=1)

n_comp_head = 20
svd_obj_head = TruncatedSVD(n_components=n_comp_head, algorithm='arpack')
svd_obj_head.fit(full_tfidf_word_head)
train_svd_word_head = pd.DataFrame(svd_obj_head.transform(train_tfidf_word_head))
test_svd_word_head = pd.DataFrame(svd_obj_head.transform(test_tfidf_word_head))
    
train_svd_word_head.columns = ['svd_word_head_'+str(i) for i in range(n_comp_head)]
test_svd_word_head.columns = ['svd_word_head_'+str(i) for i in range(n_comp_head)]
df = pd.concat([df, train_svd_word_head], axis=1)
df_test = pd.concat([df_test, test_svd_word_head], axis=1)

# df.head()

# train_y = pd.concat([df['SentimentTitle'], df['SentimentHeadline']], axis=1)
train_y_t = df['SentimentTitle']
train_y_h = df['SentimentHeadline']

train=df.drop(['index','Title','Headline','Facebook','GooglePlus','LinkedIn','SentimentTitle','SentimentHeadline'],axis=1)
# test=df_test.drop(['IDLink','PublishDate','Source','index','Title','Headline'],axis=1)

# train.head()

test=df_test.drop(['IDLink','PublishDate','Source','index','Title','Headline','Facebook','GooglePlus','LinkedIn'],axis=1)

# test.head()

topic_map = {'obama':1, 'economy':2,'microsoft':3,'palestine':4}

train['Topic'] = train['Topic'].map(topic_map)
test['Topic'] = test['Topic'].map(topic_map)

import lightgbm as lgb

model_t = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', num_leaves=1200,
                    learning_rate=0.05, n_estimators=1000, max_depth=12,
                    metric='rmse', bagging_fraction=0.8, feature_fraction=0.8, reg_lambda=0.9)

model_t.fit(train,train_y_t)

model_h = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', num_leaves=1200,
                    learning_rate=0.05, n_estimators=1000, max_depth=12,
                    metric='rmse', bagging_fraction=0.8, feature_fraction=0.8, reg_lambda=0.9)

model_h.fit(train,train_y_h)

pred_t=model_t.predict(test)

pred_h=model_h.predict(test)

testtemp = pd.read_csv('dataset/test_file.csv')

# testtemp.head()

submissions=pd.DataFrame(columns=['IDLink','SentimentTitle','SentimentHeadline'])
submissions['IDLink'] = testtemp['IDLink']
submissions['SentimentTitle'] = pred_t
submissions['SentimentHeadline'] = pred_h

submissions.to_csv('submission4.csv',index=False)

# sub = pd.read_csv('submission2.csv')

# sub.head()

# suniv = pd.read_csv('submission3.csv')

# suniv.head()

# meanz=pd.DataFrame(columns=['IDLink','SentimentTitle','SentimentHeadline'])
# meanz['IDLink'] = testtemp['IDLink']
# meanz['SentimentTitle'] = 0.6*sub['SentimentTitle']+0.4*suniv['SentimentTitle']
# meanz['SentimentHeadline'] = 0.6*sub['SentimentHeadline']+0.4*suniv['SentimentHeadline']

# meanz.to_csv('submission4.csv',index=False)

