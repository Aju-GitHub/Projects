#!/usr/bin/env python
# coding: utf-8

# This file contains a python code for classifying negative, positive as well as neutral reviews from a tweet dataset. I used Support vector machine for classification/regression (SVC) and an important vectorization technique called Tfidf. 

# In[13]:


# Importing necessary libraries for pre-processing

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[14]:


# Importing models needed

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


# In[15]:


#Loading the Data set of tweets

dt = pd.read_csv(r"D:\3rd semester\Deep Learning\COVIDSenti-A.csv")


# In[12]:


dt


# In[17]:


# Getting the data types

dt.dtypes


# In[19]:


# Getting the number of contents

dt.count()


# In[20]:


# Importing necessary libraries for natural language processing

import re
import string


# In[21]:


# Importing necessary models for language processing

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[24]:


# Defining function to make the attributes ready

def preprocess_text(tweet):
    
    # Removing punctuation
    tweet = tweet.translate(str.maketrans('','', string.punctuation))
    
    # Converting all tweets to lowercase
    tweet = tweet.lower()
    
    # Removing stop words
    stop_wds = set(stopwords.words('english'))
    tokens = tweet.split()
    tweet = [token for token in tokens if token not in stop_wds]
    tweet= ''.join(tweet)
    
    # Apply stemming or lemmatization
    stemmer = PorterStemmer()
    tweet = stemmer.stem(tweet)
    return tweet


# In[25]:


dt1 = pd.read_csv(r"D:\3rd semester\Deep Learning\COVIDSenti-A.csv")

# Preprocess the text data
dt1['tweet'] = dt1['tweet'].apply(preprocess_text)


# In[26]:


dt1.head()


# In[30]:


# Splitting data for training and testing

X_train, X_test, y_train, y_test = train_test_split(dt1['tweet'], dt1['label'], test_size=0.2)


# In[31]:


# Transforming text data into numerical vectors using a TfidVectorizer

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# In[33]:


# Train an SVM model using the SVC class

model = SVC(kernel = 'linear', C=1)
model.fit(X_train, y_train)


# In[66]:


# Evaluate the model's performance on the test set
accuracy = model.score(X_test, y_test)
print(f'Test set accuracy: {accuracy:.2f}')

# Use the model to classify new, unseen tweets
New_tweet = "very good"
prediction = model.predict


# In[67]:


ve=vectorizer.transform([New_tweet])


# In[68]:


model.predict(ve)


# In[69]:


print(preprocess_text)


# In[ ]:





# In[ ]:




