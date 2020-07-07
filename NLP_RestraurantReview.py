# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:04:43 2020

@author: SUMANTRA GHOSH
"""
#importing the Libraries
import  numpy as np
import pandas as pd

 # Importing the dataset
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)


#Cleaning the text
import re
import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
# Stemming : Keeping only the root of word eg. love,loved,loving convert into ( love ) 
#Stemming is done to much sparcity
from nltk.stem.porter import PorterStemmer # Class used for stemming
corpus=[]
ps=PorterStemmer()
for i in range(0,1000):
    review=re.sub("[^a-zA-Z]"," ",dataset["Review"][i]) # Only keep letters from a-z & A-Z
    review=review.lower()  # Put all charecters in lower case
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))] # remove stopwords
    review=" ".join(review)
    corpus.append(review)
    
    
# Create Bag Of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values


# Splittinf data into test and training set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=None)


#Fitting random forest to trainig set
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=15,random_state=0,criterion='entropy')
classifier.fit(x_train,y_train)


#Predicting the result
y_pred=classifier.predict(x_test)


#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
(20+34)/200*100