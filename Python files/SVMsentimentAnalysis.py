# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 23:35:07 2022

@author: Jenny
"""

from flask import Flask, render_template, url_for
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

tweets=pd.read_csv('tweets_cleaned.csv')

# Split into train and test data

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(tweets['text'], tweets['target'], test_size = 0.1, random_state = 0)
# random_state = 0 menyatakan tidak ada pengacakan pada data yang di split yang artinya urutannya masih sama
df_train90 = pd.DataFrame()
df_train90['text'] = train_X
df_train90['target'] = train_Y

df_test10 = pd.DataFrame()
df_test10['text'] = test_X
df_test10['target'] = test_Y
df_train90

df_test10

df_train90.to_csv(r"df_train90.csv")
df_test10.to_csv(r"df_test10.csv")
# TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect_9010 = TfidfVectorizer(max_features = 5000)
tfidf_vect_9010.fit(tweets['text'])
train_X_tfidf_9010 = tfidf_vect_9010.transform(df_train90['text'])
test_X_tfidf_9010 = tfidf_vect_9010.transform(df_test10['text'])
tfidf_vect_9010


print(train_X_tfidf_9010)

print(test_X_tfidf_9010)


print(train_X_tfidf_9010.shape)
print(test_X_tfidf_9010.shape)

# You can use the below syntax to see the vocabulary that it has learned from the corpus
print(tfidf_vect_9010.vocabulary_)



from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(train_X_tfidf_9010,train_Y)


# Proses Pengujian

from sklearn.metrics import accuracy_score

predictions_SVM_9010 = model.predict(test_X_tfidf_9010)
test_prediction_9010 = pd.DataFrame()
test_prediction_9010['text'] = test_X
test_prediction_9010['target'] = predictions_SVM_9010
SVM_accuracy_9010 = accuracy_score(predictions_SVM_9010, test_Y)*100
SVM_accuracy_9010 = round(SVM_accuracy_9010,1)

test_prediction_9010

test_prediction_9010.to_csv(r"test_prediction_9010.csv")
SVM_accuracy_9010


# Accuracy, Precision, Recall, f1-score

from sklearn.metrics import classification_report

print ("\nHere is the classification report:") 
print (classification_report(test_Y, predictions_SVM_9010))

#Bar plot for bipolar reviews

labels = ['1','0']
Category1 = [58, 357]
plt.bar(labels, Category1, tick_label=labels, width=0.5, color=['coral', 'c'])
plt.xlabel('Sentiment')
plt.ylabel('Data')
plt.title('Sentiment Analysis Diagram')
##plt.savefig(r"bar_data.png")
plt.show()


#pie chart for tweet sentiment

color = ['coral', 'c']
plt.pie(Category1, labels=labels, colors=color,startangle=90, shadow=True, autopct='%1.2f%%', explode=(0.1, 0))
plt.title('pie chart for tweet sentiment')
plt.legend()
##plt.savefig(r"pie_data.png")
plt.show()


#Bar plot for Train Set

labels = ['1','0']
Category2 = [52, 321]
plt.bar(labels, Category2, tick_label=labels, width=0.5, color=['coral', 'c'])
plt.xlabel('Sentiment')
plt.ylabel('Data')
plt.title('Bar plot for Train Set')
#plt.savefig(r"bar_datalatih.png")
plt.show()



#Bar plot for Test Set

labels = ['1','0']
Category3 = [6, 36]
plt.bar(labels, Category3, tick_label=labels, width=0.5, color=['coral', 'c'])
plt.xlabel('Sentiment')
plt.ylabel('Data')
plt.title('Bar plot for Test Set')
#plt.savefig(r"bar_datauji.png")
plt.show()
#pie chart for Test Set

color = ['coral', 'c']
plt.pie(Category3, labels=labels, colors=color, startangle=90, shadow=True, autopct='%1.2f%%', explode=(0.1, 0))
plt.title('pie chart for Test Set')
plt.legend()
##plt.savefig(r"pie_datauji.png")
plt.show()


#Bar plot for Klasifikasi dengan SVM

labels = ['1','0']
Category4 = [2, 40]
plt.bar(labels, Category4, tick_label=labels, width=0.5, color=['coral', 'c'])
plt.xlabel('Kelas Sentimen')
plt.ylabel('Data')
plt.title('Bar plot for Support Vector Machine')
##plt.savefig(r"bar_svm.png")
plt.show()


#pie chart for Klasifikasi dengan SVM

color = ['coral', 'c']
plt.pie(Category4, labels=labels, colors=color,startangle=90, shadow=True, autopct='%1.2f%%', explode=(0.1, 0))
plt.title('pie chart for Support Vector Machine')
plt.legend()
##plt.savefig(r"pie_svm.png")
plt.show()