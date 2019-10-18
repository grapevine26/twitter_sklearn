import numpy as np
import pandas as pd
import os
import re
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import time
start = time.time()

##########################
# 파일 읽기 (리스트 형태)
##########################
reviews_test = []
for line in open('test.txt', 'r', encoding='UTF-8'):
	reviews_test.append(line.strip())

##############################
# 문장 전처리 (특수문자, 불용어)
##############################
stop_words = ['in', 'of', 'at', 'a', 'the']
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def preprocess_reviews(reviews):
	reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
	reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
	return reviews


reviews_test_clean = preprocess_reviews(reviews_test)

tweet_model = joblib.load('tweet_model2.pkl')
tweet_vect = joblib.load('tweet_vect2.pkl')

##############################
# 벡터화
##############################
X_test = tweet_vect.transform(reviews_test_clean)

##############################
# test data 예측 결과
##############################
pos = []
neg = []
y_predict = tweet_model.decision_function(X_test)
for i in y_predict:
	if i > 0:
		pos.append(i)
	elif i < 0:
		neg.append(i)
print(y_predict)
print(len(pos), '/ pos')
print(len(neg), '/ neg')
