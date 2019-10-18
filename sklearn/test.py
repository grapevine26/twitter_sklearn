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
from sklearn.preprocessing import MinMaxScaler
import time

start = time.time()
stop_words = ['in', 'of', 'at', 'a', 'the']
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

reviews_test = []

for line in open('review_text.txt', 'r', encoding='UTF-8'):
	reviews_test.append(line.strip())

##############################
# 문장 전처리 (특수문자, 불용어)
##############################


def preprocess_reviews(reviews):
	reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
	reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
	return reviews


reviews_test_clean = preprocess_reviews(reviews_test)


def remove_stop_words(reviews):
	removed_stop_words = []
	for review in reviews:
		removed_stop_words.append(
			' '.join([word for word in review.split() if word not in stop_words])
		)
	return removed_stop_words


reviews_test_clean = remove_stop_words(reviews_test_clean)

##############################
# 학습한 모델 불러오기
##############################
review_model = joblib.load('review_model.pkl')
review_vect = joblib.load('review_vect.pkl')
X_test = review_vect.transform(reviews_test_clean)
print(review_model.decision_function(X_test) > 0)
print(review_model.predict(X_test))

print("time :", time.time() - start)
