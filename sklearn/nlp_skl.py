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
reviews_train = []
for line in open('tweet_data2/tweet_train2.txt', 'r', encoding='UTF-8'):
	reviews_train.append(line.strip())

reviews_test = []
for line in open('tweet_data2/tweet_test2.txt', 'r', encoding='UTF-8'):
	reviews_test.append(line.strip())

# 25000개의 list. idx 0부터 idx 12499는 1, idx 12500부터 idx25000은 0
target = [1 if i < 50000 else 0 for i in range(100000)]

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


reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)


def remove_stop_words(reviews):
	removed_stop_words = []
	for review in reviews:
		removed_stop_words.append(
			' '.join([word for word in review.split() if word not in stop_words])
		)
	return removed_stop_words


reviews_train_clean = remove_stop_words(reviews_train_clean)
reviews_test_clean = remove_stop_words(reviews_test_clean)

##############################
# 벡터화
##############################
# CountVectorizer 텍스트를 토큰화 카운트하고 벡터화
# 모두 소문자로 변환시켜서 me 와 Me는 같은 특성이 된다
# n-gram
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), stop_words=stop_words)

ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
	X, target, train_size=0.75
)

##############################
# learning data 정확도 테스트
############################
# for c in [0.01, 0.05, 0.1, 0.5, 1]:
# 	svm = LinearSVC(C=c)
# 	svm.fit(X_train, y_train)
# 	print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, svm.predict(X_val))))

##############################
# 위 테스트를 토대로 test data 예측 결과
##############################
final = LinearSVC(C=0.05)
final.fit(X, target)
print("Final Accuracy: %s" % accuracy_score(target, final.predict(X_test)))

##############################
# 긍정, 부정 단어
##############################
feature_to_coef = {
	word: coef for word, coef in zip(ngram_vectorizer.get_feature_names(), final.coef_[0])
}

for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:50]:
	print(best_positive)

print("\n\n")
for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:50]:
	print(best_negative)

##############################
# test data 예측 결과
##############################
pos = []
y_predict = final.decision_function(X_test[:50000])
for i in y_predict:
	if i > 0:
		pos.append(i)

neg = []
y_predict = final.decision_function(X_test[50000:])
for i in y_predict:
	if i < 0:
		neg.append(i)

print(len(pos), '/ 50000')
print(len(neg), '/ 50000')

##############################
# 학습한 모델 저장, 불러오기
##############################
# # 변수에 저장
# saved_model = pickle.dumps(final)
# # 불러오기
# load_model = pickle.loads(saved_model)
# # 불리언 반환 true == 긍정, false == 부정
# m = load_model.decision_function(X_test[:30]) > 0
# # 1 아님 0 반환 1 == 긍정, 0 == 부정
# t = load_model.predict(X_test[:30])
# print(m)
# print(t)

# 파일에 저장
joblib.dump(ngram_vectorizer, 'tweet_vect2.pkl')
joblib.dump(final, 'tweet_model2.pkl')
#
# 불러오기
# review_model = joblib.load('review_model.pkl')
#
# print(review_model.predict(X_test[:30]))
# print("time :", time.time() - start)