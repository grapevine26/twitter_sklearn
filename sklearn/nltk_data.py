from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# stop_words = stopwords.words('english')
stop_words = ['in', 'of', 'at', 'a', 'the']
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def preprocess_reviews(reviews):
	reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
	reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
	return reviews


# 파일 읽고 리스트로 만들기
reviews_train = []
for line in open('movie_data/full_train.txt', 'r', encoding='UTF-8'):
	reviews_train.append(line.strip())

reviews_test = []
for line in open('movie_data/full_test.txt', 'r', encoding='UTF-8'):
	reviews_test.append(line.strip())

# 문장 전처리 (특수문자 등 제거)
reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)


# 1. 불용어 제거
def remove_stop_words(reviews):
	removed_stop_words = []
	for review in reviews:
		removed_stop_words.append(
			' '.join([word for word in review.split() if word not in stop_words])
		)
	return removed_stop_words


reviews_train_clean = remove_stop_words(reviews_train_clean)


# 2. 어근 추출
def get_lemmatized_text(corpus):
	from nltk.stem import WordNetLemmatizer
	lemmatizer = WordNetLemmatizer()
	return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]


reviews_train_clean = get_lemmatized_text(reviews_train_clean)

# n gram 벡터화
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

# 25000개의 list. idx 0부터 idx 12499는 1, idx 12500부터 idx25000은 0
target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(
	X, target, train_size=0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
	svm = LinearSVC(C=c)
	svm.fit(X_train, y_train)
	print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, svm.predict(X_val))))

final_svm_ngram = LinearSVC(C=0.01)
final_svm_ngram.fit(X, target)
print("Final Accuracy: %s" % accuracy_score(target, final_svm_ngram.predict(X_test)))

pos = []
y_predict = final_svm_ngram.decision_function(X_test[:12499])
for i in y_predict:
	if i > 0:
		pos.append(i)

neg = []
y_predict = final_svm_ngram.decision_function(X_test[12500:])
for i in y_predict:
	if i < 0:
		neg.append(i)

print(len(pos))
print(len(neg))
# feature_to_coef = {
# 	word: coef for word, coef in zip(ngram_vectorizer.get_feature_names(), final_svm_ngram.coef_[0])
# }
#
# for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:5]:
# 	print(best_positive)
#
#
# for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:5]:
# 	print(best_negative)
