import re
import os
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def delete_symbols(docs):
	docs = [REPLACE_NO_SPACE.sub("", line.lower()) for line in docs]
	docs = [REPLACE_WITH_SPACE.sub(" ", line) for line in docs]
	return docs


def split_line(line):
	cols = line.split("\t")
	return cols


def get_words(cols):
	words_ids = cols[4].split(" ")
	words = [w.split("#")[0].replace('_', '') for w in words_ids]
	return words


def get_positive(cols):
	return cols[2]


def get_negative(cols):
	return cols[3]


def get_objective(cols):
	return 1 - (float(cols[2]) + float(cols[3]))


def get_gloss(cols):
	return cols[5]


def get_scores(filepath, sentiword):
	f = open(filepath)
	totalobject = 0.0
	count = 0.0
	totalpositive = 0.0
	totalnegative = 0.0
	import time
	start = time.time()
	# SentiWordNet 파일
	for line in f:
		#
		if not line.startswith("#"):
			# swn 라인 한 줄에서 탭으로 split
			cols = split_line(line)
			# swn '단어#1 단어#2 단어#3' 에서 단어만 추출해서 리스트로 변환
			words = get_words(cols)

			for word in sentiword:
				# 내가 입력한 단어가 swn 에 있다면
				if word in words:
					# print("For given word {0} - {1}".format(word, get_gloss(cols)))
					# print("P Score: {0}".format(get_positive(cols)))
					# print("N Score: {0}".format(get_negative(cols)))
					# print("O Score: {0}\n".format(get_objective(cols)))
					totalobject = totalobject + get_objective(cols)
					totalpositive = totalpositive + float(get_positive(cols))
					totalnegative = totalnegative + float(get_negative(cols))
					print(word, ":", totalobject, ":", totalpositive, ":", totalnegative)
					count = count + 1
	if count > 0:
		if totalpositive > totalnegative:
			print("Positive word : 1")
			print("Positive value : ", totalpositive)
			print("Negative value : ", totalnegative)
		else:
			print("Negative : -1")
			print("Positive value : ", totalpositive)
			print("Negative value : ", totalnegative)

		print("average object Score : ", totalobject / count)
	print(time.time() - start)


comment = """
good
"""
comment = ''.join(delete_symbols(comment))
print(comment)
sentiword = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", comment).split())
stop_words = set(stopwords.words('english'))

sentiword = sentiword.lower().split(" ")
filtered_sentence = [w for w in sentiword if not w in stop_words]
get_scores(os.path.dirname(os.path.abspath(__file__)) + "\\SentiWordNet_3.0.0.txt", filtered_sentence)

for i in filtered_sentence:
	synsets = swn.senti_synsets(i)
	for j in synsets:
		print(j)

a = swn.synsets('good')
for i in a:
	print(i)