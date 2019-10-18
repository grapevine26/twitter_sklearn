import re
import os
import csv
from nltk.corpus import stopwords
from get_twitter.get_tweet import get_tweet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def split_line(line):
	cols = line.split("\t")
	return cols


def get_words(cols):
	words_ids = cols[4].split(" ")
	words = [w.split("#")[0] for w in words_ids]
	return words


def get_positive(cols):
	return cols[2]


def get_negative(cols):
	return cols[3]


def get_objective(cols):
	return 1 - (float(cols[2]) + float(cols[3]))


def get_gloss(cols):
	return cols[5]


def get_scores():
	swn_F = open(os.path.dirname(os.path.abspath(__file__)) + "\\SentiWordNet_3.0.0.txt")

	swn_F = swn_F.readlines()
	reviews_test = []
	train_f = open('train.csv', 'r', encoding='UTF-8')
	# for line in csv.reader(train_f):
	# 	if line[1] == '1':
	# 		sentiword = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", line[2]).split())
	# 		stop_words = set(stopwords.words('english'))
	#
	# 		sentiword = sentiword.lower().split(" ")
	# 		filtered_sentence = [w.replace('#', '').replace('!', '') for w in sentiword if not w in stop_words]
	# 		# print(filtered_sentence)
	# 		if '' in filtered_sentence:
	# 			filtered_sentence.remove('')
	# 			if '' in filtered_sentence:
	# 				filtered_sentence.remove('')
	# 		reviews_test.append(filtered_sentence)

	tweet_list = get_tweet()
	for line in tweet_list:
		sentiword = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", line).split())
		stop_words = set(stopwords.words('english'))

		sentiword = sentiword.lower().split(" ")
		filtered_sentence = [w.replace('#', '').replace('!', '') for w in sentiword if not w in stop_words]
		# print(filtered_sentence)
		if '' in filtered_sentence:
			filtered_sentence.remove('')
			if '' in filtered_sentence:
				filtered_sentence.remove('')
		reviews_test.append(filtered_sentence)

	pos = 0
	neg = 0
	for sentiword in reviews_test:
		totalobject = 0.0
		count = 0.0
		totalpositive = 0.0
		totalnegative = 0.0
		print(sentiword)
		# SentiWordNet 파일
		for line in swn_F:
			if not line.startswith("#"):
				# 라인 한 줄에서 탭 split
				cols = split_line(line)
				# 단어#1 에서 단어만 추출해서 리스트
				words = get_words(cols)
				# 내가 입력한 단어
				for word in sentiword:
					# 내가 입력한 단어가 swn 에 있다면
					if word in words:
						if word == "not":
							totalobject = totalobject + 0
							totalpositive = totalpositive + 0
							totalnegative = totalnegative + 16
							count = count + 1
						else:
							# print("For given word {0} - {1}".format(word, get_gloss(cols)))
							# print("P Score: {0}".format(get_positive(cols)))
							# print("N Score: {0}".format(get_negative(cols)))
							# print("O Score: {0}\n".format(get_objective(cols)))
							totalobject = totalobject + get_objective(cols)
							totalpositive = totalpositive + float(get_positive(cols))
							totalnegative = totalnegative + float(get_negative(cols))
							# print(word, ":", totalobject, ":", totalpositive, ":", totalnegative)
							count = count + 1

		if count > 0:
			if totalpositive > totalnegative:
				pos += 1
				# print('pos :', pos)
				# print("Positive value : ", totalpositive)
				# print("Negative value : ", totalnegative)
			else:
				neg += 1
				# print('neg :', neg)

			# print("Positive value : ", totalpositive)
				# print("Negative value : ", totalnegative)
			# print("average object Score : ", totalobject / count)
	print(len(reviews_test))
	print(pos)
	print(neg)


comment = """
the service is horrible, but the food here is great
"""

# get_scores()
# stop_words = set(stopwords.words('english'))
# print(stop_words)
score = analyser.polarity_scores(comment)
print(score)