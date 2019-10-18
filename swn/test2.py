import re
import os
from nltk.corpus import stopwords


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


def get_scores(filepath, sentiword):
	f = open(filepath)
	totalobject = 0.0
	count = 0.0
	totalpositive = 0.0
	totalnegative = 0.0

	# SentiWordNet 파일
	for line in f:
		#
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
			print("Positive word : 1")
			print("Positive value : ", totalpositive)
			print("Negative value : ", totalnegative)
		else:
			print("Negative : -1")
			print("Positive value : ", totalpositive)
			print("Negative value : ", totalnegative)

		print("average object Score : ", totalobject / count)



comment = """
	
horrible quality images- eats batteries like crazy!!!!!  All of these Kodak Easyshare cameras seem to have the same problem with horrible image processing.  Colors look good but the compression is atrocious.  If I use the "2-megapixel" image as my computer wallpaper (~1megapixel) there is a great deal of JPEG artifacting evident.  These images are OKAY to print as 3.5" x 5" prints but not much larger.  Even 4"x6" prints start to show artifacting and noise.  As if it isn't bad enough to have poor images, the camera also eats 2 AA batteries for every 35 images captured!!! I have to bring a pack with extra batteries everywhere I take my camera!     	1

"""
# print(comment)
sentiword = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", comment).split())
stop_words = set(stopwords.words('english'))

sentiword = sentiword.lower().split(" ")
filtered_sentence = [w for w in sentiword if not w in stop_words]
# print(filtered_sentence)
get_scores(os.path.dirname(os.path.abspath(__file__)) + "\\SentiWordNet_3.0.0.txt", filtered_sentence)
