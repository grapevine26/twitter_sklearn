from nltk.corpus import sentiwordnet as swn
# import nltk
# nltk.download()
# bd = swn.senti_synsets('bad')
# print(bd)
# print(bd.pos_score())
# print(bd.neg_score())
# print(bd.obj_score())
# print(list(swn.senti_synsets('good')))
# a = swn.senti_synsets('good')
# for i in a:
#     print(i)
#
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk


def penn_to_wn(tag):
	"""
	Convert between the PennTreebank tags to simple Wordnet tags
	"""
	# 형용사
	if tag.startswith('J'):
		return wn.ADJ
	# 명사
	elif tag.startswith('N'):
		return wn.NOUN
	# 부사
	elif tag.startswith('R'):
		return wn.ADV
	# 동사
	elif tag.startswith('V'):
		return wn.VERB

	return None


lemmatizer = WordNetLemmatizer()


def get_sentiment(word, tag):
	""" returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """

	wn_tag = penn_to_wn(tag)
	if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
		return []

	lemma = lemmatizer.lemmatize(word, pos=wn_tag)
	print(lemma)
	if not lemma:
		return []

	synsets = wn.synsets(word, pos=wn_tag)
	if not synsets:
		return []

	# Take the first sense, the most common
	synset = synsets[0]
	swn_synset = swn.senti_synset(synset.name())

	return [swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()]


lm = WordNetLemmatizer()
ps = PorterStemmer()
words_data = ['backs', 'movie', 'is', 'wonderful']
words_data = [lm.lemmatize(x) for x in words_data]

pos_val = nltk.pos_tag(words_data)
print(pos_val)
senti_val = [get_sentiment(x, y) for (x, y) in pos_val]
print(senti_val)
########################################################################
#
# text = "I was at the same screenwriters conference and saw the movie. life's I thought the writer - Sue Smith - very clearly summarised what the film was about. However, the movie really didn't need explanation. I thought the themes were abundantly clear, and inspiring. A movie which deals with the the ability to dare, to face fear - especially fear passed down from parental figures - and overcome it and, in doing so, embrace life possibilities, is a film to be treasured and savoured. I enjoyed it much more than the much-hyped 'Somersault.' I also think Mandy62 was a bit unkind to Hugo Weaving. As a bloke about his vintage, I should look so good! I agree that many Australian films have been lacklustre recently, but 'Peaches' delivers the goods. I'm glad I saw it.".replace('.', '').replace(',', '').replace('-', '')
#
# # 불용어 리스트
# stop_words = set(stopwords.words("english"))
# print(stop_words)
# # 텍스트 토큰화
# words = text.split()
# print(words)
# # 토근화된 단어 중 불용어 제거
# result = [word for word in words if word.lower() not in stop_words]
# # 단어와 단어의 품사르 튜플로 리턴
# tag = pos_tag(result)
# for i in tag:
#     # swn 에 적용할 품사의 약자 리턴
#     tag = penn_to_wn(i[1])
#     # 표제어 추출 (어근)
#     n = WordNetLemmatizer()
#     try:
#         lemma = n.lemmatize(word=i[0], pos=tag)
#         synsets = wn.synsets(lemma, pos=tag)
#         if not synsets:
#           pass
#
#         # Take the first sense, the most common
#         synset = synsets[0]
#         swn_synset = swn.senti_synset(synset.name())
#         print(synset.lemmas(), ':', swn_synset)
#         print()
#     except Exception as e:
#         pass
