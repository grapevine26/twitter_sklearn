from nltk.sentiment.vader import SentimentIntensityAnalyzer
from get_twitter import get_tweet

analyser = SentimentIntensityAnalyzer()

search_word = 'a'
get_cnt = 3
start_date = '2019-10-01'
end_date = '2019-10-10'

tweet_list = get_tweet.get_tweet(search_word, get_cnt, start_date, end_date)

pos = 0
neg = 0
pos_f = open('pos.txt', 'w', encoding='UTF-8')
neg_f = open('neg.txt', 'w', encoding='UTF-8')
for text in tweet_list:
	if len(text) > 50:
		score = analyser.polarity_scores(text)
		if 0 < score['compound']:
			pos += 1
			pos_f.write(text)
		elif 0 > score['compound']:
			neg += 1
			neg_f.write(text)

print(len(tweet_list))
print(pos)
print(neg)
