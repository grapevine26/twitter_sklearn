import datetime
import time
import re
import GetOldTweets3 as got
from tqdm import tqdm_notebook

REGEX_HTTP = 'https?://(\w*:\w*@)?[-\w.]+(:\d+)?(/([\w/_.]*(\?\S+)?)?)? ?…?'
REGEX_PIC = 'pic.twitter.com/[\w]+'
REGEX_TAG = '@[\w_]+'


def get_tweet(search_word, get_cnt, start_date, end_date):
	days_range = []

	start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
	end = datetime.datetime.strptime(end_date, "%Y-%m-%d")

	date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]
	for date in date_generated:
		days_range.append(date.strftime("%Y-%m-%d"))

	print("=== 설정된 트윗 수집 기간은 {} 에서 {} 까지 입니다 ===".format(days_range[0], days_range[-1]))
	print("=== 총 {}일 간의 데이터 수집 ===".format(len(days_range)))

	# 수집 기간 맞추기
	start_date = days_range[0]
	end_date = (datetime.datetime.strptime(days_range[-1], "%Y-%m-%d") + datetime.timedelta(days=1)).strftime(
		"%Y-%m-%d")  # setUntil이 끝을 포함하지 않으므로, day + 1

	# 트윗 수집 기준 정의
	tweetCriteria = got.manager.TweetCriteria().setQuerySearch(search_word) \
		.setSince(start_date) \
		.setUntil(end_date) \
		.setMaxTweets(get_cnt)

	# 수집 with GetOldTweet3
	print("Collecting data start.. from {} to {}".format(days_range[0], days_range[-1]))
	start_time = time.time()

	tweet = got.manager.TweetManager.getTweets(tweetCriteria)
	print("Collecting data end.. {0:0.2f} Minutes".format(time.time() - start_time))
	print("=== Total num of tweets is {} ===".format(len(tweet)))

	tweet_list = []
	for index in tqdm_notebook(tweet):
		# print('='*50)
		# print(index.username)
		# print(index.text)
		# print(index.date.strftime("%Y-%m-%d"))
		# print(index.date.strftime("%H:%M:%S"))
		# print(index.favorites)
		text = re.sub(REGEX_HTTP + '|' + REGEX_PIC + '|' + REGEX_TAG, '', index.text)
		tweet_list.append(text)

	return tweet_list


if __name__ == '__main__':
	word = 'https://twitter.com/'
	cnt = 100
	startDate = '2019-10-01'
	endDate = '2019-10-10'
	tweet_list = get_tweet(word, cnt, startDate, endDate)
