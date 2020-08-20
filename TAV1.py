import tweepy
from datetime import date, datetime, timedelta
import unicodedata
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize
import time

import pandas
pandas.set_option('display.max_rows', 50)
pandas.set_option('display.max_columns', 10)
pandas.set_option('display.width', 2000)

def authentication(api_key, api_secret_key, access_token, access_token_secret):
    print('Authenticating...')

    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

    print('Authentication successful.')
    return api

def search_query(query, api):
    instance_tweets = []
    for tweet in api.search(q = query, lang = 'en', count = 10, result_type = 'popular', until = date.today()):
        instance_tweets.append((tweet.created_at, tweet.user.screen_name, tweet.user.name, tweet.user.location, tweet.user.followers_count, tweet.user.verified, tweet.entities['hashtags'], tweet.text))
    
    return instance_tweets

def sentiment_analysis(data):
    sentiments = []
    analyzer = SentimentIntensityAnalyzer()

    for tweet in data:
        tokenized_tweet = sent_tokenize(tweet[7])
        temp_polarity_scores = []
        neg_score = 0
        neu_score = 0
        pos_score = 0
        compound_score = 0

        for sentence in tokenized_tweet:
            temp_polarity_scores.append(analyzer.polarity_scores(sentence))
        
        for score in temp_polarity_scores:
            neg_score += score['neg']
            neu_score += score['neu']
            pos_score += score['pos']
            compound_score += score['compound']
        
        avg_neg_score = neg_score / len(temp_polarity_scores)
        avg_neu_score = neu_score / len(temp_polarity_scores)
        avg_pos_score = pos_score / len(temp_polarity_scores)
        avg_compound_score = compound_score / len(temp_polarity_scores)

        sentiments.append((avg_neg_score, avg_neu_score, avg_pos_score, avg_compound_score))

    return sentiments




def compile_dataframe(data, query, sentiments):
    df = pandas.DataFrame()

    record_date = []
    tweet_date = []
    screen_name = []
    name = []
    location = []
    followers = []
    verified = []
    # hashtags = []
    text = []
    neg_scores = []
    neu_scores = []
    pos_scores = []
    compound_scores = []

    for tweet_no in range(len(data)):
        record_date.append(date.today())
        tweet_date.append(data[tweet_no][0].date())
        screen_name.append(data[tweet_no][1])
        name.append(unicodedata.normalize('NFD', data[tweet_no][2]).encode('ascii', 'ignore').decode('utf-8'))
        location.append(data[tweet_no][3])
        followers.append(data[tweet_no][4])
        verified.append(data[tweet_no][5])
        # hashtags.append(unicodedata.normalize('NFD', data[tweet_no][6]).encode('ascii', 'ignore').decode('utf-8'))
        text.append(unicodedata.normalize('NFD', data[tweet_no][7]).encode('ascii', 'ignore').decode('utf-8'))
        neg_scores.append(sentiments[tweet_no][0])
        neu_scores.append(sentiments[tweet_no][1])
        pos_scores.append(sentiments[tweet_no][2])
        compound_scores.append(sentiments[tweet_no][3])

    df['Record Date'] = record_date
    df['Tweet Date'] = tweet_date
    df['Screen Name'] = screen_name
    df['Name'] = name
    df['Location'] = location
    df['Follower Count'] = followers
    df['Verified'] = verified
    # df['Hashtags'] = hashtags
    df['Tweet'] = text
    df['Negative Sentiment'] = neg_scores
    df['Neutral Sentiment'] = neu_scores
    df['Positive Sentiment'] = pos_scores
    df['Compound Sentiment'] = compound_scores

    df.index.name = 'Rank'

    return df

def main():
    api_key = 'Your API Key'
    api_secret_key = 'Your API Secret'
    access_token = 'Your Access Token'
    access_token_secret = 'Your Access Token Secret'

    query = 'Your Topic Here'

    api = authentication(api_key, api_secret_key, access_token, access_token_secret)
    data = search_query(query, api)
    sentiments = sentiment_analysis(data)

    dataframe = compile_dataframe(data, query, sentiments)
    print(dataframe)
    # output_file = open(f'Exported Data\\{query}.csv', 'a+', newline = '')
    dataframe.to_csv(f'Exported Data\\{query}.csv', sep = ',', encoding = 'utf-8', mode = 'a', header = False)


if __name__ == '__main__':
    main()