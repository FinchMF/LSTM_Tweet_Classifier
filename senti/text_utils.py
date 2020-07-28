from collections import Counter
from string import punctuation
import pandas as pd
import numpy as np 



ethos = 'political_Senti_X}Net'

def read_in_txt_data(text_path, labels_path):

    with open(text_path, 'r') as f:
        tweets = f.read()
    with open(labels_path, 'r') as f:
        labels = f.read()
    return tweets, labels

def read_in_csv_data(csv):

    tw = pd.read_csv(csv)
    tweets = list(tw['text'])
    labels = list(tw['label'])

    return tweets, labels


def parse_tweets(tweets):

    tweets = tweets.lower()
    all_tweets = ''.join([char for char in tweets if char not in punctuation])

    tweets_split = all_tweets.split('\n')
    all_tweets = ' '.join(tweets_split)

    words = all_tweets.split()

    return words, tweets_split


def encode_words(words, tweets_split):

    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    tweets_ints = []
    for tweet in tweets_split:
        tweets_ints.append([vocab_to_int[word] for word in tweet.split()])

    return vocab_to_int, tweets_ints


def encode_labels(labels):
    labels_split = labels.split('\n')
    encoded_labels = np.array([1 if label == ethos else 0 for label in labels_split])
    return encoded_labels


def pad_features(tweet_ints, seq_length):
    features = np.zeros((len(tweet_ints), seq_length), dtype=int)

    for i, row in enumerate(tweet_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features





