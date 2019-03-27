import pandas as pd
import sys
import codecs
from tweet import Tweet
from Classifier_Tester import Classifier_Tester
import classifier
import nltk

def read_excel(input_file, parser):
    tweets = []
    for index, row in input_file.iterrows():
        tweets.append(Tweet(row, parser))
    return tweets

def strip_records_with_useless_labels(tweets):
    clean_tweets = []
    for tweet in tweets:
        if tweet.category in ["-1", "0", "1"]:
            clean_tweets.append(tweet)
    return clean_tweets

def strip_html_tags(tweets):
    for tweet in tweets:
        tweet.trim_tags()

if __name__ == "__main__":

    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    sys.stderr = codecs.getwriter('utf8')(sys.stderr)

    # Validate that Stanford Core NLP Server is Running
    parser = nltk.CoreNLPParser(url="http://localhost:9000", tagtype="pos")


    input_file_obama = pd.read_excel("training_data.xlsx", header=0, sheet_name="Obama", dtype=unicode)
    input_file_romney = pd.read_excel("training_data.xlsx", header=0, sheet_name="Romney", dtype=unicode)
    tweets_obama = read_excel(input_file_obama, parser)
    tweets_romney = read_excel(input_file_romney, parser)

    tweets_obama = strip_records_with_useless_labels(tweets_obama)
    tweets_romney = strip_records_with_useless_labels(tweets_romney)
    tweets_all = []
    tweets_all.extend(tweets_romney)
    tweets_all.extend(tweets_obama)

    for t in tweets_all:
        t.trim_links()
        # print(t.text)


    classifiers = [classifier.Linear_SVM, classifier.Classifier, classifier.Random_Forest, classifier.SGD, classifier.DecisionTree]
    sets = [tweets_obama, tweets_romney]
    features = ["text"] #, "hashtags", "callout", "links", "bigrams"]

    for classifier in classifiers:
        for set in sets:
            for feature in features:
                tester = Classifier_Tester(set, 10, classifier, feature)
                accuracy = tester.run_test()
                print(getattr(classifier, "name") + " " + feature + " " + accuracy.__str__())
