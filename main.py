import pandas as pd
import sys
import codecs
from tweet import Tweet
from Classifier_Tester import Classifier_Tester
import classifier
import nltk

def read_excel(input_file, pos_words, neg_words):
    tweets = []
    for index, row in input_file.iterrows():
        tweets.append(Tweet(row, pos_words, neg_words))
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

    # Positive and negative word lexicon from:
    #
    # Minqing Hu and Bing Liu.
    # "Mining and Summarizing Customer Reviews." Proceedings of the ACM SIGKDD International Conference
    #   on Knowledge Discovery and Data Mining(KDD - 2004), Aug 22 - 25, 2004, Seattle, Washington, USA,

    pos_words = open("opinion-lexicon-English/positive-words.txt", 'r').readlines()
    neg_words = open("opinion-lexicon-English/negative-words.txt", 'r').readlines()

    for i, word in enumerate(pos_words):
        pos_words[i] = word.replace('\r\n', '')

    for i, word in enumerate(neg_words):
        neg_words[i] = word.replace('\r\n', '')

    input_file_obama = pd.read_excel("training_data.xlsx", header=0, sheet_name="Obama", dtype=unicode)
    input_file_romney = pd.read_excel("training_data.xlsx", header=0, sheet_name="Romney", dtype=unicode)

    tweets_obama = read_excel(input_file_obama, pos_words, neg_words)
    tweets_romney = read_excel(input_file_romney, pos_words, neg_words)

    tweets_obama = strip_records_with_useless_labels(tweets_obama)
    tweets_romney = strip_records_with_useless_labels(tweets_romney)


    classifiers = [classifier.Opinion_Word_Count, classifier.Two_Step, classifier.Naive_Bayes, classifier.Naive_Bayes_Op_Words, classifier.Linear_SVM, classifier.Linear_SVM_w_Opinion_Words, classifier.Classifier, classifier.SGD]
    sets = [tweets_obama, tweets_romney]
    features = ["text", "hashtags", "callout", "links"] #, "bigrams"]
    results = []

    for classifier in classifiers:
        for set in sets:
            if set is tweets_obama:
                set_name = "obama"
            else:
                set_name = "romney"

            for feature in features:
                tester = Classifier_Tester(set, 10, classifier, feature)
                a, p, r, f = tester.run_test()

                s = getattr(classifier, "name") + ", " + set_name + ", " + feature + ", " + str(a) + ", "
                for i in [0, 1, 2]:
                    s += (str(p[i]) + ", " + str(r[i]) + ", " + str(f[i]) + ", ")

                print(s)
