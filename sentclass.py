import sys
import pandas as pd
from classifier import Linear_SVM
from main import read_excel
from main import strip_records_with_useless_labels
from tweet import Tweet
# Read the arg entry


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 sentclass.py file sheetname")
        exit(0)

    filename = sys.argv[1]
    sheetname = sys.argv[2]

    if sheetname != "Obama" and sheetname != "Romney":
        print("sheetname must be Obama or Romney")
        exit(0)

    training_file = ""
    if sheetname == "Obama":
        training_file = "obama_training.csv"
    else:
        training_file = "romney_training.csv"

    # Read in the Training Data

    training_set = pd.read_csv(training_file, header=0)

    # Convert File to class Tweet
    training_tweets = []
    for index, line in training_set.iterrows():
        training_tweets.append(Tweet(line, training=True))

    training_tweets = strip_records_with_useless_labels(training_tweets)

    # Train the classifier
    cls = Linear_SVM(training_tweets, "text")

    # Read the test data
    try:
        test_set = pd.read_csv(filename, header=0)

    except:
        print("Could not open file: " + filename + " Sheet: " + sheetname)
        exit(0)

    for index, row in test_set.iterrows():
        tweet = Tweet(row, training=False)
        if tweet.text == "":
            tweet.classified = 0
        else:
            cls.classify([tweet])
        print(str(tweet.id) + ";;" + str(tweet.classified))

