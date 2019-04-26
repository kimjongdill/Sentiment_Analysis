import sys
import pandas as pd
from classifier import Linear_SVM
from main import read_excel
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

    # Read in the Training Data

    training_set = pd.read_excel("training_data.xlsx", header=0, sheet_name=sheetname, dtype=unicode)
    training_set = read_excel(training_set, [], [])

    # Train the classifier
    cls = Linear_SVM(training_set, "text")

    # Read the test data
    try:
        test_set = pd.read_excel(filename, header=0, sheet_name=sheetname, dtype=unicode)

    except:
        print("Could not open file: " + filename + " Sheet: " + sheetname)
        exit(0)

    for index, row in test_set.iterrows():
        tweet = Tweet(row, [], [])
        if tweet.text == "":
            tweet.classified = 0
        else:
            cls.classify([tweet])
        test_set.loc[index, 'Your class label'] = tweet.classified


    with pd.ExcelWriter("output.xlsx") as writer:
        test_set.to_excel(writer, sheet_name=sheetname, index=False)
