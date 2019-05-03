Sentiment Classifier
George Dill
UIC CS583 - Professor Bing Liu

https://github.com/kimjongdill/CS583_Project2
Made public after 5/3/2019

Files in Repo:
sentclass.py
main.py
Classifier_Tester.py
classifier.py
tweet.py
Sentiment Classification of Political Tweets.pdf

This repo contains files from a class project on sentiment classification of
political tweets from Spring 2019 section of CS583 - Data Mining Text Mining. 

A labeled data set of tweets about the 2012 presidential debates between 
Barack Obama and Mitt Romney is provided, separated by subject (Obama or 
Romney). 

We train a sentence level sentiment classifier on each data set. 

sentclass.py is the final sentiment classifier submitted. Usage: 

python3 sentclass.py <file_to_classify> {Obama, Romney}. 

<file_to_classify> must be a comma separated value file with headers
"Tweet_ID" and "Tweet_Text"

Classifications are printed to stdout with formate Tweet_ID;;Class

sentclass.py cleans the text of tweets by trimming leading twitter handles and 
hyperlinks. In the remaining text @ and # are removed. The remainder is split 
on white space, converted to lower case, and transformed to a tf-idf vector 
with unigrams and bigrams. 

A linear support vector machine is trained and used to classify the tweets in 
<file_to_classify>

tweet.py contains data class Tweet.py. The constructor peforms the data 
cleaning described above. The class contains artifacts of other feature 
extraction tested in main.py. 

classifier.py contains the Linear_SVM class and its parameters used by 
sentclass.py but also classifiers and parameters tried in the paper. 

Classifier_Tester.py drives n-fold cross validation on the classifier 
specified. 

main.py is the test driver for the various combinations of features tested 
for this project. However, tweet.py was modified for the turn-in version 
due to the change in format of input data. To test main.py please revert 
to commit # 0b86529213fd271220d5e875b98da2b5ca5b004f

main.py can be invoked with python3 main.py. 

This program requires scikit learn. install with pip install scikit-learn

Some experiments run by main.py require nltk, pip install nltk, and 
the stanford parser to be running on the local machine. For instructions 
on stanford parser see: https://nlp.stanford.edu/software/lex-parser.html

