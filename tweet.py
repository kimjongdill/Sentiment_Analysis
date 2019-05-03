import nltk
import pandas as pd
import re
from datetime import datetime, date, time
from bisect import bisect_left

def find(word, list):
    i = bisect_left(list, word)
    if i != len(list) and list[i] == word:
        return True
    return False

class Tweet:

    def type_check(self, var):
        if not isinstance(var, unicode):
             return u""
        return var

    def __init__(self, line, training=True):
        # print(line)
        self.text = (str(line['Tweet_text'])).decode("utf-8", "ignore")      #self.type_check(line['Tweet_Text'])
        self.id = line['Tweet_ID']          #self.type_check(line['Tweet_ID'])
        # self.time = self.type_check(line['time'])
        # Only training tweets will have a class label in the constructor
        if training == True:
            self.category = line['class']   # self.type_check(line['class'])

        # Gather features on the tweet
        self.hashtags = ""
        self.callout = ""
        self.links = 0
        self.trim_tags()
        self.opinion_count = 0
        text = self.text.split()

        # Store the hashtags, callout, and count links
        for word in text:
            if word[0] == "#":
                self.hashtags += (word + " ")
            if word[0] == '@':
                self.callout += (word + " ")
            if word.find("http") >= 0:
                self.links += 1


        # Remove leading whitespace
        self.text = re.sub(r"^\s*", "", self.text)

        # Callouts at the beginning seem aimed at friends
        # Lets remove them and see what we get
        while bool(re.search(r"^@\S+\b", self.text)) or bool(re.search(r"^[R,r][T,t]\b", self.text)):
            self.text = re.sub(r"^@\S+\s*", "", self.text)
            self.text = re.sub(r"^[R,r][T,t]\s*", "", self.text)

        # Remove Hyperlinks
        self.trim_links()
        self.text = self.text.replace("@", "")
        self.text = self.text.replace("#", "")

        # self.count_opinion_words(pos_words, neg_words)
        # self.subjective = True if self.opinion_count >= 1 else False

        # try:
        #     self.subjective = False if int(self.category) == 0 else True
        # except:
        #     self.subjective = False

        #self.bigrams = self.extract_relevant_bigrams(parser)
        if self.links > 0:
            self.links = "true"
        else:
            self.links = "false"

    def count_opinion_words(self, pos_words, neg_words):
        #tagged_op_words = self.extract_relevant_op_words(parser)
        pos = 1
        self.opinion_score = 0
        tokenizer = nltk.RegexpTokenizer(r'\w+')

        for word in tokenizer.tokenize(self.text):
            try:
                word = word.lower()
                if find(word, pos_words):
                    self.opinion_score += 1
                    self.opinion_count += 1
                if find(word, neg_words):
                    self.opinion_score -= 1
                    self.opinion_count += 1
            except:
                continue


    def extract_relevant_op_words(self, parser):
        if len(self.text.split()) == 0:
            return ""

        tags = parser.tag(self.text.split())
        bigrams = []
        for index, (word, tag) in enumerate(tags):
            # The last word can't be the start of a bigram
            if index > len(tags) - 2:
                break
            next_tuple = tags[index + 1]
            next_word, next_tag = next_tuple
            if tag == "JJ":
                if next_tag in ["NN", "NNS"]:
                    bigrams.append((word, tag))
                    bigrams.append((next_word, next_tag))
                if next_tag in ["JJ"] and index < len(tags) - 3:
                    next_next_word, next_next_tag = tags[index + 2]
                    if next_next_tag not in ["NN, NNS"]:
                        bigrams.append((word, tag))
                        bigrams.append((next_word, next_tag))

            if tag in ["RB", "RBR", "RBS"]:
                if next_tag in ["VB", "VBD", "VBN", "VBG"]:
                    bigrams.append((word, tag))
                    bigrams.append((next_word, next_tag))
                if next_tag == "JJ" and index < len(tags) - 3:
                    next_next_word, next_next_tag = tags[index + 2]
                    if next_next_tag not in ["NN, NNS"]:
                        bigrams.append((word, tag))
                        bigrams.append((next_word, next_tag))

            if tag in ["NN", "NNS"]  and next_tag == "JJ" and index < len(tags) - 3:
                next_next_word, next_next_tag = tags[index + 2]
                if next_next_tag not in ["NN, NNS"]:
                    bigrams.append((word, tag))
                    bigrams.append((next_word, next_tag))
        return bigrams

    def __str__(self):
        string = u"text: " + self.text.encode('ascii', 'ignore') + \
                 u" date: " + self.date.encode('ascii', 'ignore') + \
                 u" time: " + self.time.encode('ascii', 'ignore') + \
                 u" category: " + self.category.encode('ascii', 'ignore')
        return string

    def trim_tags(self):
        self.text = self.text.replace("<e>", "")
        self.text = self.text.replace("</e>", "")
        self.text = self.text.replace("<a>", "")
        self.text = self.text.replace("</a>", "")


    def trim_links(self):
        self.text = re.sub(r'http\S+\s*', '', self.text)