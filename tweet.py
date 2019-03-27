import nltk
import pandas as pd
import re
from datetime import datetime, date, time


class Tweet:

    def type_check(self, var):
        if not isinstance(var, unicode):
             return u""
        return var

    def __init__(self, line, parser):
        # print(line)
        self.text = self.type_check(line['text'])
        self.date = self.type_check(line['date'])
        self.time = self.type_check(line['time'])
        self.category = self.type_check(line['class'])

        # Gather features on the tweet
        self.hashtags = ""
        self.callout = ""
        self.links = 0
        self.trim_tags()
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

        self.bigrams = self.extract_relevant_bigrams(parser)
        if self.links > 0:
            self.links = "true"
        else:
            self.links = "false"

    def extract_relevant_bigrams(self, parser):
        if len(self.text.split()) == 0:
            return ""

        tags = parser.tag(self.text.split())
        bigrams = ""
        for index, (word, tag) in enumerate(tags):
            # The last word can't be the start of a bigram
            if index > len(tags) - 2:
                break
            next_tuple = tags[index + 1]
            next_word, next_tag = next_tuple
            if tag == "JJ":
                if next_tag in ["NN", "NNS"]:
                    bigrams += (word + " " + next_word)
                if next_tag in ["JJ"] and index < len(tags) - 3:
                    next_next_word, next_next_tag = tags[index + 2]
                    if next_next_tag not in ["NN, NNS"]:
                        bigrams += (word + " " + next_word)

            if tag in ["RB", "RBR", "RBS"]:
                if next_tag in ["VB", "VBD", "VBN", "VBG"]:
                    bigrams += (word + " " + next_word)
                if next_tag == "JJ" and index < len(tags) - 3:
                    next_next_word, next_next_tag = tags[index + 2]
                    if next_next_tag not in ["NN, NNS"]:
                        bigrams += (word + " " + next_word)

            if tag in ["NN", "NNS"]  and next_tag == "JJ" and index < len(tags) - 3:
                next_next_word, next_next_tag = tags[index + 2]
                if next_next_tag not in ["NN, NNS"]:
                    bigrams += (word + " " + next_word)
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