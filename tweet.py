import pandas as pd
import re
from datetime import datetime, date, time


class Tweet:

    def type_check(self, var):
        if not isinstance(var, unicode):
             return u""
        return var

    def __init__(self, line):
        # print(line)
        self.text = self.type_check(line['text'])
        self.date = self.type_check(line['date'])
        self.time = self.type_check(line['time'])
        self.category = self.type_check(line['class'])
        self.hashtags = ""
        self.callout = ""
        self.links = 0
        self.trim_tags()
        text = self.text.split()
        for word in text:
            if word[0] == "#":
                self.hashtags += (word + " ")
            if word[0] == '@':
                self.callout += (word + " ")
            if word.find("http") >= 0:
                self.links += 1

        if self.links > 0:
            self.links = "true"
        else:
            self.links = "false"


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
        self.text = re.sub(r'http.+', '', self.text)