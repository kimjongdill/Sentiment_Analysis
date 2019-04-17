import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import *
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

class Dummy_Classifier:

    def __init__(self):
        return

    def fit(self, X, Y):
        return

    def predict(self, X):
        result = []
        for record in X:
            result.append(random.randint(-1, 1))
        return result


class Classifier:

    name = "Random Guess"

    def __init__(self, training_set, feature):
        data = []
        labels = []
        self.feature = feature
        for record in training_set:
            data.append(getattr(record, feature))
            labels.append(int(record.category))

        self.clf = Pipeline([
            ('vect', CountVectorizer(lowercase=True, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', Dummy_Classifier()),
        ])

        self.clf.fit(data, labels)

    def classify(self, test_set):
        data = []
        for record in test_set:
            data.append(getattr(record, self.feature))

        predict = self.clf.predict(data)

        for iter, val in enumerate(predict):
            test_set[iter].classified = val

    def __str__(self):
        return self.name

class Random_Forest(Classifier):
    name = "Random Forest"

    def __init__(self, training_set, feature):
        data = []
        labels = []
        self.feature = feature
        for record in training_set:
            data.append(getattr(record, feature))
            labels.append(int(record.category))

        self.clf = Pipeline([
            ('vect', CountVectorizer(lowercase=True, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', RandomForestClassifier()),
        ])

        self.clf.fit(data, labels)


class K_Means(Classifier):

    name = "KMeans"

    def __init__(self, training_set, field):
        data = []
        labels = []
        for record in training_set:
            data.append(getattr(record, field))
            labels.append(int(record.category))

        self.clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', KMeans(init="k-means++", n_clusters=3, )),
        ])

        self.clf.fit(data, labels)



class DecisionTree(Classifier):

    name = "Decision Tree"

    def __init__(self, training_set, feature):
        data = []
        labels = []
        self.feature = feature
        for record in training_set:
            data.append(getattr(record, feature))
            labels.append(int(record.category))

        self.clf = Pipeline([
            ('vect', CountVectorizer(lowercase=True, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', DecisionTreeClassifier()),
        ])

        self.clf.fit(data, labels)


class SGD(Classifier):

    name = "SGD"

    def __init__(self, training_set, feature):
        data = []
        labels = []
        self.feature = feature
        for record in training_set:
            data.append(getattr(record, feature))
            labels.append(int(record.category))

        self.clf = Pipeline([
            ('vect', CountVectorizer(lowercase=True, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                  alpha=1e-3, random_state=42,
                                  max_iter=5, tol=None)),
        ])

        self.clf.fit(data, labels)

def augment_with_opinion_words(record):
    words = ""
    if record.opinion_score == 0:
        words = "opscoreneu"
    if record.opinion_score < 0:
        words = "opscoreneg"
    if record.opinion_score > 0:
        words = "opscorepos"
    return words


class Linear_SVM_w_Opinion_Words(Classifier):
    name = "Linear SVM w Op Words"

    def __init__(self, training_set, feature):
        data = []
        labels = []
        self.feature = feature

        for record in training_set:
            words = augment_with_opinion_words(record)
            r_data = getattr(record, feature)
            r_data += words

            data.append(r_data)
            labels.append(int(record.category))

        self.clf = Pipeline([
            ('vect', CountVectorizer(lowercase=True, ngram_range=(1, 2))),
            ('clf', LinearSVC()),
        ])

        self.clf.fit(data, labels)


    def classify(self, test_set):
        data = []
        for record in test_set:
            r_data = getattr(record, self.feature)
            r_data += augment_with_opinion_words(record)
            data.append(r_data)

        predict = self.clf.predict(data)

        for iter, val in enumerate(predict):
            test_set[iter].classified = val

class Naive_Bayes(Classifier):

    name = "Naive Bayes"

    def __init__(self, training_set, feature):
        data = []
        labels = []
        self.feature = feature
        for record in training_set:
            data.append(getattr(record, feature))
            labels.append(int(record.category))

        self.clf = Pipeline([
            ('vect', CountVectorizer(lowercase=True, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])

        self.clf.fit(data, labels)

class Naive_Bayes_Op_Words(Classifier):

    name = "Naive Bayes With Op Words"

    def __init__(self, training_set, feature):
        data = []
        labels = []
        self.feature = feature
        for record in training_set:
            r_data = getattr(record, feature)
            r_data += augment_with_opinion_words(record)
            data.append(r_data)
            labels.append(int(record.category))

        self.clf = Pipeline([
            ('vect', CountVectorizer(lowercase=True, ngram_range=(1, 2))),
            ('clf', MultinomialNB()),
        ])

        self.clf.fit(data, labels)

    def classify(self, test_set):
        data = []
        for record in test_set:
            r_data = getattr(record, self.feature)
            r_data += augment_with_opinion_words(record)
            data.append(r_data)

        predict = self.clf.predict(data)

        for iter, val in enumerate(predict):
            test_set[iter].classified = val

class Linear_SVM(Classifier):

    name = "Linear SVM"

    def __init__(self, training_set, feature):
        data = []
        labels = []
        self.feature = feature
        for record in training_set:
            data.append(getattr(record, feature))
            labels.append(int(record.category))

        self.clf = Pipeline([
            ('vect', CountVectorizer(lowercase=True, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearSVC()),
        ])

        self.clf.fit(data, labels)

class SciKit_SVC(Classifier):

    name = "SciKit SVC"

    def __init__(self, training_set, feature):
        data = []
        labels = []
        self.feature = feature
        for record in training_set:
            data.append(getattr(record, feature))
            labels.append(int(record.category))

        self.clf = Pipeline([
            ('vect', CountVectorizer(lowercase=True, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', SVC()),
        ])

        self.clf.fit(data, labels)
