import random
from statistics import mean

class Confusion_Matrix:
    def __init__(self, tp, fp, fn, classification):
        self.classification = classification
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __str__(self):
        s = "tp: " + self.tp.__str__() + " fp: " + self.fp.__str__() + " fn: " + self.fn.__str__() + " class: " + self.classification.__str__()
        return s

    def precision(self):
        if self.tp == 0:
            return 0.0
        return float(self.tp) / (float(self.tp) + float(self.fp))

    def recall(self):
        if self.tp == 0:
            return 0.0
        return float(self.tp) / (float(self.tp) + float(self.fn))

    def f_score(self):
        if self.tp == 0:
            return 0.0
        p = self.precision()
        r = self.recall()
        return (2*p*r) / (p + r)


class Classifier_Tester:

    def __init__(self, data_set, n, classifier, feature):

        if classifier is None:
            raise Exception("Classifier must be a function")

        if n <= 0 or n >= len(data_set):
            raise Exception("n-way validation must be greater than 0 less than data set size")

        if len(data_set) == 0:
            raise Exception("Cannot validate the empty set")

        self.feature = feature
        self.data_subset = []
        for i in range(n):
            self.data_subset.append([])

        for item in data_set:
            while True:
                rand = random.randint(0, 9)
                if len(self.data_subset[rand]) <= (len(data_set) / n) :
                    break
            self.data_subset[rand].append(item)

        self.classifier = classifier
        self.n = n

    def get_accuracy(self, test_set):
        correct = 0.0

        for record in test_set:
            if int(record.category) == record.classified:
                correct += 1
        return correct / len(test_set)

    def build_confusion_matrix(self, test_set, classification):
        true_pos = 0
        false_pos = 0
        false_neg = 0

        for record in test_set:
            # True Positive The label matches the classification and the classification is positive
            if ((int(record.category) == record.classified) and (record.classified == classification)):
                true_pos += 1
            # False Positive The label does not match the classification and the classification is positive
            if (int(record.category) != record.classified) and (record.classified == classification):
                false_pos += 1
            # False Negative The label matches the classification but the classifier does not
            if (int(record.category) == classification) and (record.classified != classification):
                false_neg += 1

        return Confusion_Matrix(true_pos, false_pos, false_neg, classification)


    def run_test(self):
        accuracy = []
        matrices = [[],[],[]]

        for i in range(self.n):
            accuracy.append(0)
            training_set = []
            test_set = self.data_subset[i]
            for j in range(self.n):
                if i is not j:
                    training_set.extend(self.data_subset[j])
            classifier = self.classifier(training_set, self.feature)
            classifier.classify(test_set)
            accuracy[i] = self.get_accuracy(test_set)
            for c in [0, 1, 2]:
                matrices[c].append(self.build_confusion_matrix(test_set, c - 1))

        a = mean(accuracy)
        p = [[],[],[]]
        r = [[],[],[]]
        f = [[],[],[]]
        for c in [0, 1, 2]:
            for matrix in matrices[c]:
                p[c].append(matrix.precision())
                r[c].append(matrix.recall())
                f[c].append(matrix.f_score())
            p[c] = mean(p[c])
            r[c] = mean(r[c])
            f[c] = mean(f[c])


        return (a, p, r, f)
