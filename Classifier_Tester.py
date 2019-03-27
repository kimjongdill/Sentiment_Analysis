import random
from statistics import mean

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



    def run_test(self):
        accuracy = []

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

        return mean(accuracy)
