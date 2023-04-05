import json
from collections import defaultdict

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from naive_bayes import NaiveBayes

if __name__ == "__main__":
    iris_data = datasets.load_iris()
    x = iris_data.data
    y = iris_data.target

    test_sizes = [i/150 for i in range(149, 0, -1)]
    results = defaultdict(list)
    results['x'] = test_sizes

    for size in test_sizes:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=420)

        classifier = NaiveBayes()
        classifier.fit(x_train, y_train)

        y_predict = classifier.predict(x_test)

        accuracy = np.sum(y_test == y_predict) / len(y_test)
        results['y'].append(accuracy)

    with open("results.json", "w") as file:
        json.dump(results, file)