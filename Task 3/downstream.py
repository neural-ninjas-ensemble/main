import torch
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def affine_transformation(x, a=0.6, b=1.2):
    return a * x + b

def padding(x):
    return np.pad(x, ((0, 0), (0, 10)), mode="constant", constant_values=0)

def shuffle(x):
    return np.random.default_rng(seed=42).permutation(x, axis=1)

def binary(x):
    x[x > 0.5] = 1
    x[x <= 0.5] = 0
    return x


if __name__ == '__main__':
    data = np.load("../DefenseTransformationEvaluate.npz")
    test_data = np.load("../DefenseTransformationSubmit.npz")
    print(data.files)
    print(test_data.files)
    print(data["representations"].shape)
    print(test_data["representations"].shape)
    print(data["labels"].shape)
    transform = lambda x: binary(padding(x))
    X_train, X_test, y_train, y_test = train_test_split(transform(data["representations"]), data["labels"], test_size=0.2, random_state=42)
    classifier = MLPClassifier(hidden_layer_sizes=(96, 32, 10), max_iter=1000)
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))

    X_train, X_test, y_train, y_test = train_test_split(data["representations"] + np.random.normal(size=data["representations"].shape, scale=1.25), data["labels"], test_size=0.2, random_state=42)
    classifier = MLPClassifier(hidden_layer_sizes=(96, 32, 10), max_iter=1000)
    classifier.fit(X_train, y_train)

    print(classifier.score(X_test, y_test))
    print(classifier.score(X_test + np.random.normal(size=X_test.shape, scale=1.25), y_test))
    result = test_data["representations"] + np.random.normal(size=test_data["representations"].shape, scale=1.25)
    np.savez("result2.npz", representations=result)