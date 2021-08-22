import numpy
import pandas
from sklearn.dummy import DummyClassifier


class MyModel(object):
    def __init__(self):
        self.model = DummyClassifier(strategy="most_frequent")

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
