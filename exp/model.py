import numpy
import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class MyModel(object):
    def __init__(self):
        self.nb = MultinomialNB()
    
    def _get_dummy_x(self, X):
        result = pandas.DataFrame()

        for col in X:
            dummies = pandas.get_dummies(X[col])
            for dummy_label in dummies:
                col_name = "{},{}".format(col, dummy_label)
                result[col_name] = dummies[dummy_label]
        return result

    def fit(self, X, y):
        data = self._get_dummy_x(X)

        self.nb.fit(data, y)
    
    def predict(self, X):
        data = self._get_dummy_x(X)
        return self.nb.predict(data)
