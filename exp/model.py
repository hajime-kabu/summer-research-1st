from itertools import product
from itertools import repeat

import numpy
import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class MyModel(object):
    def __init__(self):
        self.nb = MultinomialNB()

    def _get_dummy_x(self, X):
        result = X.apply(lambda row: "".join([str(int(elm)) for elm in row]), axis=1)

        result = pandas.get_dummies(result)
        all_patterns = product("01", repeat=len(X.columns))
        for pattern in all_patterns:
            key = "".join(pattern)
            if key in result:
                continue
            result[key] = pandas.Series(list(repeat(0, len(X))), X.index.values)
        return result

    def fit(self, X, y):
        dummy_x = self._get_dummy_x(X)
        self.nb.fit(dummy_x, y)
    
    def predict(self, X):
        dummy_x = self._get_dummy_x(X)
        return self.nb.predict(dummy_x)
