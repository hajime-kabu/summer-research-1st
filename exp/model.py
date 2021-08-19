import numpy
import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class MyModel(object):
    def __init__(self, config):
        self.nb = MultinomialNB()
        self.classifier = globals()[config["Classifier"]]
        self.default_params = config.get("DefaultParams", {})
        self.param_grid = config["ParamGrid"]
        self.cv = config["CV"]
        self.scoring = config["Scoring"]
    
    def _get_dummy_x(self, X):
        result = pandas.DataFrame()

        for col in X:
            dummies = pandas.get_dummies(X[col])
            for dummy_label in dummies:
                col_name = "{},{}".format(col, dummy_label)
                result[col_name] = dummies[dummy_label]
        return result
    
    def _transform_x(self, X):
        dummy_x = self._get_dummy_x(X)

        columns = dummy_x.columns
        log_prob = [
            {
                columns[j]: self.nb.feature_log_prob_[i][j] 
                for j in range(len(columns))
            }
            for i in range(len(self.nb.feature_log_prob_))
        ]

        data = {}
        for gap in X:
            past_ys = X[gap]
            new_values = []
            for date in X.index:
                past_y = past_ys[date]
                log_prob_key = "{},{}".format(gap, past_y)
                new_value = log_prob[1][log_prob_key] - log_prob[0][log_prob_key]
                new_values.append(new_value)
            data[gap] = new_values
        prior = []
        for date in X.index:
            prior.append(self.nb.class_log_prior_[1] - self.nb.class_log_prior_[0])
        data["prior"] = prior

        return pandas.DataFrame(data, index=X.index)

    def fit(self, X, y):
        dummy_x = self._get_dummy_x(X)

        self.nb.fit(dummy_x, y)

        data = self._transform_x(X)

        self.main_model = GridSearchCV(
            self.classifier(**self.default_params),
            self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
        )
        self.main_model.fit(data, y)
    
    def predict(self, X):
        data = self._transform_x(X)
        return self.main_model.predict(data)
