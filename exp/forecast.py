import argparse
import os
import sys

import pandas
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

sys.path.append("..")

from lib import base_dir_path
from lib import load_json

config_path = os.path.join(base_dir_path, "exp", "config.json")
config = load_json(config_path)


def read_data(data_dir, data_config):
    data_file = data_config["File"]
    data_path = os.path.join(base_dir_path, data_dir, data_file)
    data_params = data_config["ReadCsvParams"]

    return pandas.read_csv(data_path, **data_params)


def prepare_x(df, x_config):
    use_cols = x_config["UseColumns"]
    return df[use_cols]


def prepare_data(df, prep_config):
    index_col = prep_config["Index"]
    df = df.set_index(index_col)

    y_config = prep_config["Y"]
    orig_col = y_config["OriginalColumn"]

    col = df[orig_col]
    df = df.drop(orig_col, axis=1)

    threshold = y_config["Threshold"]
    y = col.map(lambda x: 0 if pandas.isna(x) or x <= threshold else 1)

    x_config = prep_config["X"]
    X = prepare_x(df, x_config)

    return X, y


def output(tn, fp, fn, tp, accuracy):
    print(f"正かいは晴れでよそくも晴れ: {tn}, 正かいは晴れでよそくは雨: {fp}")
    print(f"正かいは雨でよそくは晴れ: {fn}, 正かいは雨でよそくも雨: {tp}")
    print()
    print(f"正答りつ: {accuracy}")


def main():
    data_config = config["Data"]
    data_dir = data_config["Dir"]

    past_data_config = data_config["PastData"]
    past_df = read_data(data_dir, past_data_config)

    current_data_config = data_config["CurrentData"]
    current_df = read_data(data_dir, current_data_config)

    prep_config = data_config["FeatureEngineering"]
    X_train, y_train = prepare_data(past_df, prep_config)
    X_test, y_test = prepare_data(current_df, prep_config)

    model = DummyClassifier(strategy="uniform")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)

    output(tn, fp, fn, tp, accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
