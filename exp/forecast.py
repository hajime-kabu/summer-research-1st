import argparse
import os
import sys

from datetime import date
import pandas
from sklearn.naive_bayes import MultinomialNB
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


def prepare_x(df, n):
    previous_ys = [df["y"].shift(periods=i) for i in range(1, n + 1)]
    df = pandas.DataFrame({i: previous_ys[i] for i in range(n)})
    df["previous_y_list"] = df.values.tolist()
    joined_ys = df["previous_y_list"].map(lambda x: ",".join([str(elm) for elm in x]))
    
    return pandas.get_dummies(joined_ys)


def prepare_data(df, prep_config):
    index_col = prep_config["Index"]
    df = df.set_index(index_col)

    y_config = prep_config["Y"]
    orig_col = y_config["OriginalColumn"]

    col = df[orig_col]
    df = df.drop(orig_col, axis=1)

    threshold = y_config["Threshold"]
    df["y"] = col.map(lambda x: 0 if pandas.isna(x) or x <= threshold else 1)

    X = prepare_x(df, prep_config["N"])

    q = prep_config["Query"]
    X = X.query(q)
    df = df.query(q)
    y = df["y"]

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

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    result_df = pandas.DataFrame(
        {"正かい": y_test, "よそく": y_pred}, index=y_test.index
    )
    print(result_df)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)

    output(tn, fp, fn, tp, accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
