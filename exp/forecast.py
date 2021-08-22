import argparse
import os
import sys

from datetime import date
import pandas
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

sys.path.append("..")

from lib import base_dir_path
from lib import load_json
from exp.model import MyModel

config_path = os.path.join(base_dir_path, "exp", "config.json")
config = load_json(config_path)


def read_data(data_dir, data_config):
    data_file = data_config["File"]
    data_path = os.path.join(base_dir_path, data_dir, data_file)
    data_params = data_config["ReadCsvParams"]

    return pandas.read_csv(data_path, **data_params)


def prepare_x(df, n):
    result = pandas.DataFrame()

    for i in range(n):
        periods = i + 1
        result[periods] = df["y"].shift(periods=periods)
    return result


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

    model = MyModel()
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
