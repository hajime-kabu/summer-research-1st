import json


def load_json(fullpath):
    with open(fullpath, "rb") as fp:
        payload = fp.read().decode("utf-8")
        return json.loads(payload)
