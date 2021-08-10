import os.path

import json

from .loader import load_json

abspath_for_this_script = os.path.abspath(__file__)
file_dir = os.path.dirname(abspath_for_this_script)
base_dir_path = os.path.join(file_dir, "..")

del abspath_for_this_script
del file_dir
