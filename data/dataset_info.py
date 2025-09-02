import os
import random
from .anomaly_generator import WhiteNoiseGenerator, DRAEMGenerator
import json

def get_data_root(dataset):
    assert dataset in DATA_SUBDIR, f"Only support {DATA_SUBDIR.keys()}, but entered {dataset}"
    return os.path.join(DATA_ROOT, DATA_SUBDIR[dataset])

import socket
import os


DATA_ROOT = '/home/duke/playground/side_pro/M2AD_data'

DATA_SUBDIR = {
    # 'm2ad': 'M2AD_256',
    'm2ad': 'M2AD',

}

# CLASS_NAMES = {
#     'm2ad': ['Bird', 'Car', 'Cube', 'Dice', 'Doll', 'Holder', 'Motor', 'Ring', 'Teapot', 'Tube']
# }

CLASS_NAMES = {
    'm2ad': ['Bird']
}


ANOMALY_GENERATOR = {
    'white_noise': WhiteNoiseGenerator,
    'draem': DRAEMGenerator
}

EXPERIMENTAL_SETUP = {
    'unsupervised': 'meta_unsupervised.json',
}


def _split_meta_unsupervised(info: dict):
    return [info], [EXPERIMENTAL_SETUP['unsupervised']]

def split_meta(info: dict, root: str):
    results_info = []
    results_json = []

    _info, _json = _split_meta_unsupervised(info)
    results_info += _info
    results_json += _json

    for _info, _json in zip(results_info, results_json):
        full_path = os.path.join(root, _json)

        with open(full_path, 'w') as f:
            f.write(json.dumps(_info, indent=4) + "\n")

        print(f'Save {_json}...')