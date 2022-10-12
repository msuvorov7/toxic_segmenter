import argparse
import logging
import os
import pickle
import sys

import numpy as np
import yaml
from typing import Tuple

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def clear_text():
    pass


def download_dataframe(mode: str, directory_path: str) -> Tuple[list, list]:
    """
    Загрузка датасета в память
    :param mode: train или test датафрейм
    :param directory_path: путь до папки с сырыми данными
    :return:
    """
    tokens_filename = f'{mode}_tokens.pkl'
    tokens_filename = directory_path + '/' + tokens_filename

    tags_filename = f'{mode}_tags.pkl'
    tags_filename = directory_path + '/' + tags_filename

    with open(tokens_filename, 'rb') as file:
        tokens = pickle.load(file)

    with open(tags_filename, 'rb') as file:
        tags = pickle.load(file)

    tags_flatten = [item for sublist in tags for item in sublist]

    logging.info(
        f'{int(sum(tags_flatten))} positive class '
        f'of {len(tags_flatten)} labels ({np.round((sum(tags_flatten) / len(tags_flatten) * 100), 1)}%)'
    )

    return tokens, tags


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args_parser.add_argument('--mode', default='train', dest='mode')
    args = args_parser.parse_args()

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    data_raw_dir = fileDir + config['data']['raw']
    download_dataframe(args.mode, data_raw_dir)
