import argparse
import logging
import os
import pickle
import sys

import numpy as np
import yaml
from typing import Tuple, List

from gensim.models import FastText


sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.feature.dataset import ToxicDataset


fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def fit_fasttext_model(text: List[list], directory_path: str) -> FastText:
    vocabulary = FastText(text)
    with open(directory_path + 'fasttext.model', 'wb') as file:
        vocabulary.save(file)

    return vocabulary


def load_fasttext_model(directory_path: str) -> FastText:
    vocabulary = FastText.load(directory_path + 'fasttext.model')
    return vocabulary


def create_dataset(embedding_model: FastText,
                   text: List[list],
                   tags: List[list],
                   directory_path: str,
                   mode: str,
                   ) -> None:
    features = list(map(lambda sentence: list(map(lambda item: embedding_model.wv[item], sentence)), text))

    dataset = ToxicDataset(features, tags)

    with open(directory_path + f'{mode}_dataset.pkl', 'wb') as file:
        pickle.dump(dataset, file)


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

    assert args.mode in ('train', 'test')

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    data_raw_dir = fileDir + config['data']['raw']
    data_processed_dir = fileDir + config['data']['processed']
    tokens, tags = download_dataframe(args.mode, data_raw_dir)

    if args.mode == 'train':
        fasttext_model = fit_fasttext_model(tokens, fileDir + config['models'])
    else:
        fasttext_model = load_fasttext_model(fileDir + config['models'])

    create_dataset(embedding_model=fasttext_model,
                   text=tokens,
                   tags=tags,
                   directory_path=data_processed_dir,
                   mode=args.mode,
                   )
    logging.info(f'dataset saved in {data_processed_dir}')
