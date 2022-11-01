import argparse
import logging
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
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


def save_fasttext_model(directory_path: str, embedding_model: FastText) -> None:
    """
    Функция для обучения модели FastText на наборе токенов
    :param directory_path: путь для сохранения обученной модели
    :param embedding_model: обученная модель
    :return:
    """
    with open(directory_path + 'fasttext.model', 'wb') as file:
        embedding_model.save(file)
    logging.info(f'fasttext_model saved in {directory_path}')


def load_fasttext_model(directory_path: str) -> FastText:
    """
    Функция для загрузки обученной модели
    :param directory_path: путь до директории с моделью
    :return:
    """
    vocabulary = FastText.load(directory_path + 'fasttext.model')
    return vocabulary


def download_dataframe(directory_path: str, mode: str) -> pd.DataFrame:
    """
    Загрузка датасета в память
    :param directory_path: путь до папки с сырыми данными
    :param mode: train/test
    :return:
    """
    dataframe = pd.read_parquet(directory_path + f'{mode}_df.parquet')
    is_toxic = dataframe['tags'].apply(lambda item: 1 if sum(item) > 0 else 0)

    logging.info(
        f'{int(sum(is_toxic))} positive class '
        f'of {len(is_toxic)} labels ({np.round((sum(is_toxic) / len(is_toxic) * 100), 1)}%)'
    )

    return dataframe


def preprocess_token(token: str) -> str:
    token = token.lower()
    pattern = r'[.,!?"()]'
    text = re.sub(pattern, '', string=token)

    emoji_pattern = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+"
    )
    return emoji_pattern.sub(r'', text)


def build_feature(dataframe: pd.DataFrame, embedding_model_path: str, mode: str) -> tuple:
    tokens = dataframe['raw_tokens']
    tags = dataframe['tags']

    cleaned_tokens = tokens.apply(lambda item: [preprocess_token(token) for token in item])

    if mode == 'train':
        embedding_model = FastText(cleaned_tokens, vector_size=300, min_n=4, window=4)
        # embedding_model.build_vocab(cleaned_tokens)
        # embedding_model.train(cleaned_tokens)
        save_fasttext_model(embedding_model=embedding_model, directory_path=embedding_model_path)
    else:
        embedding_model = load_fasttext_model(embedding_model_path)

    features = cleaned_tokens.apply(
        lambda sentence: np.array([embedding_model.wv[item] for item in sentence])
    )

    return features, tags


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
    embedding_model_dir = fileDir + config['models']
    df = download_dataframe(data_raw_dir, args.mode)
    features, tags = build_feature(df, embedding_model_dir, args.mode)

    dataset = ToxicDataset(features, tags)

    with open(data_processed_dir + f'{args.mode}_dataset.pkl', 'wb') as file:
        pickle.dump(dataset, file)

    logging.info(f'dataset saved in {data_processed_dir}')
