import argparse
import logging
import os
import pickle
import sys

import compress_fasttext
import numpy as np
import pandas as pd
import yaml

from gensim.models import FastText

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.feature.dataset import ToxicDataset
from src.utils.preprocess_rules import Preprocessor

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
    Функция для сохранения модели FastText
    :param directory_path: путь для сохранения обученной модели
    :param embedding_model: обученная модель
    :return:
    """
    with open(directory_path + 'fasttext.model', 'wb') as file:
        embedding_model.save(file)
    logging.info(f'fasttext_model saved in {directory_path}')


def download_dataframe(directory_path: str, mode: str) -> pd.DataFrame:
    """
    Загрузка датасета в память
    :param directory_path: путь до папки с сырыми данными
    :param mode: train/test
    :return: датафрейм
    """
    dataframe = pd.read_parquet(directory_path + f'{mode}_df.parquet')
    is_toxic = dataframe['tags'].apply(lambda item: 1 if sum(item) > 0 else 0)

    logging.info(
        f'{int(sum(is_toxic))} positive class '
        f'of {len(is_toxic)} labels ({np.round((sum(is_toxic) / len(is_toxic) * 100), 1)}%)'
    )

    return dataframe


def build_feature(dataframe: pd.DataFrame, embedding_model_path: str, fit_fasttext: bool) -> tuple:
    """
    Извлечение признаков для обучения модели.
    Принимается токенизированный и обработанный текст, затем каждый токен получает вектор из модели FastText
    :param dataframe: датафрейм для извлечения признаков
    :param embedding_model_path: путь для сохранения модели FastText
    :param fit_fasttext: флаг обучать ли с нуля FastText
    :return: признаки и метки
    """
    tokens = dataframe['raw_tokens']
    tags = dataframe['tags']

    preprocessor = Preprocessor()

    cleaned_tokens = tokens.apply(lambda item: [preprocessor.forward(token) for token in item])

    if fit_fasttext:
        logging.info('fitting FastText...')
        embedding_model = FastText(cleaned_tokens, vector_size=300, min_n=3, max_n=5, window=4).wv
        embedding_model = compress_fasttext.prune_ft_freq(embedding_model, pq=False)
        save_fasttext_model(embedding_model=embedding_model, directory_path=embedding_model_path)
    else:
        logging.info('loading FastText...')
        # source: http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html
        # how to compress source:
        #   model = FastText.load_fasttext_format('path/to/file/ft_native_300_ru_twitter_nltk_word_tokenize.bin')
        #   model.save(embedding_model_path + 'fasttext_pretrained.model')
        #   large_fasttext = FastText.load(embedding_model_path + 'fasttext_pretrained.model').wv
        #   tiny_fasttext = compress_fasttext.prune_ft_freq(large_fasttext, pq=False)
        #   tiny_fasttext.save(embedding_model_path + 'tiny_fasttext.model')
        embedding_model = compress_fasttext.CompressedFastTextKeyedVectors.load(embedding_model_path + 'tiny_fasttext.model')  # or 'fasttext.model'

    logging.info('get FastText embeddings...')
    features = cleaned_tokens.apply(
        lambda sentence: np.array([embedding_model[item] for item in sentence])
    )

    return features, tags


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args_parser.add_argument('--mode', default='train', dest='mode')
    args_parser.add_argument('--fit_fasttext', type=bool, default=False, dest='fit_fasttext')
    args = args_parser.parse_args()

    assert args.mode in ('train', 'test')

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    data_raw_dir = fileDir + config['data']['raw']
    data_processed_dir = fileDir + config['data']['processed']
    embedding_model_dir = fileDir + config['models']
    df = download_dataframe(data_raw_dir, args.mode)
    features, tags = build_feature(dataframe=df,
                                   embedding_model_path=embedding_model_dir,
                                   fit_fasttext=args.fit_fasttext,
                                   )

    dataset = ToxicDataset(features, tags)

    with open(data_processed_dir + f'{args.mode}_dataset.pkl', 'wb') as file:
        pickle.dump(dataset, file)

    logging.info(f'dataset saved in {data_processed_dir}')
