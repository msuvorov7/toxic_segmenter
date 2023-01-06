import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.utils.augmentator import Augmentator
from src.utils.preprocess_rules import Preprocessor
from src.utils.tokenizer import Tokenizer

fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def augment(df: pd.DataFrame, probability: float) -> pd.DataFrame():
    """
    Применение аугментаций к датафрейму
    :param df: датафрейм для аугментаций
    :param probability: вероятность применения аугментации
    :return: аугментированный датафрейм
    """
    augmentator = Augmentator(probability=probability)

    df['raw_tokens'] = df['raw_tokens'].apply(lambda item: [augmentator.randomly_replace_to_latin(token) for token in item])
    df['raw_tokens'] = df['raw_tokens'].apply(lambda item: [augmentator.randomly_remove(token) for token in item])
    df['raw_tokens'] = df['raw_tokens'].apply(lambda item: [augmentator.randomly_noise(token) for token in item])
    df['raw_tokens'] = df['raw_tokens'].apply(lambda item: [augmentator.randomly_replace_to_latin(token) for token in item])

    return df


def create_dataframe(directory_path: str, test_size: float):
    """
    Сборка датафрейма. Неразмеченный текст сообщений зранится в файле twitter_corpus.csv.
    Словарь токсичных слов хранится в toxic_vocabulary.csv
    :param directory_path: путь до данных
    :param test_size: размер для тестового датасета
    :return:
    """
    vocab = pd.read_csv(directory_path + 'toxic_vocabulary.csv')['word'].values
    preprocessor = Preprocessor()
    tokenizer = Tokenizer()

    # ограничение на 1 млн записей для скорости обучения
    df = pd.read_csv(directory_path + 'twitter_corpus.csv', engine='python').sample(1_000_000, random_state=42)
    logging.info(f'dataset loaded: {df.shape}')

    df.to_csv(directory_path + 'twitter_sample.csv', index=False)

    tokens = df['text'].apply(str).apply(lambda item: tokenizer.tokenize(item))
    tags = tokens.apply(lambda item: [1 if preprocessor.forward(token) in vocab else 0 for token in item])
    logging.info('dataset tokenized')

    dataframe = pd.DataFrame(np.array([tokens, tags], dtype='object').T, columns=['raw_tokens', 'tags'])

    logging.info('starting augmentation...')
    dataframe = augment(dataframe, 0.1)
    logging.info('end augmentation')

    is_toxic = dataframe['tags'].apply(lambda item: 1 if sum(item) > 0 else 0)

    logging.info(
        f'{int(sum(is_toxic))} positive class '
        f'of {len(is_toxic)} labels ({np.round((sum(is_toxic) / len(is_toxic) * 100), 1)}%)'
    )

    flatten_tag = []
    for tags in dataframe['tags']:
        flatten_tag += tags
    logging.info(f'{sum(flatten_tag)} toxic of {len(flatten_tag)}')

    train_df, test_df, _, _ = train_test_split(dataframe,
                                               is_toxic,
                                               test_size=test_size,
                                               stratify=is_toxic,
                                               random_state=42,
                                               )

    train_df.to_parquet(directory_path + 'train_df.parquet', index=False)
    test_df.to_parquet(directory_path + 'test_df.parquet', index=False)
    logging.info('train/test splits created')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args_parser.add_argument('--test_size', default=0.4, dest='test_size', type=float)
    args = args_parser.parse_args()

    assert args.test_size < 1
    assert args.test_size > 0

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    data_raw_dir = fileDir + config['data']['raw']
    create_dataframe(data_raw_dir, args.test_size)
