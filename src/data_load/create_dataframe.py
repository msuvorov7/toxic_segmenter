import argparse
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split

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


def tokenize(text: str) -> list:
    return text.split()


def clear_token(token: str) -> str:
    token = token.lower()
    pattern = r'[.,!?"()]'
    return re.sub(pattern, '', string=token)


def create_dataframe(directory_path: str, test_size: float):
    vocab = pd.read_csv(directory_path + 'toxic_vocabulary.csv')['word'].values

    with open(directory_path + "dataset.txt", "r") as file:
        dataset = file.read().split("\n")

    labels = list(map(lambda x: list(map(lambda y: y[9:], x.split(" ")[0].split(","))), dataset))[: -1]
    text = list(map(lambda x: " ".join(x.split(" ")[1:]), dataset))[: -1]

    df = pd.DataFrame(np.array([labels, text], dtype='object').T, columns=["label", "text"])

    tokens = df['text'].apply(lambda item: tokenize(item))
    cleaned_tokens = tokens.apply(lambda item: [clear_token(token) for token in item])
    tags = cleaned_tokens.apply(lambda item: [1 if token in vocab else 0 for token in item])

    dataframe = pd.DataFrame(np.array([tokens, tags], dtype='object').T, columns=['raw_tokens', 'tags'])
    is_toxic = dataframe['tags'].apply(lambda item: 1 if sum(item) > 0 else 0)

    logging.info(
        f'{int(sum(is_toxic))} positive class '
        f'of {len(is_toxic)} labels ({np.round((sum(is_toxic) / len(is_toxic) * 100), 1)}%)'
    )

    train_df, test_df, _, _ = train_test_split(dataframe,
                                               is_toxic,
                                               test_size=test_size,
                                               stratify=is_toxic,
                                               random_state=42,
                                               )

    train_df.to_parquet(directory_path + 'train_df.parquet', index=False)
    test_df.to_parquet(directory_path + 'test_df.parquet', index=False)


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
