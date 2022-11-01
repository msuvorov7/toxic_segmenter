import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
from gensim.models import FastText

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.feature.preprocess_rules import Preprocessor
from src.data_load.create_dataframe import tokenize
from src.feature.build_feature import load_fasttext_model

fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def load_model(directory_path: str) -> nn.Module:
    """
    функция для загрузки модели
    :param directory_path: имя директории
    :return:
    """
    model_path = directory_path + 'model.torch'
    model = torch.load(model_path)
    return model


def predict(message: str) -> (list, list):
    tokens = tokenize(message)
    preprocessor = Preprocessor()
    cleaned_tokens = [preprocessor.forward(token) for token in tokens]
    encoded = [fasttext_model.wv[item] for item in cleaned_tokens]

    preds = F.softmax(model(torch.tensor(np.array(encoded))), dim=1)[:, 1].cpu().detach().numpy()

    return tokens, preds


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args = args_parser.parse_args()

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    model = load_model(fileDir + config['models'])
    logging.info(f'model loaded')

    fasttext_model = load_fasttext_model(fileDir + config['models'])
    logging.info(f'fasttext model loaded')

    text = 'пидрила злоебучий убери свою смазливую морду. собака конченная вот ты кто. знаю я породу этого хуеплета. мазь и словарь проверь'
    tokens, preds = predict(text)

    for token, pred in zip(tokens, preds):
        print(f'{token}: {pred:.2f}')
