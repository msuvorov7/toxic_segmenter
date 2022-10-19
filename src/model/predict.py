import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
from gensim.models import FastText

from src.feature.build_feature import load_fasttext_model

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


class ContentWrapper:
    """
    Класс для обработки текста перед входом в модель
    """
    def __init__(self, text: str, embedding_model: FastText):
        self.text = text
        self.embedding_model = embedding_model

    def tokenize(self) -> list:
        return self.text.split()

    def encode(self, sentence: list) -> np.ndarray:
        return np.array([self.embedding_model.wv[item] for item in sentence])

    def transform(self) -> torch.Tensor:
        tokens = self.tokenize()
        encoded = self.encode(tokens)
        return torch.tensor(encoded)


def load_model(directory_path: str) -> nn.Module:
    """
    функция для загрузки модели
    :param directory_path: имя директории
    :return:
    """
    model_path = directory_path + 'model.torch'
    model = torch.load(model_path)
    return model


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

    text = 'paste your toxic message'
    content = ContentWrapper(text, fasttext_model)

    predicted = F.softmax(model(content.transform()), dim=1)[:, 1].cpu().detach().numpy()
    logging.info('answer is ready:')

    for token, pred in zip(content.tokenize(), predicted):
        print(f'{token}: {pred:.2f}')
