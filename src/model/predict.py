import argparse
import logging
import os
import sys

import compress_fasttext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.utils.preprocess_rules import Preprocessor
from src.data_load.create_dataframe import tokenize

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
    encoded = [fasttext_model[item] for item in cleaned_tokens]

    prediction = model(torch.tensor(np.array(encoded)))
    prediction = prediction.view(-1, prediction.shape[2])

    preds = F.softmax(prediction, dim=1)[:, 1].cpu().detach().numpy()

    return tokens, preds


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    # args_parser.add_argument('--fasttext_name', default='fasttext_pretrained.model', dest='fasttext_name')
    args_parser.add_argument('--fasttext_name', default='tiny_fasttext.model', dest='fasttext_name')
    args = args_parser.parse_args()

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    model = load_model(fileDir + config['models'])
    logging.info(f'model loaded')

    # fasttext_model = load_fasttext_model(fileDir + config['models'] + args.fasttext_name)
    fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(fileDir + config['models'] + args.fasttext_name)
    logging.info(f'fasttext model loaded')

    text = """
    пидрила злоебучий убери свою смазливую морду.
    собака конченная вот ты кто.
    знаю я породу этих хуеплетов.
    мазь и словарь проверь.
    копать не строить.
    мне кажется этот пидарок слишком драмматизирует.
    еб@нько прикрой, пидрк.
    пиздацирк какой-то.
    """
    tokens, preds = predict(text)

    for token, pred in zip(tokens, preds):
        if pred > 0.5:
            print(f'🤬 {token}: {pred:.2f}')
        else:
            print(f'{token}: {pred:.2f}')
