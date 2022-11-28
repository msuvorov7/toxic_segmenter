import argparse
import logging
import os
import sys

import compress_fasttext
import numpy as np
import onnxruntime

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


def softmax(z):
    exp = np.exp(z - np.max(z))
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
    return exp


def load_model(directory_path: str):
    """
    функция для загрузки модели
    :param directory_path: имя директории
    :return:
    """
    return onnxruntime.InferenceSession(directory_path + 'segmenter.onnx')


def predict(message: str, fasttext_model, ort_session) -> (list, list):
    tokens = tokenize(message)
    preprocessor = Preprocessor()
    cleaned_tokens = [preprocessor.forward(token) for token in tokens]
    encoded = [fasttext_model[item] for item in cleaned_tokens]

    ort_inputs = {ort_session.get_inputs()[0].name: encoded}
    ort_outs = ort_session.run(None, ort_inputs)
    labels = np.argmax(ort_outs[0][0], axis=1)
    predictions = softmax(ort_outs[0][0])

    return tokens, predictions[:, 1]


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    # args_parser.add_argument('--fasttext_name', default='fasttext_pretrained.model', dest='fasttext_name')
    args_parser.add_argument('--fasttext_name', default='tiny_fasttext.model', dest='fasttext_name')
    args = args_parser.parse_args()

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    ort_session = load_model(fileDir + config['models'])
    logging.info(f'model loaded')

    # fasttext_model = load_fasttext_model(fileDir + config['models'] + args.fasttext_name)
    fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(fileDir + config['models'] + args.fasttext_name)
    logging.info(f'fasttext model loaded')

    texts = [
        'пидрила злоебучий убери свою смазливую морду.',
        'собака конченная вот ты кто.',
        'знаю я породу этих хуеплетов.',
        'копать не строить.',
        'мне кажется этот пидарок слишком драмматизирует.',
        'еб@нько прикрой, пидрк.',
        'пиздацирк какой-то.',
        'мазь и словарь проверь.',
    ]

    for text in texts:
        tokens, preds = predict(text, fasttext_model, ort_session)
        result_msg = ''
        for token, pred in zip(tokens, preds):
            if pred > 0.5:
                result_msg += f'🤬 {token} {pred:.2f} '
            else:
                result_msg += f'{token} {pred:.2f} '
        print(result_msg)
