import argparse
import logging
import os
import pickle
import sys

from datasets import load_dataset
import yaml

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


def download_dataset(data_raw_directory: str) -> None:
    """
    Загрузка и сохранение датасета в raw данные
    :param data_raw_directory: путь до raw директории
    :return:
    """
    dataset = load_dataset("tesemnikov-av/toxic_dataset_ner")

    train_tokens = dataset['train']['tokens']
    train_tags = dataset['train']['tags']
    train_tags = list(
        map(lambda sentence: list(
                map(lambda item: 1 if item == 'TOXIC' else 0,
                    sentence)
            ),
            train_tags)
    )

    test_tokens = dataset['test']['tokens']
    test_tags = dataset['test']['tags']
    test_tags = list(
        map(lambda sentence: list(
            map(lambda item: 1 if item == 'TOXIC' else 0,
                sentence)),
            test_tags)
    )

    logging.info('datasets loaded')

    with open(data_raw_directory + 'train_tokens.pkl', 'wb') as file:
        pickle.dump(train_tokens, file)

    with open(data_raw_directory + 'test_tokens.pkl', 'wb') as file:
        pickle.dump(test_tokens, file)

    with open(data_raw_directory + 'train_tags.pkl', 'wb') as file:
        pickle.dump(train_tags, file)

    with open(data_raw_directory + 'test_tags.pkl', 'wb') as file:
        pickle.dump(test_tags, file)

    logging.info(f'datasets saved: {data_raw_directory}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args = args_parser.parse_args()

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    data_raw_dir = fileDir + config['data']['raw']
    download_dataset(data_raw_dir)
