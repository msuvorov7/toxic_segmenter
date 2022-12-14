import argparse
import os
import sys
import zipfile

import yaml

fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))


def create_archive(models_path: str) -> None:
    """
    Создать архив для отправки в S3
    :param models_path: путь до моделей
    :return:
    """
    with zipfile.ZipFile('serverless_functions.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(models_path + 'tiny_fasttext.model', 'models/tiny_fasttext.model')
        zf.write(models_path + 'segmenter.onnx', 'models/segmenter.onnx')
        zf.write(fileDir + 'src/utils/preprocess_rules.py', 'src/utils/preprocess_rules.py')
        zf.write(fileDir + 'src/utils/tokenizer.py', 'src/utils/tokenizer.py')
        zf.write(fileDir + 'src/utils/transformer.py', 'src/utils/transformer.py')
        zf.write(fileDir + 'src/telegram_bot/run.py', 'run.py')
        zf.write(fileDir + 'src/telegram_bot/requirements.txt', 'requirements.txt')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args = args_parser.parse_args()

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    model_path = fileDir + config['models']
    create_archive(model_path)
