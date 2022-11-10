import logging
import os
import sys

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

import torch.nn.functional as F

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.data_load.create_dataframe import tokenize
from src.feature.build_feature import load_fasttext_model
from src.model.predict import load_model
from src.feature.preprocess_rules import Preprocessor

fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


with open(fileDir + 'params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)


model = load_model(fileDir + config['models'])
logging.info(f'model loaded')

fasttext_model = load_fasttext_model(fileDir + config['models'] + 'fasttext_pretrained.model')
logging.info(f'fasttext model loaded')


def predict(text: str) -> str:
    tokens = tokenize(text)
    preprocessor = Preprocessor()
    cleaned_tokens = [preprocessor.forward(token) for token in tokens]
    encoded = [fasttext_model.wv[item] for item in cleaned_tokens]

    toxic_smile = '🤬'
    log_message = ''
    prediction = model(torch.tensor(np.array(encoded)))
    prediction = prediction.view(-1, prediction.shape[2])

    preds = F.softmax(prediction, dim=1)[:, 1].cpu().detach().numpy()
    for token, pred, cl in zip(tokens, preds, cleaned_tokens):
        if pred > 0.5:
            log_message += toxic_smile
        log_message += f'{token}| {cl} |{pred:.3f}\n'

    print(log_message)

    result_message = ''

    for token, pred in zip(tokens, preds):
        if pred > 0.5:
            result_message += f'{toxic_smile} '
            continue
        result_message += f'{token} '

    return result_message


@dp.message_handler(commands=['start'])
async def welcome_start(message):
    await message.answer('Hello')


@dp.message_handler(lambda message: message.caption is not None, content_types=['photo'])
async def parse_photo(message: types.message):
    result_message = predict(message.caption)
    print(message.caption)
    await message.answer_photo(photo=message.photo[-1].file_id, caption=result_message)


@dp.message_handler(content_types=['text'])
async def parse_text(message: types.message):
    text = message.text
    print(text)
    result_message = predict(text)
    await message.answer(text=result_message)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
