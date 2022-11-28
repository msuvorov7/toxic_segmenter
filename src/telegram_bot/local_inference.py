import logging
import os
import sys

import compress_fasttext
import yaml
from dotenv import load_dotenv

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.model.predict import load_model
from src.model.predict import predict

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

fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(fileDir + config['models'] + 'tiny_fasttext.model')
logging.info(f'fasttext model loaded')


def get_result_message(text: str) -> str:
    tokens, preds = predict(text, fasttext_model, model)
    toxic_smile = '🤬'
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
    result_message = get_result_message(message.caption)
    print(message.caption)
    await message.answer_photo(photo=message.photo[-1].file_id, caption=result_message)


@dp.message_handler(content_types=['text'])
async def parse_text(message: types.message):
    text = message.text
    print(text)
    result_message = get_result_message(text)
    await message.answer(text=result_message)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
