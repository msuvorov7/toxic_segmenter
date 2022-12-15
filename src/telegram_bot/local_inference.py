import logging
import os
import sys

import compress_fasttext
import onnxruntime

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.utils.transformer import FeatureTransformer

fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


def get_result_message(text: str) -> str:
    transformer = FeatureTransformer(fasttext_model, model)
    tokens = transformer.tokenizer.tokenize(text)
    probabilities = transformer.predict(text)

    threshold = 0.2
    toxic_smile = '🤬'
    censored_tokens = [tok if prob < threshold else toxic_smile for (tok, prob) in zip(tokens, probabilities)]

    return transformer.tokenizer.detokenize(censored_tokens)


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
    model = onnxruntime.InferenceSession('../models/segmenter.onnx')
    logging.info(f'model loaded')

    fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('../models/tiny_fasttext.model')
    logging.info(f'fasttext model loaded')

    executor.start_polling(dp, skip_updates=True)
