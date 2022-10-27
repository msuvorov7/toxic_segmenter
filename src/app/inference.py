import logging
import os
import sys

import telebot
import yaml
from dotenv import load_dotenv

import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.feature.build_feature import preprocess_token
from src.data_load.create_dataframe import tokenize
from src.feature.build_feature import load_fasttext_model
from src.model.predict import load_model

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

bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def welcome_start(message):
    bot.send_message(message.chat.id, 'Hello')


with open(fileDir + 'params.yaml') as conf_file:
    config = yaml.safe_load(conf_file)


model = load_model(fileDir + config['models'])
logging.info(f'model loaded')

fasttext_model = load_fasttext_model(fileDir + config['models'])
logging.info(f'fasttext model loaded')


@bot.message_handler(content_types=['text'])
def predict(message):
    text = message.text
    print(text)
    tokens = tokenize(text)
    cleaned_tokens = [preprocess_token(token) for token in tokens]
    encoded = [fasttext_model.wv[item] for item in cleaned_tokens]

    toxic_smile = '🤬'
    result_message = ''

    preds = [model.predict_proba(encode)[:, 1][0].round(2) for encode in encoded]
    for token, pred in zip(tokens, preds):
        if pred > 0.5:
            result_message += toxic_smile
        result_message += f'{token}: {pred}\n'

    # bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
    bot.send_message(chat_id=message.chat.id, text=result_message)


bot.polling()

if __name__ == '__main__':
    print(TOKEN)
