import datetime
import json
import logging
import os
import numpy as np
import compress_fasttext
import onnxruntime
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import ydb
import uuid

from preprocess_rules import Preprocessor

log = logging.getLogger(__name__)

# Create driver in global space.
driver = ydb.Driver(
    endpoint=os.environ.get('YDB_ENDPOINT'),
    database=os.environ.get('YDB_DATABASE')
)
# Wait for the driver to become active for requests.
driver.wait(fail_fast=True, timeout=5)
# Create the session pool instance to manage YDB sessions.
pool = ydb.SessionPool(driver)


def insert_query(session, id_key, message):
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    sql = f"""
    INSERT INTO `raw-tg-request`
    (
        id,
        message,
        ts
    )
    VALUES
    (
        "{id_key}",
        "{message}",
        DateTime("{ts}")
    );
    """
    # Create the transaction and execute query.
    return session.transaction().execute(
      sql,
      commit_tx=True,
      settings=ydb.BaseRequestSettings().with_timeout(3).with_operation_timeout(2)
    )


def tokenize(text: str) -> list:
    return text.split()


async def welcome_start(message):
    await message.answer('Hello')


async def predict(message: types.message):
    ort_session = onnxruntime.InferenceSession('segmenter.onnx')
    log.info(f'segmenter model loaded')

    fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
        'tiny_fasttext.model'
    )
    log.info(f'fasttext model loaded')

    msg = message.text
    tokens = tokenize(msg)
    preprocessor = Preprocessor()
    cleaned_tokens = [preprocessor.forward(token) for token in tokens]
    encoded = [fasttext_model[item] for item in cleaned_tokens]

    ort_inputs = {ort_session.get_inputs()[0].name: encoded}
    ort_outs = ort_session.run(None, ort_inputs)
    labels = np.argmax(ort_outs[0][0], axis=1)

    toxic_smile = '🤬'
    result_message = ''

    for token, pred in zip(tokens, labels):
        if pred > 0.5:
            result_message += f'{toxic_smile} '
            continue
        result_message += f'{token} '

    await message.answer(text=result_message)


# Functions for Yandex.Cloud
async def register_handlers(dp: Dispatcher):
    """Registration all handlers before processing update."""

    dp.register_message_handler(welcome_start, commands=['start'])
    dp.register_message_handler(predict, content_types=['text'])

    log.debug('Handlers are registered.')


async def process_event(event, dp: Dispatcher):
    """
    Converting an Yandex.Cloud functions event to an update and
    handling tha update.
    """

    update = json.loads(event['body'])
    log.debug('Update: ' + str(update))

    Bot.set_current(dp.bot)
    update = types.Update.to_object(update)
    await dp.process_update(update)


async def handler(event, context):
    """Yandex.Cloud functions handler."""

    if event['httpMethod'] == 'POST':
        id_key = str(uuid.uuid4())
        message = str(json.loads(event['body']))

        # YDB query
        session = driver.table_client.session().create()
        insert_query(session, id_key, message)

        # Bot and dispatcher initialization
        bot = Bot(os.environ.get('TELEGRAM_BOT_TOKEN'))
        dp = Dispatcher(bot)

        await register_handlers(dp)
        await process_event(event, dp)

        return {'statusCode': 200, 'body': 'ok'}
    return {'statusCode': 405}
