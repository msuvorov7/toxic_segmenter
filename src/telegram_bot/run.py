import datetime
import json
import logging
import os
import compress_fasttext
import numpy as np
import onnxruntime
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
import ydb
import uuid

from src.utils.transformer import FeatureTransformer

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


def insert_request_query(session, id_key: str, message: str, ts: str):
    """Query for insert raw message to YDB."""

    sql = """
    DECLARE $ID AS String;
    DECLARE $MSG AS String;
    DECLARE $TS AS String;
    INSERT INTO `raw-tg-request`
    (
        id,
        message,
        ts
    )
    VALUES
    (
        $ID,
        $MSG,
        CAST($TS as DateTime)
    );
    """
    prepared_query = session.prepare(sql)

    # Create the transaction and execute query.
    return session.transaction().execute(
        prepared_query,
        {
            '$ID': id_key.encode(),
            '$MSG': message.encode(),
            '$TS': ts.encode(),
        },
        commit_tx=True,
        settings=ydb.BaseRequestSettings().with_timeout(3).with_operation_timeout(2)
    )


def insert_predicted_query(session, id_key: str, predicted: np.ndarray, ts: str):
    """Query to insert model prediction."""

    pass


async def welcome_start(message):
    """Aiogram helper handler."""

    await message.answer('Hello')


async def text_handler(message: types.message):
    """Aiogram handler only for text messages."""

    session = driver.table_client.session().create()

    username = message.from_user.username
    id_key = str(uuid.uuid4())
    msg = message.text

    # YDB insert request query
    request_ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    insert_request_query(session, id_key, msg, request_ts)

    # Predict Model
    fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('models/tiny_fasttext.model')
    model = onnxruntime.InferenceSession('models/segmenter.onnx')
    transformer = FeatureTransformer(fasttext_model, model)
    tokens = transformer.tokenizer.tokenize(msg)
    predictions = transformer.predict(msg)

    # YDB insert predicted query
    response_ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    insert_predicted_query(session, id_key, predictions, response_ts)

    toxic_smile = '🤬'
    result_message = f'@{username}:\n'
    threshold = 0.2

    if max(predictions) > threshold:
        spoiler_tokens = [tok if pred < threshold else f'<tg-spoiler>{tok}</tg-spoiler>' for tok, pred in zip(tokens, predictions)]
        result_message += transformer.tokenizer.detokenize(spoiler_tokens)

        await message.delete()
        await message.answer(text=result_message, parse_mode='HTML')


# Functions for Yandex.Cloud
async def register_handlers(dp: Dispatcher):
    """Registration all handlers before processing update."""

    dp.register_message_handler(welcome_start, commands=['start'])
    dp.register_message_handler(text_handler, content_types=['text'])

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

        # Bot and dispatcher initialization
        bot = Bot(os.environ.get('TELEGRAM_BOT_TOKEN'))
        dp = Dispatcher(bot)

        await register_handlers(dp)
        await process_event(event, dp)

        return {'statusCode': 200, 'body': 'ok'}
    return {'statusCode': 405}
