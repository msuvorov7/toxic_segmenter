import logging
import os
import sys

import compress_fasttext
import numpy as np
import onnxruntime
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.utils.preprocess_rules import Preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('models/tiny_fasttext.model')

ort_session = onnxruntime.InferenceSession('models/segmenter.onnx')


def tokenize(text: str) -> list:
    return text.split()


def softmax(z):
    exp = np.exp(z - np.max(z))
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
    return exp


@app.get('/about')
async def about(request: Request):
    logging.info('about')
    return templates.TemplateResponse("about.html", context={"request": request})


@app.post('/predict', response_class=HTMLResponse)
async def predict(request: Request, msg: str = Form(...)):

    logging.info(f'message: {msg}')

    tokens = tokenize(msg)
    preprocessor = Preprocessor()
    cleaned_tokens = [preprocessor.forward(token) for token in tokens]
    encoded = [fasttext_model[item] for item in cleaned_tokens]

    ort_inputs = {ort_session.get_inputs()[0].name: encoded}
    ort_outs = ort_session.run(None, ort_inputs)
    labels = np.argmax(ort_outs[0][0], axis=1)
    preds = softmax(ort_outs[0][0])[:, 1]

    toxic_smile = '🤬'
    result_message = ''

    for token, pred in zip(tokens, preds):
        if pred > 0.5:
            result_message += f'{toxic_smile} '
            continue
        result_message += f'{token} '

    debug_dict = {
        'tokens': tokens,
        'cleaned': cleaned_tokens,
        'nearest': [fasttext_model.most_similar(item)[0][0] for item in cleaned_tokens],
        'preds': preds.round(3),
        'labels': labels,
    }

    return templates.TemplateResponse("index.html",
                                      context={
                                          "request": request,
                                          "predicted": result_message,
                                          "request_sentence": msg,
                                          "debug_dict": debug_dict,
                                      }
                                      )


@app.post('/most_similar', response_class=HTMLResponse)
async def most_similar(request: Request, word: str = Form(...)):
    similar = fasttext_model.most_similar(word)[:5]
    logging.info(similar)
    return templates.TemplateResponse("index.html",
                                      context={
                                          "request": request,
                                          "similar": similar,
                                          "request_word": word,
                                      }
                                      )


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "80")), log_level="info")
