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


@app.post('/predict')
def predict(msg=Form()):

    logging.info(f'message: {msg}')

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

    return HTMLResponse(content=f"<p>{result_message}</p>")


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "80")), log_level="info")
