import argparse
import logging
import os
import sys

import compress_fasttext
import onnxruntime
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.utils.transformer import FeatureTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get('/about')
async def about(request: Request):
    logging.info('about')
    return templates.TemplateResponse("about.html", context={"request": request})


@app.post('/predict', response_class=HTMLResponse)
async def predict(request: Request, msg: str = Form(...)):

    logging.info(f'message: {msg}')

    toxic_smile = '🤬'
    threshold = 0.2

    transformer = FeatureTransformer(fasttext_model, segment_model)
    tokens = transformer.tokenizer.tokenize(msg)
    processed_tokens = [transformer.preprocessor.forward(token) for token in tokens]
    probabilities = transformer.predict(msg)

    censored_tokens = [tok if prob < threshold else toxic_smile for (tok, prob) in zip(tokens, probabilities)]
    predicted = transformer.tokenizer.detokenize(censored_tokens)

    debug_dict = {
        'tokens': tokens,
        'processed': processed_tokens,
        'nearest': [fasttext_model.most_similar(item)[0][0] for item in processed_tokens],
        'preds': probabilities.round(3),
        'labels': (probabilities > threshold).astype(int),
    }

    return templates.TemplateResponse("index.html",
                                      context={
                                          "request": request,
                                          "predicted": predicted,
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
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--fasttext_model', dest='fasttext_model', required=True)
    args_parser.add_argument('--model', dest='model', required=True)
    args = args_parser.parse_args()

    fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(args.fasttext_model)
    segment_model = onnxruntime.InferenceSession(args.model)

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "80")), log_level="info")
