FROM python:3.9-slim

WORKDIR /code

RUN pip install --upgrade pip

COPY ./Docker/toxic_segmenter/requirements.txt /code/
RUN pip install -r requirements.txt

COPY ./src/app /code/app
COPY ./src/utils/ /code/app/src/utils/
COPY ./models/tiny_fasttext.model /code/app/models/
COPY ./models/segmenter.onnx /code/app/models/

WORKDIR app/

CMD ["python3", "run.py", "--fasttext_model", "models/tiny_fasttext.model", "--model", "models/segmenter.onnx"]

# docker build --platform=linux/amd64 --pull --rm -f Docker/toxic_segmenter/Dockerfile -t test_toxic_segmenter:latest .