# toxic_segmenter

Model for find and replacement toxic words for Russian Language

## Install
### Create .env file
- create `.env` file like `.env.example` in root dir
- create credentials for `minio` in `~/.aws/credentials` like that:
  ```
  [default]
  AWS_ACCESS_KEY_ID=username
  AWS_SECRET_ACCESS_KEY=password
  AWS_S3_BUCKET=arts
  ```
- set `AWS_*` keys in `.env` with credentials from `~/.aws/credentials`
- set `POSTGRES_*` keys like:
  ```
  POSTGRES_USER=root
  POSTGRES_PASSWORD=root
  POSTGRES_DB=test_db
  ```
- make sure that keys in `.env` and `docker-compose.yml` are equal

### Register postgres db in pgAdmin
- open 127.0.0.1:5050 in browser
- login with creds in docker-compose.yaml (admin@admin.com/root)
- `docker ps` and find `CONTAINER ID` for postgres
- `docker inspect <CONTAINER ID>` and find value for key `IPAddress`
- go back to browser and register Server (Servers -> Register -> Server -> Connection -> fill fields 
with `IPAddress`/`POSTGRES_USER`/`POSTGRES_PASSWORD`)

### Model Registry in MLflow
- open 127.0.0.1:9001 in browser
- login with creds in docker-compose.yaml (username/password)
- create bucket `arts` as in .env file (`AWS_S3_BUCKET`)

### MLFlow
- `docker build -f Docker/mlflow_image/Dockerfile -t mlflow_server .`


### Train Pipeline
Start docker containers with command `docker-compose up -d --build`. Container with name
`test_toxic_segmenter` will not start in first run because we didn't fit our model and didn't 
build this image.

For running dvc pipeline you need to get dataset from http://study.mokoron.com
in data/raw/ directory. In my repo this dataset called `twitter_corpus.csv`:

|text                                                                  |
|----------------------------------------------------------------------|
|Пропавшая в Хабаровске школьница почти сутки провела в яме у коллектор|
|"ЛЕНТА, Я СЕГОДНЯ ПОЛГОДА ДИРЕКШИОНЕЕЕЕР! С:                          |
|...                                                                   |

You also need a `toxic_vocabulary.csv`:

|word    |
|--------|
|отбросов|
|свинью  |
|дауна   |
|...     |


Run with command `dvc repro`. 

**Warning**: If you want to create and fit fasttext model, you should to make True `fit_fasttext` 
flag in [build_feature.py](./src/feature/build_feature.py)

### Start App
- `docker build --platform=linux/amd64 --pull --rm -f Docker/toxic_segmenter/Dockerfile -t test_toxic_segmenter:latest .`
- `docker-compose up -d --build`
- open 127.0.0.1:8000 in your browser and check results

### Deploy on Yandex Serverless Container
- [install](https://cloud.yandex.ru/docs/cli/quickstart) yandex cli:
`curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash`
- get `OAuth-token`
- run `yc init` and paste `OAuth-token`. Set other params with instruction-link before
- [create](https://cloud.yandex.ru/docs/iam/operations/sa/create) yandex server account (`yc iam service-account create --name tester`)
- `yc container registry create --name toxic-segmenter`
- `yc container registry configure-docker`
- `docker tag test_toxic_segmenter \cr.yandex/<registry_id>/test_toxic_segmenter:latest`
- `docker push \cr.yandex/<registry_id>/test_toxic_segmenter:latest`
- `yc serverless container create --name test-toxic-segmenter`
- ```
  yc serverless container revision deploy \
  --container-name test-toxic-segmenter \
  --image cr.yandex/<registry_id>/test_toxic_segmenter:latest \
  --cores 1 \
  --memory 1GB \
  --concurrency 1 \
  --execution-timeout 30s \
  --service-account-id <service_acc_id>
  ```

### Deploy on Yandex Serverless Functions (telegram bot)
- [install](https://cloud.yandex.ru/docs/cli/quickstart) yandex cli:
`curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash`
- get `OAuth-token`
- run `yc init` and paste `OAuth-token`. Set other params with instruction-link before
- [create](https://cloud.yandex.ru/docs/iam/operations/sa/create) yandex server account (`yc iam service-account create --name tester`)
- [create](https://cloud.yandex.ru/docs/storage/operations/buckets/create) s3 bucket `toxic-bucket` for zip archive
- [install](https://cloud.yandex.ru/docs/storage/tools/aws-cli) and configure `aws cli`
- create `zip` archive: `python src/telegram_bot/serverless_functions.py`
- upload on bucket:
  ```
  aws --endpoint-url=https://storage.yandexcloud.net/ \
      --profile yandex \
      s3 cp \
      servless_functions.zip \
      s3://toxic-bucket/
  ```
- create cloud function: `yc serverless function create --name=toxic-segmenter`
- make public invoke: `yc serverless function allow-unauthenticated-invoke toxic-segmenter`
- upload new version:
  ```
  yc serverless function version create \
    --function-name=toxic-segmenter \
    --runtime python39 \
    --entrypoint run.handler \
    --memory 1024m \
    --execution-timeout 3s \
    --package-bucket-name toxic-bucket \
    --package-object-name servless_functions.zip \
    --add-service-account id=<id>,alias=<alias> \
    --environment TELEGRAM_BOT_TOKEN=<tg-token>
  ```
- set webhook for telegram:
  - paste into browser: `https://api.telegram.org/bot<tg-token>/setWebHook?url=<toxic-segmenter-link>`
  - or make this with terminal:
       ```
       curl \
       --request POST \
       --url https://api.telegram.org/bot<tg-token>/setWebhook \
       --header 'content-type: application/json' \
       --data '{"url": "<toxic-segmenter-link>"}'
       ```

**Warning**: after the local launch of the telegram bot, you must re-install the webhook on Cloud Functions