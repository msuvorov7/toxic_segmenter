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
- release version:
  ```
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


### Yandex Serverless DataBase
- [create](https://cloud.yandex.ru/docs/cli/operations/authentication/service-account) token for 
service account `yc iam key create --service-account-name <service_acc_name> --output key.json --folder-id <ID_каталога>`
- install ydb `curl -sSL https://storage.yandexcloud.net/yandexcloud-ydb/install.sh | bash`
- [begin](https://ydb.tech/ru/docs/reference/ydb-sdk/example/python/?ysclid=lb87iuocqf671961291) of work
- you can connect to ydb with python code:
```python
import ydb
import ydb.iam
endpoint = 'grpcs://ydb.serverless.yandexcloud.net...'
database = '/ru-central1/...'
driver = ydb.Driver(
        endpoint=endpoint,
        database=database,
        # construct the service account credentials instance
        #   service account key should be in the local file,
        credentials=ydb.iam.ServiceAccountCredentials.from_file(
            '~/key.json',
        )
)
def execute_query(session):
    # Create the transaction and execute the query.
    # All transactions must be committed using the `commit_tx` flag in the last
    # statement. The either way to commit transaction is using `commit` method of `TxContext` object, which is
    # not recommended.
    return session.transaction().execute(
        "select * from `your-table-name`;",
        commit_tx=True,
        settings=ydb.BaseRequestSettings().with_timeout(3).with_operation_timeout(2),
    )
with driver:
    # wait until driver become initialized
    driver.wait(fail_fast=True, timeout=5)

    # Initialize the session pool instance and enter the context manager.
    # The context manager automatically stops the session pool.
    # On the session pool termination all YDB sessions are closed.
    with ydb.SessionPool(driver) as pool:

        # Execute the query with the `retry_operation_helper` the.
        # The `retry_operation_sync` helper used to help developers
        # to retry YDB specific errors like locks invalidation.
        # The first argument of the `retry_operation_sync` is a function to retry.
        # This function must have session as the first argument.
        result = pool.retry_operation_sync(execute_query)  # use result.rows to see rows
```