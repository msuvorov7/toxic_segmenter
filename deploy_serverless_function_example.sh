python src/telegram_bot/serverless_functions.py

aws --endpoint-url=https://storage.yandexcloud.net/ [--profile yandex] s3 cp serverless_functions.zip s3://<bucket-name>/

yc serverless function version create \
  --function-name=<function-name> \
  --runtime python39 \
  --entrypoint run.handler \
  --memory 1024m \
  --execution-timeout 3s \
  --package-bucket-name <bucket-name> \
  --package-object-name serverless_functions.zip \
  --service-account-id <service_acc_id> \
  --environment TELEGRAM_BOT_TOKEN=<tg_token>,YDB_ENDPOINT=<endpoint>,YDB_DATABASE=<database_url>
