aws --endpoint-url=https://storage.yandexcloud.net/ [--profile YOUR_PROFILE_NAME] s3 cp servless_functions.zip s3://<bucket-name>/

yc serverless function version create \
  --function-name=<function-name> \
  --runtime python39 \
  --entrypoint run.handler \
  --memory 1024m \
  --execution-timeout 3s \
  --package-bucket-name <bucket-name> \
  --package-object-name servless_functions.zip \
  --add-service-account alias=<alias>,name=<name> \
  --environment TELEGRAM_BOT_TOKEN=<TOKEN>
