docker build --platform=linux/amd64 --pull --rm -f Docker/toxic_segmenter/Dockerfile -t test_toxic_segmenter:latest .

docker tag test_toxic_segmenter cr.yandex/<registry_id>/test_toxic_segmenter:latest

docker push \cr.yandex/<registry_id>/test_toxic_segmenter:latest

yc serverless container revision deploy \
  --container-name test-toxic-segmenter \
  --image cr.yandex/<registry_id>/test_toxic_segmenter:latest \
  --cores 1 \
  --memory 256MB \
  --concurrency 1 \
  --execution-timeout 30s \
  --service-account-id <service_acc_id>