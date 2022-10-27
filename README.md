# toxic_segmenter

Модель для поиска токсичной лексики в тексте

## Цель проекта
Повысить культуру речи в диалоговых переписках посредством нахождения оскорбительной речи.

## Финальный продукт
На данный момент минимальной задачей является успешный поиск и сокрытие нецензурных слов 
в сообщении 
с акцентом на все богатство словообразования русского языка и уловок отправителя (замена 
букв на латинские, повторения букв, замена на цифры и похожие знаки).

Развитем этой идеи хочется видеть модель, способную изменять стиль сообщения на более 
уважительный к собеседнику.

В качестве сервиса выбор пал на Телеграм-бота.

## Этапы разработки
Данный план - это не более чем набросок для структуризации получаемых знаний и будет 
модифицироваться при продвижении разработки проекта.

- Определение целей и задач проекта, его ограничений, доступных на момент разрабтки 
ресурсов, определение метрик оценивания полученного результата
- Выбор и подготовка обучающих данных
- Выбор модели ML, ее обучение, построение первичного baseline-решения
- Тюнинг модели с возможными откатами к предыдущим этапам для повышения финального 
качества
- Написание полного пайплайна работы модели в черновом варианте
- Перенос пайплайна в структурированный репозиторий с применением MLOps практик
- Развертывание модели, CI/CD процессы
- Инференс модели на платформе Телеграм, первичные тесты, проверка на 
отказоустойчивость
- Реализовать мониторинг модели для отследивания data-drift
- Автоматизировать по возможности пайплайн со своевременным переобучение модели 
и online-валидацией

## Отчет по работе

### 06/10
На данный момент определены цели проекта, составлен план действий и проведен 
первичный анализ в Jupyter Notebook. Должен сразу предупредить, что считаю 
этот инструмент наиболее подходящим на начальном этапе, поскольку он позволяет
с высокой скоростью проверять различные гипотезы, писать черновой код, который 
подвергнется многократной доработке или переделке. С версионирование такого 
подхода, конечно, мало что приятного выйдет, плюс время все же тратится на 
оформление блокнота.

Саму задачу можно свести к классификации токенов-слов на 2 класса. Датасет найден 
на просторах huggingface, но его разметка меня не вполне устраивает, многие на мой 
взгляд, сочентания не отмечены как оскорбителные, поэтому придется подумать над 
доразметкой. Архитектура модели для baseline-решения крайне проста: embedding -> LSTM -> Linear.


Уже сейчас сформировалиь идеи, которыми хочется воспользоваться при решении задачи.
Кажется, что может выстрелить идея с FastText-эмбедингами, поскольку есть потенциал 
справляться с умышленной "порчей" таргетных слов. Также хочется применить аугментации 
над текстом, чтобы подготовить модель к нестандартным паттернам написания слов.

### 13/10
В голову пришла идея попробовать подход с semi-supervised learning для доразметки тегов,
но навскидку ничего не вышло, нет хорошей кластерной структуры в 2d. Набросал stages для 
dvc, чтобы можно было основные моменты запускать для тестов.

Смотрел несколько статей по своей темматике. Народ пишет, что BERT показывает лучшие результаты,
но в срезе классификации. Полноценно задачу с поиском мало у кого встретишь. Я все же надеялся 
найти способ автоматической разметки, поскольку это пригодилось бы в последующем переобучении.
Заметил один хак, который можно использовать для обхода модели: можно использовать более изысканные 
словообразования (лучше совмещать с вполне чистой лексикой). Из-за этого модель на FastText выдает
наибольшее сходство с нетоксичными эмбеддингами. Припас для себя несколько таких кейсов для теста.

### 20/10
Решил пойти немного другим путем и сконцентироваться на построении базового пайплайна, чтобы создать
работающие интеграции между экспериментами, хранилищем и выкаткой модели. Хотелось бы сделать максимально
незвависимыми части с подготовкой данных (там есть идеи по доразметке, токенизации, аугментации), 
непосредственно разработкой и тестированием модели, реализацией конечного сервиса.

Удалось перевести пайплайн по обучению на рельсы dvc, написать простой [скрипт](./src/model/predict.py)
для проверки на своих данных, приспособить под проект mlflow + minio + postgres в виде docker-контейнеров
(но пока это в ветке `feature/mlops_stuff`). Пока не успел, но надо бы постепенно писать документацию и
инструкцию по развертке.

### 27/10
Нашел большой [датасет](https://www.kaggle.com/datasets/alexandersemiletov/toxic-russian-comments/code) и 
написал для него полу автомаическую разметку на основе поиска слов в словаре. Словарь перенес из таргетов
предыдущего датасета и добавил новых примеров. Часть слов добавлял через модель логистической регрессии:
обучил её классификации токенов на fasttext эмбеддингах и добавил топ слов с самым высоким скором. Планирую
в будущем подумать над оптимизацией процесса апдейта словаря.

Также написал и проверил скрипт для tg-бота