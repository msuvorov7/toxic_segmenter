import argparse
import logging
import os
import pickle
import random
import sys

import mlflow
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.model.model import ToxicSegmenter


fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

load_dotenv()
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(remote_server_uri)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def collate_fn(batch) -> dict:
    """
    Обработчик батча перед входом в модель.
    Забивает предложения pad-токенами до длинны самого длинного
    предложения в батче
    :param batch: батч данных
    :return:
    """
    max_len = max(len(row['feature']) for row in batch)

    feature = torch.empty((len(batch), max_len, 100), dtype=torch.float32)
    tag = torch.empty((len(batch), max_len), dtype=torch.long)

    for idx, row in enumerate(batch):
        to_pad = max_len - len(row['feature'])
        _feat = np.array(row['feature'])
        _tag = row['tag']
        feature[idx] = torch.cat((torch.tensor(_feat), torch.zeros((to_pad, 100))), axis=0)
        tag[idx] = torch.cat((torch.tensor(_tag), torch.zeros(to_pad)))
    return {
        'feature': feature,
        'tag': tag,
    }


def train(model: nn.Module,
          training_data_loader: DataLoader,
          validating_data_loader: DataLoader,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: str
          ) -> (float, float, float):
    """
    Функция для обучения модели на одной этохе
    :param model: модель
    :param training_data_loader: набор для обучения
    :param validating_data_loader: набор для валидации
    :param criterion: функция потерь
    :param optimizer: оптимизатор функции потерь
    :param device: обучение на gpu или cpu
    :return:
    """
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    for batch in tqdm(training_data_loader):
        text, label = batch['feature'], batch['tag']
        text = text.to(device)
        label = label.view(-1).to(device)

        y_predict = model(text)
        y_predict = y_predict.view(-1, y_predict.shape[2])
        loss = criterion(y_predict, label)
        optimizer.zero_grad()
        train_loss += loss.item()
        loss.backward()

        optimizer.step()

    train_loss /= len(training_data_loader)

    model.eval()
    y_true, y_pred, label_pred = [], [], []
    for batch in tqdm(validating_data_loader):

        text = batch['feature'].to(device)
        labels = batch['tag'].view(-1).to(device)

        prediction = model(text)
        prediction = prediction.view(-1, prediction.shape[2])
        label_predict = torch.argmax(prediction, dim=1).view(-1)
        preds = F.softmax(prediction, dim=1)[:, 1]

        y_true += labels.cpu().detach().numpy().ravel().tolist()
        y_pred += preds.cpu().detach().numpy().ravel().tolist()
        label_pred += label_predict.cpu().detach().numpy().ravel().tolist()

        loss = criterion(prediction, labels)

        val_loss += loss.item()

    val_loss /= len(validating_data_loader)

    # ba_scores = []
    # for th in np.linspace(0, 1, 100):
    #     a = (y_pred > th).astype(int)
    #     ba_scores.append(balanced_accuracy_score(y_true, a))
    # plt.figure(figsize=(16, 6))
    # plt.grid()
    # plt.plot(ba_scores)
    # plt.show()

    val_roc = roc_auc_score(y_true, y_pred)
    val_bac = balanced_accuracy_score(y_true, label_pred)
    logging.info(f'balanced accuracy: {val_bac}')
    mlflow.log_metric('val_bac', val_bac)

    th = 0.15
    val_bac = balanced_accuracy_score(y_true, (np.asarray(y_pred) > th).astype(int))
    logging.info(f'balanced accuracy threshold {th}: {val_bac}')

    return train_loss, val_loss, val_roc


def test(model: nn.Module,
         test_data_loader: DataLoader,
         device: str,
         ):

    model.eval()
    y_true, y_pred, label_pred = [], [], []
    for batch in tqdm(test_data_loader):
        text = batch['feature'].to(device)
        labels = batch['tag'].view(-1).to(device)

        prediction = model(text)
        prediction = prediction.view(-1, prediction.shape[2])
        label_predict = torch.argmax(prediction, dim=1).view(-1)
        preds = F.softmax(prediction, dim=1)[:, 1]

        y_true += labels.cpu().detach().numpy().ravel().tolist()
        y_pred += preds.cpu().detach().numpy().ravel().tolist()
        label_pred += label_predict.cpu().detach().numpy().ravel().tolist()

    test_roc = roc_auc_score(y_true, y_pred)
    logging.info(f'test roc auc: {test_roc}')
    test_bac = balanced_accuracy_score(y_true, label_pred)
    logging.info(f'balanced accuracy: {test_bac}')
    mlflow.log_metric('test_bac', test_bac)


def fit(model: nn.Module,
        training_data_loader: DataLoader,
        validating_data_loader: DataLoader,
        epochs: int
        ) -> (list, list):
    """
    Основной цикл обучения по эпохам
    :param model: модель
    :param training_data_loader: набор для обучения
    :param validating_data_loader: набор для валидации
    :param epochs: число эпох обучения
    :return:
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(np.array([0.12, 0.88]), dtype=torch.float32), reduction='mean')

    train_losses = []
    val_losses = []

    train_rocs = []
    val_rocs = []

    for epoch in range(1, epochs+1):
        train_loss, val_loss, val_roc = train(model, training_data_loader,
                                              validating_data_loader, criterion, optimizer, device)
        print()
        print('Epoch: {}, Training Loss: {}, Validation Loss: {}, ROC_AUC: {}'.format(epoch,
                                                                                      train_loss,
                                                                                      val_loss,
                                                                                      val_roc)
              )

        mlflow.log_metric('val_roc_auc', val_roc)
        mlflow.log_metric('val_loss', val_loss)
        mlflow.log_metric('train_loss', train_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        val_rocs.append(val_roc)

    return train_rocs, val_rocs


def save_onnx_model(model: nn.Module, directory_path: str) -> None:
    """
    функция для сохранения состояния onnx модели
    :param model: модель
    :param directory_path: имя директории
    :return:
    """
    model_path = directory_path + 'segmenter.onnx'
    dummy_input = torch.randn(10, 100)
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        input_names=['input'],
        dynamic_axes={"input": {0: "width"}})

    mlflow.onnx.log_model(artifact_path='models',
                          onnx_model=onnx.load(model_path),
                          registered_model_name='segmenter.onnx',
                          )

    logging.info(f'onnx model saved: {model_path}')


def save_torch_model(model: nn.Module, directory_path: str) -> None:
    """
    функция для сохранения состояния torch модели
    :param model: модель
    :param directory_path: имя директории
    :return:
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    model_path = directory_path + 'model.torch'
    torch.save(model, model_path)

    logging.info(f'torch model saved: {model_path}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args_parser.add_argument('--epoch', default=3, type=int, dest='epoch')
    args = args_parser.parse_args()

    assert args.epoch > 0

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    data_processed_dir = fileDir + config['data']['processed']

    with open(data_processed_dir + 'train_dataset.pkl', 'rb') as file:
        train_dataset = pickle.load(file)
    with open(data_processed_dir + 'test_dataset.pkl', 'rb') as file:
        test_dataset = pickle.load(file)

    logging.info(f'datasets loaded: {len(train_dataset)}, {len(test_dataset)}')

    train_size = len(train_dataset)
    validation_size = int(0.3 * train_size)

    train_data, valid_data = random_split(train_dataset, [train_size - validation_size, validation_size],
                                          generator=torch.Generator().manual_seed(42)
                                          )
    y = []
    for item in train_data:
        y += item['tag'].tolist()
    logging.info(f"class_weight: {compute_class_weight(class_weight='balanced', classes=np.unique(y), y=np.array(y))}")
    logging.info(f'mean: {np.mean(y)*100:.3f}%')

    train_loader = DataLoader(train_data, batch_size=32, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

    model = ToxicSegmenter(embedding_dim=100, hidden_size=256, output_dim=2)
    logging.info(f'model created')

    mlflow.set_experiment('base model')

    with mlflow.start_run(description='twitter corpus | augment 0.1 | tiny fasttext twitter\n gru + relu | weight [0.12, 0.88] | epoch 2'):
        logging.info(mlflow.get_artifact_uri())
        mlflow.log_param('train_data_len', len([item for sublist in train_dataset.tags for item in sublist]))
        mlflow.log_param('train_data_pos', sum([item for sublist in train_dataset.tags for item in sublist]))
        _, _ = fit(model, train_loader, valid_loader, args.epoch)
        test(model, test_loader, 'cpu')
        save_torch_model(model, fileDir + config['models'])
        save_onnx_model(model, fileDir + config['models'])
