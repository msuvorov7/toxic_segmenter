import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.model.model import ToxicSegmenter


fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def collate_fn(batch) -> dict:

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
    y_true, y_pred = [], []
    for batch in tqdm(validating_data_loader):

        text = batch['feature'].to(device)
        labels = batch['tag'].view(-1).to(device)

        prediction = model(text)
        prediction = prediction.view(-1, prediction.shape[2])
        preds = F.softmax(prediction, dim=1)[:, 1]
        y_true += labels.cpu().detach().numpy().ravel().tolist()
        y_pred += preds.cpu().detach().numpy().ravel().tolist()

        loss = criterion(prediction, labels)

        val_loss += loss.item()

    val_loss /= len(validating_data_loader)
    val_roc = roc_auc_score(y_true, y_pred)

    return train_loss, val_loss, val_roc


def fit(model: nn.Module,
        training_data_loader: DataLoader,
        validating_data_loader: DataLoader,
        epochs: int
        ) -> (list, list):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

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

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        val_rocs.append(val_roc)

    return train_rocs, val_rocs


def save_model(model: nn.Module, directory_path: str) -> None:
    """
    функция для сохранения состояния модели
    :param model: модель
    :param directory_path: имя директории
    :return:
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    model_path = directory_path + 'model.torch'
    torch.save(model, model_path)
    logging.info(f'model saved: {model_path}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml', dest='config')
    args_parser.add_argument('--epoch', default=5, type=int, dest='epoch')
    args = args_parser.parse_args()

    assert args.epoch > 0

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    data_processed_dir = fileDir + config['data']['processed']

    with open(data_processed_dir + 'train_dataset.pkl', 'rb') as file:
        train_dataset = pickle.load(file)
    with open(data_processed_dir + 'test_dataset.pkl', 'rb') as file:
        test_dataset = pickle.load(file)

    logging.info(f'datasets loaded')

    train_data, valid_data = random_split(train_dataset, [35_000, 10_000],
                                          generator=torch.Generator().manual_seed(42)
                                          )
    train_loader = DataLoader(train_data, batch_size=32, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

    model = ToxicSegmenter(embedding_dim=100, hidden_size=256, output_dim=2)
    logging.info(f'model created')

    _, _ = fit(model, train_loader, valid_loader, args.epoch)
    save_model(model, fileDir + config['models'])